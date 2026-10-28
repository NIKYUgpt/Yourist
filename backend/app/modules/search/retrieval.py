"""Поиск норм на дату: полнотекстовый + векторный, объединённые по рангам (RRF).

Ищем только среди версий статей, действовавших на заданную дату: для каждого
отслеживаемого акта берётся редакция на эту дату, фрагменты фильтруются по её
сопоставлению. Явные ссылки в вопросе («ст. 81») подтягивают статью напрямую.
"""

import logging
import math
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from sqlalchemy import bindparam, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.ai.providers.base import EmbedKind, ProviderError, Timer
from app.modules.ai.service import AIGateway
from app.modules.corpus import queries
from app.modules.corpus.models import Act, ArticleVersion, RedactionArticle
from app.modules.corpus.titles import short_title_for
from app.modules.search.models import Chunk

logger = logging.getLogger(__name__)

RRF_K = 60
CANDIDATES = 40
EXPLICIT_BONUS = 1.0  # явно названная статья всегда выше найденных по смыслу
# сколько лучших результатов каждого способа попадает в выдачу в любом случае
VECTOR_HEADS = 2
LEXICAL_HEADS = 0  # лучший полнотекстовый результат часто случаен — без гарантии
PREFIX_CHARS = 6
EXPAND_FOR_VECTORS = True

_WORD = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)
# «ст. 81», «статья 19.1», «статьи 341.1-1», «ст.ст. 81»
_ARTICLE_REF = re.compile(
    r"(?:\bст\.\s*(?:ст\.\s*)?|\bстать[яиеюйёх]\w*\s+)(\d+(?:[.-]\d+)*)", re.IGNORECASE
)
_STOPWORDS = frozenset(
    (
        "что",
        "как",
        "если",
        "или",
        "для",
        "при",
        "без",
        "это",
        "мой",
        "моя",
        "мое",
        "мои",
        "меня",
        "мне",
        "его",
        "она",
        "они",
        "так",
        "все",
        "был",
        "была",
        "были",
        "есть",
        "можно",
        "нужно",
        "ли",
        "чем",
        "кто",
        "где",
        "когда",
        "какой",
        "какая",
        "какие",
        "какое",
        "почему",
        "зачем",
        "мы",
        "вы",
        "вам",
        "нас",
        "наш",
        "ваш",
        "уже",
        "еще",
        "ещё",
        "только",
        "тоже",
        "также",
        "этот",
        "эта",
        "эти",
        "тот",
        "там",
        "тут",
        "сейчас",
    )
)


@dataclass
class Hit:
    chunk_id: int
    act_id: int
    act: str
    key: str
    heading: str
    refs: list[Any]
    context: str
    text: str
    redaction_id: int
    valid_from: date
    score: float
    lexical_rank: int | None = None
    vector_rank: int | None = None
    explicit: bool = False


@dataclass
class RedactionUsed:
    act_id: int
    act: str
    redaction_id: int
    valid_from: date


@dataclass
class SearchResult:
    query: str
    on: date
    hits: list[Hit]
    redactions: list[RedactionUsed]
    space: str | None = None
    notes: list[str] = field(default_factory=list)  # почему поиск неполный
    timings_ms: dict[str, int] = field(default_factory=dict)


def explicit_article_keys(query: str) -> list[str]:
    keys: list[str] = []
    for match in _ARTICLE_REF.finditer(query):
        if match.group(1) not in keys:
            keys.append(match.group(1))
    return keys


# Бытовые слова → термины закона. Пользователь пишет «гендиректор», «больничный»,
# «декрет», а в кодексе — «руководитель организации», «временная нетрудоспособность»,
# «отпуск по уходу за ребенком»; без перевода поиск такие нормы пропускает
# (evaluation/results/2026-10-27-labor-hard: вопрос про увольнение директора).
_LEGAL_TERMS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(p, re.IGNORECASE), terms)
    for p, terms in [
        (
            r"ген\.?\s*директор|генеральн\w*\s+директор|директор\w*\s+(организаци|компани|фирм)"
            r"|руководител\w*\s+(компани|фирм)",
            "руководитель организации",
        ),
        (r"больничн", "временная нетрудоспособность"),
        (r"декрет", "отпуск по беременности и родам отпуск по уходу за ребенком"),
        (r"зарплат|\bзп\b", "заработная плата"),
        (r"отпускны", "оплата отпуска"),
        (r"сокращени|сократ", "сокращение численности или штата работников"),
        (
            r"уволит\w*ся|увольня\w*сь|по\s+собственному|уйти\s+раньше|хочу\s+уйти",
            "расторжение трудового договора по инициативе работника",
        ),
        (r"переработк", "сверхурочная работа"),
        (r"испытательн\w*\s+срок", "испытание при приеме на работу"),
        (r"за\s+свой\s+сч[её]т", "отпуск без сохранения заработной платы"),
        (r"арми|призыв", "призыв на военную службу"),
        (r"пьян", "состояние алкогольного опьянения"),
        (r"командировк", "служебная командировка"),
        (
            r"подрост|несовершеннолетн|младше\s+(16|18)|\b1[4-7]\s*(лет|год)",
            "работники в возрасте до восемнадцати лет",
        ),
        (
            r"задерж\w*\s+(зарплат|выплат)|компенсаци\w*\s+за\s+задержк",
            "нарушение сроков выплаты заработной платы денежная компенсация",
        ),
        # «куда обращаться», «можно ли оспорить» — без этого сроки обращения в суд
        # (ст. 392) в выдачу не попадали (два составных вопроса, 28.10.2026)
        (
            r"оспор|обжал|куда\s+(обращ|идти|жалова|пожалова)|к\s+кому\s+(обращ|идти)"
            r"|\bв\s+суд\b|подать\s+(иск|в\s+суд)",
            "сроки обращения в суд за разрешением индивидуального трудового спора",
        ),
    ]
]


def expand_query(query: str) -> str:
    """Вопрос, дополненный терминами закона для найденных бытовых слов."""
    extra = [terms for pattern, terms in _LEGAL_TERMS if pattern.search(query)]
    return " ".join([query, *dict.fromkeys(extra)])


def query_words(query: str, limit: int = 24) -> list[str]:
    words: list[str] = []
    for word in _WORD.findall(query.lower()):
        if len(word) >= 3 and word not in _STOPWORDS and word not in words:
            words.append(word)
    return words[:limit]


async def _redactions_on(session: AsyncSession, on: date) -> list[RedactionUsed]:
    used = []
    for act in await session.scalars(select(Act).where(Act.is_tracked).order_by(Act.id)):
        red = await queries.redaction_on(session, act.id, on)
        if red is not None:
            title = act.short_title or short_title_for(act.title) or act.title
            used.append(RedactionUsed(act.id, title, red.id, red.valid_from))
    return used


_SCOPE = """
    FROM search.chunks c
    JOIN corpus.redaction_articles ra ON ra.version_id = c.version_id
    WHERE ra.redaction_id IN :rids
"""


def lexical_term(word: str) -> str:
    """Длинное слово — по началу: «испытательный» и «испытания» дают разные основы,
    а «испыта:*» находит обе. В слове только буквы и цифры — синтаксис tsquery не сломать.
    """
    return f"{word[:PREFIX_CHARS]}:*" if len(word) > PREFIX_CHARS + 1 else word


def _match(i: int) -> str:
    return f"c.tsv @@ to_tsquery('russian', :w{i})"


async def _lexical(session: AsyncSession, words: list[str], rids: list[int], n: int) -> list[int]:
    """Полнотекстовый поиск с весом редких слов (IDF, как в BM25).

    В обычном вопросе не все слова встретятся в норме, поэтому ищем по любому слову,
    а ранжируем по сумме IDF совпавших: «прогул» весит больше, чем «работа».
    """
    if not words:
        return []
    params: dict[str, Any] = {f"w{i}": lexical_term(w) for i, w in enumerate(words)}
    params["rids"] = rids
    counts = ", ".join(f"count(*) FILTER (WHERE {_match(i)})" for i in range(len(words)))
    df_sql = text(f"SELECT count(*), {counts} {_SCOPE}").bindparams(
        bindparam("rids", expanding=True)
    )
    total, *dfs = (await session.execute(df_sql, params)).one()
    idf = {i: math.log((total - df + 0.5) / (df + 0.5) + 1) for i, df in enumerate(dfs) if df}
    if not idf:
        return []
    score = " + ".join(f"{w:.4f} * ({_match(i)})::int" for i, w in idf.items())
    any_match = " OR ".join(_match(i) for i in idf)
    sql = text(
        f"SELECT c.id {_SCOPE} AND ({any_match}) ORDER BY ({score}) DESC, c.id LIMIT :n"
    ).bindparams(bindparam("rids", expanding=True))
    return list((await session.execute(sql, {**params, "n": n})).scalars())


_VECTOR_SQL = text(
    """
    SELECT c.id
    FROM search.embeddings e
    JOIN search.chunks c ON c.id = e.chunk_id
    JOIN corpus.redaction_articles ra ON ra.version_id = c.version_id
    WHERE e.space = :space AND ra.redaction_id IN :rids
    ORDER BY e.vector <=> CAST(:qv AS vector), c.id
    LIMIT :n
    """
).bindparams(bindparam("rids", expanding=True))

_COVERAGE_SQL = text(
    """
    SELECT count(c.id), count(e.chunk_id)
    FROM search.chunks c
    JOIN corpus.redaction_articles ra ON ra.version_id = c.version_id
    LEFT JOIN search.embeddings e ON e.chunk_id = c.id AND e.space = :space
    WHERE ra.redaction_id IN :rids
    """
).bindparams(bindparam("rids", expanding=True))


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{v:.7g}" for v in vector) + "]"


async def search(
    session: AsyncSession,
    ai: AIGateway | None,
    query: str,
    on: date,
    *,
    k: int = 8,
    per_article: int = 2,
    explicit: bool = True,
) -> SearchResult:
    """explicit=False — не поднимать названные в запросе статьи: для запросов,
    составленных моделью, а не пользователем (номер статьи модель может выдумать)."""
    result = SearchResult(query=query, on=on, hits=[], redactions=[])
    result.redactions = await _redactions_on(session, on)
    if not result.redactions:
        result.notes.append(f"нет разобранных редакций, действовавших на {on:%d.%m.%Y}")
        return result
    rids = [r.redaction_id for r in result.redactions]
    by_redaction = {r.redaction_id: r for r in result.redactions}
    scores: dict[int, float] = {}
    ranks: dict[int, dict[str, int]] = {}

    def add(ids: list[int], source: str, weight: float = 1.0) -> None:
        for rank, cid in enumerate(ids, 1):
            scores[cid] = scores.get(cid, 0.0) + weight / (RRF_K + rank)
            ranks.setdefault(cid, {})[source] = rank

    timer = Timer()
    explicit_keys = explicit_article_keys(query) if explicit else []
    explicit_ids: set[int] = set()
    if explicit_keys:
        rows = await session.execute(
            select(Chunk.id)
            .join(RedactionArticle, RedactionArticle.version_id == Chunk.version_id)
            .where(RedactionArticle.redaction_id.in_(rids), RedactionArticle.key.in_(explicit_keys))
            .order_by(Chunk.ord)
        )
        explicit_ids = set(rows.scalars())
        for cid in explicit_ids:
            scores[cid] = scores.get(cid, 0.0) + EXPLICIT_BONUS

    expanded = expand_query(query)
    lexical_ids = await _lexical(session, query_words(expanded), rids, CANDIDATES)
    add(lexical_ids, "lexical")
    result.timings_ms["lexical"] = timer.ms
    vector_ids: list[int] = []

    if ai is not None:
        result.space = space = ai.embedding_key
        timer = Timer()
        try:
            [qvec] = await ai.embed([expanded if EXPAND_FOR_VECTORS else query], EmbedKind.QUERY)
        except ProviderError as exc:
            logger.warning("Векторный поиск недоступен: %s", exc)
            result.notes.append(f"векторный поиск недоступен: {exc}")
        else:
            result.timings_ms["embed"] = timer.ms
            timer = Timer()
            total, embedded = (
                await session.execute(_COVERAGE_SQL, {"space": space, "rids": rids})
            ).one()
            if embedded < total:
                result.notes.append(
                    f"векторы готовы для {embedded} из {total} фрагментов ({space})"
                )
            if embedded:
                params = {
                    "space": space,
                    "rids": rids,
                    "qv": _vector_literal(qvec),
                    "n": CANDIDATES,
                }
                vector_ids = list((await session.execute(_VECTOR_SQL, params)).scalars())
                add(vector_ids, "vector")
            result.timings_ms["vector"] = timer.ms

    if not scores:
        return result

    by_score = sorted(scores, key=lambda cid: (-scores[cid], cid))
    # RRF поднимает то, что нашли оба способа; но лучший результат одного способа
    # (например, векторного, когда слова вопроса и нормы разные) терять нельзя
    heads = vector_ids[:VECTOR_HEADS] + lexical_ids[:LEXICAL_HEADS]
    ordered = list(
        dict.fromkeys(
            [c for c in by_score if c in explicit_ids]
            + [c for c in heads if c not in explicit_ids]
            + by_score
        )
    )
    rows = await session.execute(
        select(
            Chunk,
            ArticleVersion.key,
            ArticleVersion.heading,
            ArticleVersion.act_id,
            RedactionArticle.redaction_id,
        )
        .join(ArticleVersion, ArticleVersion.id == Chunk.version_id)
        .join(RedactionArticle, RedactionArticle.version_id == Chunk.version_id)
        .where(Chunk.id.in_(ordered[: CANDIDATES * 2]), RedactionArticle.redaction_id.in_(rids))
    )
    details = {
        chunk.id: (chunk, key, heading, act_id, rid) for chunk, key, heading, act_id, rid in rows
    }

    per_key: dict[tuple[int, str], int] = {}
    # явно названной статье — до половины выдачи (самые близкие к вопросу её фрагменты),
    # остальным статьям — не больше per_article фрагментов
    explicit_limit = max(per_article, k // 2)
    for cid in ordered:
        if cid not in details:
            continue
        chunk, key, heading, act_id, rid = details[cid]
        taken = per_key.get((act_id, key), 0)
        if taken >= (explicit_limit if cid in explicit_ids else per_article):
            continue
        per_key[(act_id, key)] = taken + 1
        red = by_redaction[rid]
        result.hits.append(
            Hit(
                chunk_id=cid,
                act_id=act_id,
                act=red.act,
                key=key,
                heading=heading,
                refs=chunk.refs,
                context=chunk.context,
                text=chunk.text,
                redaction_id=rid,
                valid_from=red.valid_from,
                score=round(scores[cid], 6),
                lexical_rank=ranks.get(cid, {}).get("lexical"),
                vector_rank=ranks.get(cid, {}).get("vector"),
                explicit=cid in explicit_ids,
            )
        )
        if len(result.hits) >= k:
            break
    return result


async def search_issues(
    session: AsyncSession,
    ai: AIGateway | None,
    question: str,
    queries: list[str],
    on: date,
    *,
    k_whole: int = 4,
    k_issue: int = 3,
    cap: int = 12,
) -> tuple[SearchResult, list[list[int]]]:
    """Поиск для вопроса из нескольких проблем: часть выдачи — по вопросу целиком,
    остальное — по каждой проблеме отдельно, без повторов.

    Возвращает общую выдачу и для каждой проблемы — номера её источников (с 1)
    в этой выдаче. Ограничение cap держит промпт в контексте локальных моделей
    (8192 токена у qwen3:8b вместе с ответом).
    """
    whole = await search(session, ai, question, on, k=k_whole)
    found = [await search(session, ai, q, on, k=k_issue * 3, explicit=False) for q in queries]
    hits = list(whole.hits)
    index = {h.chunk_id: i for i, h in enumerate(hits)}
    linked: list[list[int]] = [[] for _ in queries]
    pointers = [0] * len(queries)
    # по кругу: сначала лучший фрагмент каждой проблемы, затем второй и т. д. —
    # чтобы при ограничении выдачи ни одна проблема не осталась без норм
    for _ in range(k_issue):
        for qi, result in enumerate(found):
            while pointers[qi] < len(result.hits):
                hit = result.hits[pointers[qi]]
                pointers[qi] += 1
                if hit.chunk_id in index:
                    if index[hit.chunk_id] not in linked[qi]:
                        linked[qi].append(index[hit.chunk_id])
                        break
                    continue
                if len(hits) >= cap:
                    continue  # места нет — ищем среди уже взятых
                index[hit.chunk_id] = len(hits)
                hits.append(hit)
                linked[qi].append(index[hit.chunk_id])
                break

    merged = SearchResult(
        query=question, on=on, hits=hits, redactions=whole.redactions, space=whole.space
    )
    for result in [whole, *found]:
        merged.notes.extend(n for n in result.notes if n not in merged.notes)
        for name, ms in result.timings_ms.items():
            merged.timings_ms[name] = merged.timings_ms.get(name, 0) + ms
    return merged, [[n + 1 for n in sorted(numbers)] for numbers in linked]
