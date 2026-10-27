"""Детерминированная проверка ссылок в ответе модели.

Без нейросетей: регулярными выражениями находим в ответе ссылки на источники ([2])
и на нормы («п. 6 ч. 1 ст. 81 ТК РФ») и сверяем их с корпусом на дату вопроса:
- статья есть в редакции и не утратила силу (иначе — ошибка);
- названные часть и пункт в статье есть (иначе — предупреждение);
- статья была среди найденных источников (иначе — предупреждение: модель
  сослалась на норму «по памяти», текст ей не показывали);
- цитаты в «кавычках» дословно есть в источниках (иначе — предупреждение);
- известна ли будущая редакция статьи (тогда — пометка с датой изменения).
"""

import re
from dataclasses import dataclass, field
from datetime import date

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.corpus import queries
from app.modules.corpus.models import Act, ArticleVersion, RedactionArticle
from app.modules.corpus.titles import short_title_for
from app.modules.search.chunking import cite
from app.modules.search.retrieval import Hit

OK, INFO, WARNING, ERROR = "ok", "info", "warning", "error"

_ORDINALS = {
    "перв": "1",
    "втор": "2",
    "трет": "3",
    "четверт": "4",
    "пят": "5",
    "шест": "6",
    "седьм": "7",
    "восьм": "8",
    "девят": "9",
    "десят": "10",
}
_ORDINAL = r"(?:перв|втор|трет|четверт|пят|шест|седьм|восьм|девят|десят)\w*"
_KEY = r"\d+(?:\.\d+)*(?:-\d+)?"

_ARTICLE = re.compile(
    rf"(?<![\w])(?:ст\.\s*(?:ст\.\s*)?|стать(?:я|и|е|ю|ей|ям|ями|ях)\s+)"
    rf"({_KEY}(?:\s*(?:,|и)\s*{_KEY})*)",
    re.IGNORECASE,
)
# единицы статьи перед словом «статья»: «подп. «а» п. 6 ч. 1 ст. 81»
_UNITS_BEFORE = re.compile(
    r"(?:(?:подп\.|пп\.|подпункт\w*)\s*[«\"]?([а-я])[»\"]?\s*,?\s*)?"
    r"(?:(?:п\.|пункт\w*)\s*(\d+(?:\.\d+)?)\s*,?\s*)?"
    rf"(?:(?:ч\.|част(?:ь|и|ью|ям))\s*(\d+|{_ORDINAL})\s*,?\s*)?$",
    re.IGNORECASE,
)
_ACT_AFTER = re.compile(
    r"^\s*\)?,?\s*("
    r"ТК(?:\s*РФ)?|Трудов\w*\s+кодекс\w*(?:\s+(?:РФ|Российской\s+Федерации))?|"
    r"ГК(?:\s*РФ)?|Гражданск\w*\s+кодекс\w*(?:\s+(?:РФ|Российской\s+Федерации))?|"
    r"НК(?:\s*РФ)?|Налогов\w*\s+кодекс\w*(?:\s+(?:РФ|Российской\s+Федерации))?|"
    r"КоАП(?:\s*РФ)?|УК\s*РФ|ЖК\s*РФ|СК\s*РФ|ГПК(?:\s*РФ)?|АПК(?:\s*РФ)?|"
    r"(?:Федеральн\w+\s+)?закон\w*[^;\n]{0,100}?\d+-ФЗ"
    r")",
    re.IGNORECASE,
)
_ACT_ALIASES = [
    (re.compile(r"^(ТК|Трудов)", re.I), "ТК РФ"),
    (re.compile(r"^(ГК|Гражданск)", re.I), "ГК РФ"),
    (re.compile(r"^(НК|Налогов)", re.I), "НК РФ"),
    (re.compile(r"^КоАП", re.I), "КоАП РФ"),
    (re.compile(r"^УК", re.I), "УК РФ"),
    (re.compile(r"^ЖК", re.I), "ЖК РФ"),
    (re.compile(r"^СК", re.I), "СК РФ"),
    (re.compile(r"^ГПК", re.I), "ГПК РФ"),
    (re.compile(r"^АПК", re.I), "АПК РФ"),
]
_LAW_NUMBER = re.compile(r"(\d+-ФЗ)", re.I)
# «[2]», «[1, 3]», «[1–3]», а также «[1, ч. 6]» и «[2, ст. 115 ТК РФ]» — номера в начале скобок
_MARKERS = re.compile(r"\[(\d+(?:\s*[,;–-]\s*\d+)*)(?:\s*[,;]\s*[^\]\d][^\]]*)?\]")
_QUOTE = re.compile(r"«([^«»]{20,})»")
_WORDS = re.compile(r"[а-яёa-z0-9]+")


@dataclass
class Citation:
    text: str
    act: str
    key: str
    part: str | None = None
    point: str | None = None
    subpoint: str | None = None
    status: str = "ok"
    severity: str = OK
    message: str = ""
    sources: list[int] = field(default_factory=list)  # номера источников с этой статьёй
    changes_on: list[date] = field(default_factory=list)  # будущие изменения статьи


@dataclass
class Issue:
    severity: str
    code: str
    message: str


@dataclass
class CheckReport:
    verdict: str  # passed | warnings | failed
    citations: list[Citation]
    markers: list[int]
    issues: list[Issue]

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == ERROR]


def _act_alias(raw: str) -> str:
    for pattern, short in _ACT_ALIASES:
        if pattern.match(raw.strip()):
            return short
    law = _LAW_NUMBER.search(raw)
    return f"Федеральный закон № {law.group(1)}" if law else raw.strip()


def _part_number(raw: str | None) -> str | None:
    if raw is None or raw.isdigit():
        return raw
    for stem, num in _ORDINALS.items():
        if raw.lower().startswith(stem):
            return num
    return raw


def _expand_keys(raw: str) -> list[str]:
    keys = []
    for key in (k.strip() for k in re.split(r"\s*(?:,|\bи\b)\s*", raw)):
        # «ст. 80-81» — диапазон; номера с дефисом в кодексах всегда с точкой: 341.1-1
        span = re.fullmatch(r"(\d+)-(\d+)", key)
        if span and int(span.group(2)) > int(span.group(1)):
            keys += [span.group(1), span.group(2)]
        elif key:
            keys.append(key)
    return keys


def extract(text: str, default_act: str) -> list[Citation]:
    citations = []
    for m in _ARTICLE.finditer(text):
        units = _UNITS_BEFORE.search(text[max(0, m.start() - 60) : m.start()])
        sub, point, part = units.groups() if units else (None, None, None)
        act_match = _ACT_AFTER.match(text[m.end() : m.end() + 120])
        act = _act_alias(act_match.group(1)) if act_match else default_act
        end = m.end() + (act_match.end() if act_match else 0)
        start = m.start() - (len(units.group(0)) if units else 0)
        keys = _expand_keys(m.group(1))
        for i, key in enumerate(keys):
            # единицы перед перечнем относятся к первой статье: «ч. 1 ст. 80 и 81»
            citations.append(
                Citation(
                    text=text[start:end].strip(),
                    act=act,
                    key=key,
                    part=_part_number(part) if i == 0 else None,
                    point=point if i == 0 else None,
                    subpoint=sub if i == 0 else None,
                )
            )
    return citations


def markers(text: str) -> list[int]:
    found: list[int] = []
    for m in _MARKERS.finditer(text):
        for piece in re.split(r"\s*[,;]\s*", m.group(1)):
            bounds = re.split(r"\s*[–-]\s*", piece)
            if len(bounds) == 2 and all(b.isdigit() for b in bounds):
                lo, hi = int(bounds[0]), int(bounds[1])
                nums = range(lo, hi + 1) if 0 < hi - lo < 20 else [lo, hi]
            else:
                nums = [int(b) for b in bounds if b.isdigit()]
            found.extend(n for n in nums if n not in found)
    return found


def _normalize(text: str) -> str:
    return " ".join(_WORDS.findall(text.lower().replace("ё", "е")))


def _has_unit(fragments: list[dict], c: Citation) -> bool:
    wanted = []
    if c.part:
        wanted.append(["часть", c.part])
    if c.point:
        wanted.append(["пункт", c.point])
    if c.subpoint:
        wanted.append(["подпункт", c.subpoint])
    if not wanted:
        return True
    refs = [f["ref"] for f in fragments]
    if any(ref[: len(wanted)] == wanted for ref in refs):
        return True
    # «п. 6 ст. 81» без части — принятая сокращённая запись «п. 6 ч. 1 ст. 81»
    if not c.part and c.point:
        tail = wanted
        return any(ref[i : i + len(tail)] == tail for ref in refs for i in range(len(ref)))
    return False


async def check(session: AsyncSession, answer: str, sources: list[Hit], on: date) -> CheckReport:
    acts = {
        (a.short_title or short_title_for(a.title) or a.title): a
        for a in await session.scalars(select(Act).where(Act.is_tracked))
    }
    default_act = sources[0].act if sources else next(iter(acts), "ТК РФ")
    issues: list[Issue] = []

    used_markers = markers(answer)
    for n in used_markers:
        if not 1 <= n <= len(sources):
            issues.append(Issue(ERROR, "bad_marker", f"ссылка [{n}] на несуществующий источник"))

    by_article: dict[tuple[str, str], list[int]] = {}
    for n, hit in enumerate(sources, 1):
        by_article.setdefault((hit.act, hit.key), []).append(n)

    citations = extract(answer, default_act)
    redactions: dict[int, int | None] = {}
    for c in citations:
        act = acts.get(c.act)
        label = cite(c.key, _ref(c), c.act)
        if act is None:
            c.status, c.severity = "external", INFO
            c.message = f"{c.act} нет в корпусе — ссылку не проверить"
            continue
        if act.id not in redactions:
            red = await queries.redaction_on(session, act.id, on)
            redactions[act.id] = red.id if red else None
        version = None
        if redactions[act.id] is not None:
            version = await session.scalar(
                select(ArticleVersion)
                .join(RedactionArticle, RedactionArticle.version_id == ArticleVersion.id)
                .where(
                    RedactionArticle.redaction_id == redactions[act.id],
                    RedactionArticle.key == c.key,
                )
            )
        c.sources = by_article.get((c.act, c.key), [])
        if version is None:
            c.status, c.severity = "not_found", ERROR
            c.message = f"{label}: такой статьи нет в редакции на {on:%d.%m.%Y}"
        elif version.is_repealed:
            c.status, c.severity = "repealed", ERROR
            c.message = f"{label}: статья утратила силу"
        elif not _has_unit(version.fragments, c):
            c.status, c.severity = "unit_not_found", WARNING
            c.message = f"{label}: в статье нет такой части или пункта"
        elif not c.sources:
            c.status, c.severity = "not_in_sources", WARNING
            c.message = f"{label}: статья существует, но её текста не было среди источников"
        if version is not None and not version.is_repealed:
            c.changes_on = [
                p.valid_from
                for p in await queries.article_history(session, act.id, c.key)
                if p.valid_from > on
            ]
            if c.changes_on and c.severity == OK:
                c.severity = INFO
                c.message = f"{label}: изменится с {c.changes_on[0]:%d.%m.%Y}"

    for c in citations:
        if c.severity in (ERROR, WARNING):
            issues.append(Issue(c.severity, c.status, c.message))

    corpus_text = [_normalize(h.text) for h in sources]
    for m in _QUOTE.finditer(answer):
        for piece in re.split(r"\.\.\.|…", m.group(1)):
            words = _normalize(piece)
            if len(words.split()) >= 4 and not any(words in t for t in corpus_text):
                issues.append(
                    Issue(
                        WARNING,
                        "quote_not_found",
                        f"цитаты «{piece.strip()[:80]}» нет в источниках",
                    )
                )

    if sources and not citations and not used_markers:
        issues.append(Issue(WARNING, "no_citations", "в ответе нет ссылок на нормы"))

    verdict = (
        "failed" if any(i.severity == ERROR for i in issues) else "warnings" if issues else "passed"
    )
    return CheckReport(verdict=verdict, citations=citations, markers=used_markers, issues=issues)


def _ref(c: Citation) -> list[list[str]]:
    ref = []
    if c.part:
        ref.append(["часть", c.part])
    if c.point:
        ref.append(["пункт", c.point])
    if c.subpoint:
        ref.append(["подпункт", c.subpoint])
    return ref
