"""Переносимый снимок норм. Хеш текста не означает юридическую проверку."""

import hashlib
from datetime import date

from sqlalchemy import select

from app.evaluation.answer_dataset import Case, Strict, fingerprint
from app.modules.corpus.models import Act
from app.modules.corpus.queries import contents_on


class Article(Strict):
    act: str
    act_title: str
    key: str
    heading: str
    path: list[str]
    text: str
    fragments: list[dict]
    redaction: str
    valid_from: date
    content_sha256: str
    parser_content_hash: str
    is_repealed: bool
    source_url: str


class Snapshot(Strict):
    legal_as_of: date
    articles: list[Article]
    source_status: str = "local_snapshot_not_independently_verified"

    def verify(self) -> None:
        keys = [(a.act, a.key) for a in self.articles]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate corpus articles")
        for article in self.articles:
            if article.valid_from > self.legal_as_of:
                raise ValueError("future redaction")
            if hashlib.sha256(article.text.encode()).hexdigest() != article.content_sha256:
                raise ValueError("article hash mismatch")

    @property
    def sha256(self) -> str:
        return fingerprint(self.model_dump(mode="json"))


async def freeze(session, on: date) -> Snapshot:
    articles = []
    for act in await session.scalars(select(Act).where(Act.is_tracked).order_by(Act.id)):
        found = await contents_on(session, act.id, on)
        if not found:
            continue
        red, versions = found
        for v in versions:
            articles.append(
                Article(
                    act=f"{act.source}:{act.external_id}",
                    act_title=act.short_title or act.title,
                    key=v.key,
                    heading=v.heading,
                    path=v.path,
                    text=v.text,
                    fragments=v.fragments,
                    redaction=red.external_id,
                    valid_from=red.valid_from,
                    content_sha256=hashlib.sha256(v.text.encode()).hexdigest(),
                    parser_content_hash=v.content_hash,
                    is_repealed=v.is_repealed,
                    source_url="http://actual.pravo.gov.ru/"
                    if act.source == "pravo_actual"
                    else "",
                )
            )
    snapshot = Snapshot(legal_as_of=on, articles=articles)
    snapshot.verify()
    return snapshot


def oracle_sources(case: Case, snapshot: Snapshot) -> list[dict]:
    """Только принятые нормы, все OR-варианты. Полные статьи — привилегированная диагностика."""
    snapshot.verify()
    if snapshot.legal_as_of != case.legal_as_of:
        raise ValueError("oracle snapshot date differs from the case")
    if case.gold is None:
        raise ValueError("oracle requires adjudicated gold")
    available = {(a.act, a.key): a for a in snapshot.articles if not a.is_repealed}
    selected = {}
    for issue in case.gold.issues:
        for group in issue.evidence_alternatives:
            for norm in group:
                article = available.get((norm.act, norm.article))
                if (
                    article is None
                    or article.redaction != norm.redaction
                    or article.content_sha256 != norm.content_sha256
                    or norm.quote not in article.text
                ):
                    raise ValueError("gold norm missing, stale or quote not in snapshot")
                # Точная структурная единица также должна существовать, если задана.
                units = [("часть", norm.part), ("пункт", norm.point), ("абзац", norm.paragraph)]
                required = [[kind, value] for kind, value in units if value is not None]
                if required and not any(
                    all(unit in frag.get("ref", []) for unit in required)
                    and norm.quote in frag["text"]
                    for frag in article.fragments
                ):
                    raise ValueError("gold quote does not match requested structural unit")
                selected[(article.act, article.key)] = article.model_dump(mode="json")
    return list(selected.values())
