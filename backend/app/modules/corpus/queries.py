"""Чтение корпуса: редакция и статья на дату, история статьи, разница между датами."""

import difflib
from dataclasses import dataclass, field
from datetime import date

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.corpus.models import ArticleVersion, Redaction, RedactionArticle, RedactionStatus


async def redaction_on(session: AsyncSession, act_id: int, on: date) -> Redaction | None:
    """Редакция, действовавшая на дату: последняя разобранная с датой начала не позже on."""
    return await session.scalar(
        select(Redaction)
        .where(
            Redaction.act_id == act_id,
            Redaction.status == RedactionStatus.PARSED,
            Redaction.valid_from <= on,
        )
        .order_by(Redaction.valid_from.desc(), Redaction.id.desc())
        .limit(1)
    )


async def parsed_redactions(session: AsyncSession, act_id: int) -> list[Redaction]:
    """Разобранные редакции по возрастанию даты; при совпадении дат остаётся последняя."""
    rows = await session.scalars(
        select(Redaction)
        .where(Redaction.act_id == act_id, Redaction.status == RedactionStatus.PARSED)
        .order_by(Redaction.valid_from, Redaction.id)
    )
    by_date: dict[date, Redaction] = {}
    for red in rows:
        by_date[red.valid_from] = red
    return list(by_date.values())


async def article_on(
    session: AsyncSession, act_id: int, key: str, on: date
) -> tuple[ArticleVersion, Redaction] | None:
    red = await redaction_on(session, act_id, on)
    if red is None:
        return None
    version = await session.scalar(
        select(ArticleVersion)
        .join(RedactionArticle, RedactionArticle.version_id == ArticleVersion.id)
        .where(RedactionArticle.redaction_id == red.id, RedactionArticle.key == key)
    )
    return (version, red) if version is not None else None


async def contents_on(
    session: AsyncSession, act_id: int, on: date
) -> tuple[Redaction, list[ArticleVersion]] | None:
    red = await redaction_on(session, act_id, on)
    if red is None:
        return None
    versions = await session.scalars(
        select(ArticleVersion)
        .join(RedactionArticle, RedactionArticle.version_id == ArticleVersion.id)
        .where(RedactionArticle.redaction_id == red.id)
        .order_by(RedactionArticle.position)
    )
    return red, list(versions)


@dataclass
class Period:
    """Отрезок времени, когда у статьи был один и тот же текст. version=None — статьи не было."""

    version: ArticleVersion | None
    valid_from: date
    valid_to: date | None  # не включительно; None — действует до сих пор


async def article_history(session: AsyncSession, act_id: int, key: str) -> list[Period]:
    redactions = await parsed_redactions(session, act_id)
    if not redactions:
        return []
    mapping = dict(
        (
            await session.execute(
                select(RedactionArticle.redaction_id, RedactionArticle.version_id).where(
                    RedactionArticle.redaction_id.in_([r.id for r in redactions]),
                    RedactionArticle.key == key,
                )
            )
        ).all()
    )
    versions = {
        v.id: v
        for v in await session.scalars(
            select(ArticleVersion).where(ArticleVersion.id.in_(list(set(mapping.values()))))
        )
    }

    periods: list[Period] = []
    for red in redactions:
        version = versions.get(mapping.get(red.id))
        current = periods[-1] if periods else None
        if current is not None and (current.version and current.version.id) == (
            version and version.id
        ):
            continue
        if current is not None:
            current.valid_to = red.valid_from
        periods.append(Period(version=version, valid_from=red.valid_from, valid_to=None))
    # периоды до появления статьи не интересны
    while periods and periods[0].version is None:
        periods.pop(0)
    return periods


@dataclass
class ArticleChange:
    key: str
    kind: str  # added | removed | changed
    before: ArticleVersion | None
    after: ArticleVersion | None
    diff: list[str] = field(default_factory=list)
    notes_added: list[str] = field(default_factory=list)


@dataclass
class Changes:
    before: Redaction
    after: Redaction
    items: list[ArticleChange]


async def _mapping(session: AsyncSession, redaction_id: int) -> dict[str, tuple[int, int]]:
    rows = await session.execute(
        select(RedactionArticle.key, RedactionArticle.version_id, RedactionArticle.position).where(
            RedactionArticle.redaction_id == redaction_id
        )
    )
    return {key: (vid, pos) for key, vid, pos in rows}


def text_diff(before: str, after: str) -> list[str]:
    lines = difflib.unified_diff(before.splitlines(), after.splitlines(), lineterm="", n=0)
    return [line for line in lines if not line.startswith(("---", "+++"))]


async def changes_between(
    session: AsyncSession, act_id: int, date_from: date, date_to: date
) -> Changes | None:
    before = await redaction_on(session, act_id, date_from)
    after = await redaction_on(session, act_id, date_to)
    if before is None or after is None:
        return None

    old, new = await _mapping(session, before.id), await _mapping(session, after.id)
    changed_keys = [
        k for k in new.keys() | old.keys() if old.get(k, (None,))[0] != new.get(k, (None,))[0]
    ]
    needed = {old[k][0] for k in changed_keys if k in old} | {
        new[k][0] for k in changed_keys if k in new
    }
    versions = {
        v.id: v
        for v in await session.scalars(
            select(ArticleVersion).where(ArticleVersion.id.in_(list(needed)))
        )
    }

    items = []
    for key in changed_keys:
        a = versions.get(old[key][0]) if key in old else None
        b = versions.get(new[key][0]) if key in new else None
        kind = "added" if a is None else "removed" if b is None else "changed"
        item = ArticleChange(key=key, kind=kind, before=a, after=b)
        if a is not None and b is not None:
            item.diff = text_diff(a.text, b.text)
            item.notes_added = [n for n in b.notes if n not in a.notes]
        items.append(item)

    def position(item: ArticleChange) -> int:
        return new[item.key][1] if item.key in new else old[item.key][1]

    items.sort(key=position)
    return Changes(before=before, after=after, items=items)
