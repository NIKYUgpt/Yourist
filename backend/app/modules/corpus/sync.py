"""Синхронизация корпуса с источником."""

import hashlib
import json
import logging
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from typing import Any, Protocol

from sqlalchemy import delete, insert, select, text, update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.core.time import utcnow
from app.modules.corpus.client import ActCard, SourceError, SourceRedaction
from app.modules.corpus.models import (
    SOURCE_PRAVO_ACTUAL,
    Act,
    ArticleVersion,
    Redaction,
    RedactionArticle,
    RedactionSource,
    RedactionStatus,
    SyncRun,
    SyncStatus,
)
from app.modules.corpus.parser import ParsedDocument, ParseError, parse_redaction
from app.modules.corpus.titles import short_title_for

logger = logging.getLogger(__name__)

# даты вступления в силу — по московскому времени (UTC+3 без перехода на летнее)
MSK = timezone(timedelta(hours=3))
# пространство ключей pg_advisory_lock для синхронизации корпуса
_LOCK_NAMESPACE = 7001


def today_msk() -> date:
    return datetime.now(MSK).date()


class Source(Protocol):
    async def card(self, doc_hash: str) -> ActCard: ...
    async def redactions(self, doc_hash: str) -> list[SourceRedaction]: ...
    async def redaction_html(self, redaction_id: str) -> str: ...
    async def redaction_toc(self, redaction_id: str) -> list[dict[str, Any]]: ...


def _content_hash(html: str, toc: list[dict[str, Any]]) -> str:
    raw = html + json.dumps(toc, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


class CorpusSync:
    def __init__(self, engine: AsyncEngine, source: Source):
        self.engine = engine
        self.sessions = async_sessionmaker(engine, expire_on_commit=False)
        self.source = source

    async def track_act(self, doc_hash: str) -> Act:
        """Добавляет акт в отслеживаемые (или обновляет карточку существующего)."""
        card = await self.source.card(doc_hash)
        async with self.sessions() as session:
            act = await session.scalar(
                select(Act).where(
                    Act.source == SOURCE_PRAVO_ACTUAL, Act.external_id == card.external_id
                )
            )
            if act is None:
                act = Act(source=SOURCE_PRAVO_ACTUAL, external_id=card.external_id)
                session.add(act)
            act.external_hash = card.external_hash
            act.title = card.title
            act.short_title = short_title_for(card.title)
            act.kind = card.kind
            act.number = card.number
            act.adopted_on = card.adopted_on
            act.source_state = card.state
            act.is_tracked = True
            await session.commit()
            await session.refresh(act)
            return act

    async def sync_all(self, trigger: str) -> list[SyncRun]:
        async with self.sessions() as session:
            act_ids = list(
                await session.scalars(select(Act.id).where(Act.is_tracked).order_by(Act.id))
            )
        runs = []
        for act_id in act_ids:
            run = await self.sync_act(act_id, trigger)
            if run is not None:
                runs.append(run)
        return runs

    async def sync_act(self, act_id: int, trigger: str = "manual") -> SyncRun | None:
        """Синхронизирует один акт. None — если по нему уже идёт синхронизация."""
        async with self.engine.connect() as lock_conn:
            locked = await lock_conn.scalar(
                text("SELECT pg_try_advisory_lock(:ns, :id)"), {"ns": _LOCK_NAMESPACE, "id": act_id}
            )
            # блокировка сессионная: переживает commit, держится до unlock
            await lock_conn.commit()
            if not locked:
                logger.info("Синхронизация акта %s уже выполняется — пропускаем", act_id)
                return None
            try:
                return await self._sync(act_id, trigger)
            finally:
                await lock_conn.execute(
                    text("SELECT pg_advisory_unlock(:ns, :id)"),
                    {"ns": _LOCK_NAMESPACE, "id": act_id},
                )
                await lock_conn.commit()

    async def _sync(self, act_id: int, trigger: str) -> SyncRun:
        stats = dict.fromkeys(("new", "parsed", "updated", "unchanged", "withdrawn", "failed"), 0)
        async with self.sessions() as session:
            act = await session.get(Act, act_id)
            if act is None:
                raise ValueError(f"акт {act_id} не найден")
            doc_hash = act.external_hash
            if act.short_title is None:
                act.short_title = short_title_for(act.title)
            # блокировка у нас — значит, «running» в журнале остался от прерванного запуска
            await session.execute(
                update(SyncRun)
                .where(SyncRun.act_id == act_id, SyncRun.status == SyncStatus.RUNNING)
                .values(
                    status=SyncStatus.FAILED,
                    error="прервана: сервис остановили во время синхронизации",
                    finished_at=utcnow(),
                )
            )
            run = SyncRun(act_id=act_id, trigger=trigger, status=SyncStatus.RUNNING)
            session.add(run)
            await session.commit()
            run_id = run.id

        status, error = SyncStatus.SUCCESS, None
        try:
            source_redactions = await self.source.redactions(doc_hash)
            if not source_redactions:
                raise SourceError("источник вернул пустой список редакций")
            for redaction_id in await self._reconcile(act_id, source_redactions, stats):
                await self._fetch_and_parse(redaction_id, stats)
        except Exception as exc:
            logger.exception("Синхронизация акта %s завершилась ошибкой", act_id)
            status, error = SyncStatus.FAILED, f"{type(exc).__name__}: {exc}"

        async with self.sessions() as session:
            run = await session.get(SyncRun, run_id)
            act = await session.get(Act, act_id)
            now = utcnow()
            run.status, run.error, run.stats, run.finished_at = status, error, stats, now
            act.last_checked_at = now
            if stats["new"] or stats["updated"] or stats["withdrawn"]:
                act.last_changed_at = now
            await session.commit()
            return run

    async def _reconcile(
        self, act_id: int, source_redactions: list[SourceRedaction], stats: dict[str, int]
    ) -> list[int]:
        """Сверяет список редакций с источником. Возвращает id редакций, которые надо скачать."""
        today = today_msk()
        # актуальная и будущие редакции источник может пересобрать — их проверяем каждый раз
        actual_from = max(
            (r.valid_from for r in source_redactions if r.valid_from <= today), default=None
        )

        async with self.sessions() as session:
            existing = {
                r.external_id: r
                for r in await session.scalars(select(Redaction).where(Redaction.act_id == act_id))
            }
            seen: set[str] = set()
            to_fetch: list[Redaction] = []
            for src in source_redactions:
                if not src.has_content:
                    continue
                seen.add(src.external_id)
                red = existing.get(src.external_id)
                if red is None:
                    red = Redaction(
                        act_id=act_id, external_id=src.external_id, status=RedactionStatus.PENDING
                    )
                    session.add(red)
                    stats["new"] += 1
                elif red.status in (RedactionStatus.FAILED, RedactionStatus.WITHDRAWN) or (
                    red.source_timestamp != src.source_timestamp
                ):
                    red.status = RedactionStatus.PENDING
                red.valid_from = src.valid_from
                red.source_timestamp = src.source_timestamp
                red.caption = src.caption
                red.is_official = src.is_official

                recheck = actual_from is not None and src.valid_from >= actual_from
                if red.status == RedactionStatus.PENDING or recheck:
                    to_fetch.append(red)

            for external_id, red in existing.items():
                if external_id not in seen and red.status != RedactionStatus.WITHDRAWN:
                    red.status = RedactionStatus.WITHDRAWN
                    stats["withdrawn"] += 1
            await session.commit()
            return [r.id for r in sorted(to_fetch, key=lambda r: (r.valid_from, r.external_id))]

    async def _fetch_and_parse(self, redaction_id: int, stats: dict[str, int]) -> None:
        async with self.sessions() as session:
            red = await session.get(Redaction, redaction_id)
            # хеш сохраняется только после успешного разбора — значит, редакция уже есть в корпусе
            was_parsed = red.content_hash is not None
            try:
                html = await self.source.redaction_html(red.external_id)
                toc = await self.source.redaction_toc(red.external_id)
                content_hash = _content_hash(html, toc)
                red.fetched_at = utcnow()
                if red.content_hash == content_hash:
                    # содержимое не изменилось (даже если сменилась метка времени
                    # или прошлая попытка упала) — сопоставление в БД актуально
                    red.status, red.error = RedactionStatus.PARSED, None
                    stats["unchanged"] += 1
                    await session.commit()
                    return
                doc = parse_redaction(html, toc)
            except (SourceError, ParseError) as exc:
                logger.warning("Редакция %s: %s", red.external_id, exc)
                red.status, red.error = RedactionStatus.FAILED, str(exc)
                stats["failed"] += 1
                await session.commit()
                return

            await self._store_articles(session, red, doc)
            await session.merge(
                RedactionSource(redaction_id=red.id, html=html, toc=toc, fetched_at=red.fetched_at)
            )
            red.status = RedactionStatus.PARSED
            red.error = None
            red.content_hash = content_hash
            red.articles_count = len(doc.articles)
            red.warnings = doc.warnings[:200]
            red.parsed_at = utcnow()
            stats["updated" if was_parsed else "parsed"] += 1
            await session.commit()
            if doc.warnings:
                logger.info(
                    "Редакция %s: %d предупреждений разбора", red.external_id, len(doc.warnings)
                )

    async def _store_articles(
        self, session: AsyncSession, red: Redaction, doc: ParsedDocument
    ) -> None:
        """Сохраняет новые версии статей и заново собирает сопоставление редакции."""
        hashes = {a.key: a.content_hash() for a in doc.articles}
        rows = await session.execute(
            select(ArticleVersion.key, ArticleVersion.content_hash, ArticleVersion.id).where(
                ArticleVersion.act_id == red.act_id, ArticleVersion.key.in_(list(hashes))
            )
        )
        known = {(key, h): vid for key, h, vid in rows}

        new_versions = {}
        for article in doc.articles:
            if (article.key, hashes[article.key]) not in known:
                new_versions[article.key] = ArticleVersion(
                    act_id=red.act_id,
                    key=article.key,
                    heading=article.heading,
                    title=article.title,
                    text=article.text,
                    path=article.path,
                    fragments=[asdict(f) for f in article.fragments],
                    notes=article.notes,
                    is_repealed=article.repealed,
                    content_hash=hashes[article.key],
                )
        session.add_all(new_versions.values())
        await session.flush()
        for key, version in new_versions.items():
            known[(key, hashes[key])] = version.id

        await session.execute(
            delete(RedactionArticle).where(RedactionArticle.redaction_id == red.id)
        )
        await session.execute(
            insert(RedactionArticle),
            [
                {
                    "redaction_id": red.id,
                    "key": a.key,
                    "version_id": known[(a.key, hashes[a.key])],
                    "position": a.order,
                }
                for a in doc.articles
            ],
        )
