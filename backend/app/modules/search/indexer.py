"""Построение поискового индекса: нарезка версий статей и векторы фрагментов.

Работает в процессе сервиса корпуса: после каждой синхронизации и периодически —
чтобы подхватить смену модели эмбеддингов в админке. Векторы считаются только
для активной модели; при переключении local/api недостающие досчитываются.
"""

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field

from sqlalchemy import delete, func, insert, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.modules.ai.providers.base import EmbedKind, ProviderError
from app.modules.ai.registry import Registry
from app.modules.ai.service import AIGateway, load_config
from app.modules.corpus.models import Act, ArticleVersion
from app.modules.corpus.titles import short_title_for
from app.modules.search.chunking import CHUNKER_VERSION, chunk_article
from app.modules.search.models import Chunk, Embedding, IndexedVersion

logger = logging.getLogger(__name__)

_LOCK_NAMESPACE = 7002
_VERSIONS_PER_COMMIT = 100


def chunk_context(act_title: str, path: list[str], heading: str) -> str:
    return " — ".join([act_title, *path[-1:], heading])


def embedding_input(chunk_context_: str, chunk_text: str) -> str:
    return f"{chunk_context_}\n{chunk_text}"


@dataclass
class IndexStats:
    versions: int = 0
    chunks: int = 0
    embedded: int = 0
    space: str | None = None
    errors: list[str] = field(default_factory=list)


class Indexer:
    def __init__(self, engine: AsyncEngine, registry: Registry, embed_batch: int = 64):
        self.engine = engine
        self.sessions = async_sessionmaker(engine, expire_on_commit=False)
        self.registry = registry
        self.embed_batch = embed_batch

    async def run(self) -> IndexStats | None:
        """Нарезать новые версии и досчитать векторы. None — индекс уже строится."""
        async with self.engine.connect() as lock_conn:
            locked = await lock_conn.scalar(
                text("SELECT pg_try_advisory_lock(:ns, 0)"), {"ns": _LOCK_NAMESPACE}
            )
            await lock_conn.commit()
            if not locked:
                return None
            try:
                stats = IndexStats()
                await self.build_chunks(stats)
                await self.embed_missing(stats)
                if stats.versions or stats.embedded:
                    logger.info("Поисковый индекс обновлён: %s", stats)
                return stats
            finally:
                await lock_conn.execute(
                    text("SELECT pg_advisory_unlock(:ns, 0)"), {"ns": _LOCK_NAMESPACE}
                )
                await lock_conn.commit()

    async def build_chunks(self, stats: IndexStats) -> None:
        async with self.sessions() as session:
            fresh = select(IndexedVersion.version_id).where(
                IndexedVersion.chunker_version == CHUNKER_VERSION
            )
            rows = await session.execute(
                select(ArticleVersion, Act.short_title, Act.title)
                .join(Act, Act.id == ArticleVersion.act_id)
                .where(ArticleVersion.id.not_in(fresh))
                .order_by(ArticleVersion.id)
            )
            pending = rows.all()
            for i, (version, short_title, title) in enumerate(pending, 1):
                act_title = short_title or short_title_for(title) or title
                await self._index_version(session, version, act_title, stats)
                if i % _VERSIONS_PER_COMMIT == 0:
                    await session.commit()
            await session.commit()

    async def _index_version(
        self, session: AsyncSession, version: ArticleVersion, act_title: str, stats: IndexStats
    ) -> None:
        drafts = chunk_article(version.text, version.fragments, version.is_repealed)
        # пересборка после смены алгоритма: старые фрагменты (и их векторы) удаляются
        await session.execute(delete(Chunk).where(Chunk.version_id == version.id))
        context = chunk_context(act_title, version.path, version.heading)
        if drafts:
            await session.execute(
                insert(Chunk),
                [
                    {
                        "version_id": version.id,
                        "ord": d.ord,
                        "context": context,
                        "text": d.text,
                        "refs": d.refs,
                    }
                    for d in drafts
                ],
            )
        values = {"chunker_version": CHUNKER_VERSION, "chunks_count": len(drafts)}
        await session.execute(
            pg_insert(IndexedVersion)
            .values(version_id=version.id, **values)
            .on_conflict_do_update(
                index_elements=[IndexedVersion.version_id],
                set_={**values, "indexed_at": func.now()},
            )
        )
        stats.versions += 1
        stats.chunks += len(drafts)

    async def embed_missing(self, stats: IndexStats) -> None:
        async with self.sessions() as session:
            gateway = AIGateway(self.registry, await load_config(session))
        space = stats.space = gateway.embedding_key
        while True:
            async with self.sessions() as session:
                done = select(Embedding.chunk_id).where(Embedding.space == space)
                batch = (
                    await session.execute(
                        select(Chunk.id, Chunk.context, Chunk.text)
                        .where(Chunk.id.not_in(done))
                        .order_by(Chunk.id)
                        .limit(self.embed_batch)
                    )
                ).all()
                if not batch:
                    return
                try:
                    vectors = await gateway.embed(
                        [embedding_input(c, t) for _, c, t in batch], EmbedKind.DOCUMENT
                    )
                except ProviderError as exc:
                    logger.warning("Векторы не посчитаны (%s): %s", space, exc)
                    stats.errors.append(str(exc))
                    return
                await session.execute(
                    insert(Embedding),
                    [
                        {"chunk_id": cid, "space": space, "vector": vec}
                        for (cid, _, _), vec in zip(batch, vectors, strict=True)
                    ],
                )
                await session.commit()
                stats.embedded += len(batch)


async def index_status(session: AsyncSession) -> dict:
    versions = await session.scalar(select(func.count()).select_from(ArticleVersion))
    indexed = await session.scalar(
        select(func.count())
        .select_from(IndexedVersion)
        .where(IndexedVersion.chunker_version == CHUNKER_VERSION)
    )
    chunks = await session.scalar(select(func.count()).select_from(Chunk))
    spaces = dict(
        (
            await session.execute(select(Embedding.space, func.count()).group_by(Embedding.space))
        ).all()
    )
    return {"versions": versions, "indexed_versions": indexed, "chunks": chunks, "spaces": spaces}


class IndexWorker:
    """Фоновое обновление индекса: по сигналу (после синхронизации) и по таймеру."""

    def __init__(self, indexer: Indexer, interval_sec: float):
        self.indexer = indexer
        self.interval_sec = interval_sec
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None
        self.last: IndexStats | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop(), name="search-index-loop")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    def trigger(self) -> None:
        self._wake.set()

    async def _loop(self) -> None:
        while True:
            try:
                stats = await self.indexer.run()
                if stats is not None:
                    self.last = stats
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Обновление поискового индекса завершилось ошибкой")
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._wake.wait(), timeout=self.interval_sec)
            self._wake.clear()
