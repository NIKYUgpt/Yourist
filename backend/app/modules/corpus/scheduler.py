import asyncio
import logging

from sqlalchemy import select

from app.modules.corpus.config import CorpusSettings
from app.modules.corpus.models import Act
from app.modules.corpus.sync import CorpusSync

logger = logging.getLogger(__name__)


class SyncScheduler:
    """Периодическая синхронизация отслеживаемых актов и ручной запуск в фоне."""

    def __init__(self, sync: CorpusSync, settings: CorpusSettings):
        self.sync = sync
        self.settings = settings
        self._loop: asyncio.Task | None = None
        self._tasks: set[asyncio.Task] = set()

    def start(self) -> None:
        self._loop = asyncio.create_task(self._run_forever(), name="corpus-sync-loop")

    async def stop(self) -> None:
        tasks = [t for t in (self._loop, *self._tasks) if t is not None]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def trigger(self, act_id: int, reason: str = "manual") -> None:
        task = asyncio.create_task(self._safe_sync(act_id, reason), name=f"corpus-sync-{act_id}")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def wait_idle(self) -> None:
        """Дождаться фоновых запусков (используется в тестах и при остановке)."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    async def ensure_default_acts(self) -> None:
        async with self.sync.sessions() as session:
            known = set(await session.scalars(select(Act.external_hash)))
        for doc_hash in self.settings.default_acts:
            if doc_hash in known:
                continue
            try:
                act = await self.sync.track_act(doc_hash)
                logger.info("Акт добавлен в отслеживаемые: %s", act.title)
            except Exception:
                logger.exception("Не удалось добавить акт %s, попробуем в следующий раз", doc_hash)

    async def _safe_sync(self, act_id: int, reason: str) -> None:
        try:
            await self.sync.sync_act(act_id, trigger=reason)
        except Exception:
            logger.exception("Синхронизация акта %s завершилась ошибкой", act_id)

    async def _run_forever(self) -> None:
        reason = "startup"
        while True:
            try:
                await self.ensure_default_acts()
                runs = await self.sync.sync_all(trigger=reason)
                logger.info("Синхронизация корпуса (%s): %s", reason, [r.stats for r in runs])
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Плановая синхронизация корпуса завершилась ошибкой")
            reason = "schedule"
            await asyncio.sleep(self.settings.sync_interval_hours * 3600)
