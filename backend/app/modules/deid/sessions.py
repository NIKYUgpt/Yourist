"""Ограниченная по времени память меток. Нет БД, файлов и восстановления после рестарта."""

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from uuid import UUID, uuid4

from app.core.time import utcnow


class SessionExpired(Exception):
    pass


class SessionBusy(Exception):
    pass


class SessionLimit(Exception):
    pass


@dataclass
class SessionInfo:
    id: UUID
    user_id: UUID
    expires_at: datetime


@dataclass
class _Entry:
    info: SessionInfo
    mapping: dict[str, str] = field(default_factory=dict, repr=False)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    closed: bool = False


class SessionStore:
    def __init__(
        self,
        ttl_seconds: int = 86400,
        max_sessions: int = 256,
        max_entities: int = 256,
        max_chars: int = 32768,
        clock: Callable[[], datetime] = utcnow,
    ):
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self.max_entities = max_entities
        self.max_chars = max_chars
        self.clock = clock
        self.entries: dict[UUID, _Entry] = {}
        self._cleaner: asyncio.Task | None = None

    def prune(self) -> None:
        now = self.clock()
        for key, entry in list(self.entries.items()):
            if entry.info.expires_at <= now:
                self.remove(key, entry.info.user_id)

    def create(self, user_id: UUID) -> SessionInfo:
        self.prune()
        if len(self.entries) >= self.max_sessions:
            raise SessionLimit("Достигнут предел активных сеансов")
        info = SessionInfo(uuid4(), user_id, self.clock() + timedelta(seconds=self.ttl_seconds))
        self.entries[info.id] = _Entry(info)
        return info

    def remove(self, key: UUID, user_id: UUID) -> None:
        entry = self.entries.get(key)
        if entry is not None and entry.info.user_id == user_id:
            entry.closed = True
            entry.mapping.clear()
            del self.entries[key]

    @asynccontextmanager
    async def use(self, key: UUID, user_id: UUID) -> AsyncIterator[dict[str, str]]:
        self.prune()
        entry = self.entries.get(key)
        if entry is None or entry.info.user_id != user_id:
            raise SessionExpired("Сеанс обезличивания завершён; начните новый диалог")
        if entry.lock.locked():
            raise SessionBusy("Предыдущее сообщение ещё обрабатывается")
        async with entry.lock:
            working = entry.mapping.copy()
            try:
                yield working
                self.prune()
                if entry.closed:
                    raise SessionExpired("Сеанс обезличивания завершён; начните новый диалог")
                if (
                    len(working) > self.max_entities
                    or sum(map(len, working.values())) > self.max_chars
                ):
                    raise SessionLimit("Достигнут предел данных сеанса; начните новый диалог")
                entry.mapping = working.copy()
            finally:
                working.clear()

    def start(self) -> None:
        if self._cleaner is None:
            self._cleaner = asyncio.create_task(self._cleanup())

    async def _cleanup(self) -> None:
        while True:
            await asyncio.sleep(60)
            self.prune()

    async def aclose(self) -> None:
        if self._cleaner is not None:
            self._cleaner.cancel()
            with suppress(asyncio.CancelledError):
                await self._cleaner
            self._cleaner = None
        for entry in list(self.entries.values()):
            self.remove(entry.info.id, entry.info.user_id)
