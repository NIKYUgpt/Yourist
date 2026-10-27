"""Сервис корпуса законов: синхронизация с pravo.gov.ru, чтение редакций, поисковый индекс.

Отдельный процесс (свой контейнер): своя нагрузка — скачивание и разбор редакций
по расписанию — и свой жизненный цикл. Пользователей не хранит: права проверяет
по роли из access-токена основного API.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.core import health
from app.db.session import engine
from app.modules.ai.config import get_provider_settings
from app.modules.ai.registry import build_registry
from app.modules.corpus.client import PravoClient
from app.modules.corpus.config import get_corpus_settings
from app.modules.corpus.router import router as corpus_router
from app.modules.corpus.scheduler import SyncScheduler
from app.modules.corpus.sync import CorpusSync
from app.modules.search.indexer import Indexer, IndexWorker
from app.modules.search.router import index_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_corpus_settings()
    client = PravoClient(settings)
    registry = build_registry(get_provider_settings())
    worker = IndexWorker(Indexer(engine, registry), settings.index_interval_minutes * 60)
    scheduler = SyncScheduler(CorpusSync(engine, client), settings, after_sync=worker.trigger)
    app.state.corpus_client = client
    app.state.corpus_sync = scheduler.sync
    app.state.scheduler = scheduler
    app.state.ai_registry = registry
    app.state.index_worker = worker
    if settings.index_enabled:
        worker.start()
    if settings.sync_enabled:
        scheduler.start()
    yield
    await scheduler.stop()
    await worker.stop()
    await client.aclose()
    await registry.aclose()
    await engine.dispose()


def create_app() -> FastAPI:
    logging.basicConfig(level=logging.INFO)
    app = FastAPI(title="Yourist Corpus", version=__version__, lifespan=lifespan)
    app.include_router(health.router, prefix="/api/v1")
    app.include_router(corpus_router, prefix="/api/v1")
    app.include_router(index_router, prefix="/api/v1")
    return app


app = create_app()
