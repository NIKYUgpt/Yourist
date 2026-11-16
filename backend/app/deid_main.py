"""Сервис деперсонализации: отдельный процесс на границе персональных данных.

Не подключается к БД и не пишет тексты в журнал. Наружу не публикуется —
доступен только основному API во внутренней сети.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI

from app import __version__
from app.modules.ai.config import get_provider_settings
from app.modules.ai.registry import build_registry
from app.modules.deid.api import get_deid_settings, install_identity_handlers, session_store
from app.modules.deid.api import router as deid_router
from app.modules.deid.names import morph


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.ai_registry = build_registry(get_provider_settings())
    app.state.deid_sessions.start()
    morph()  # словари pymorphy3 грузим при старте, а не на первом запросе
    yield
    await app.state.deid_sessions.aclose()
    await app.state.ai_registry.aclose()


health = APIRouter(tags=["health"])


@health.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


def create_app() -> FastAPI:
    logging.basicConfig(level=logging.INFO)
    app = FastAPI(title="Yourist De-identification", version=__version__, lifespan=lifespan)
    install_identity_handlers(app)
    app.state.deid_sessions = session_store(get_deid_settings())
    app.include_router(health, prefix="/api/v1")
    app.include_router(deid_router, prefix="/api/v1")
    return app


app = create_app()
