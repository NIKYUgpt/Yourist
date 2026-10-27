"""Основной API: пользователи, вопросы к ассистенту, поиск норм, настройки нейросетей."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.core import health
from app.core.config import get_settings
from app.db.session import engine
from app.modules.ai.config import get_provider_settings
from app.modules.ai.registry import build_registry
from app.modules.ai.router import router as ai_router
from app.modules.assistant.router import router as assistant_router
from app.modules.auth.router import router as auth_router
from app.modules.deid.api import DeidClient, get_deid_settings
from app.modules.search.router import router as search_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.ai_registry = build_registry(get_provider_settings())
    app.state.deid_client = DeidClient(get_deid_settings(), app.state.ai_registry)
    yield
    await app.state.deid_client.aclose()
    await app.state.ai_registry.aclose()
    await engine.dispose()


def create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(level=logging.INFO)

    app = FastAPI(title="Yourist API", version=__version__, lifespan=lifespan)
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    app.include_router(health.router, prefix="/api/v1")
    app.include_router(auth_router, prefix="/api/v1")
    app.include_router(ai_router, prefix="/api/v1")
    app.include_router(search_router, prefix="/api/v1")
    app.include_router(assistant_router, prefix="/api/v1")
    return app


app = create_app()
