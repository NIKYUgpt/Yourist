import asyncio
import os
import re
from collections.abc import AsyncIterator

import asyncpg
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text, update
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

# Тесты работают с отдельной БД; переменные задаём до импорта приложения
TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+asyncpg://yourist:yourist@127.0.0.1:5432/yourist_test"
)
os.environ["DATABASE_URL"] = TEST_DATABASE_URL
os.environ["ENV"] = "test"
os.environ["MAIL_BACKEND"] = "console"
os.environ["SECRET_KEY"] = "test-secret-key-that-is-long-enough-for-hs256"

from app.core.mail import ConsoleMailer, get_mailer  # noqa: E402
from app.db.registry import Base  # noqa: E402
from app.db.session import get_session  # noqa: E402
from app.main import app  # noqa: E402
from app.modules.auth.models import Role, User  # noqa: E402

PASSWORD = "correct-horse-42"


async def _ensure_database(url: str) -> None:
    u = make_url(url)
    conn = await asyncpg.connect(
        user=u.username, password=u.password, host=u.host, port=u.port or 5432, database="postgres"
    )
    try:
        if not await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", u.database):
            await conn.execute(f'CREATE DATABASE "{u.database}"')
    finally:
        await conn.close()


async def _prepare() -> None:
    await _ensure_database(TEST_DATABASE_URL)
    engine = create_async_engine(TEST_DATABASE_URL, poolclass=NullPool)
    schemas = {"public"} | {t.schema for t in Base.metadata.tables.values() if t.schema}
    async with engine.begin() as conn:
        # тестовую БД каждый раз собираем с нуля, схемы модулей — тоже
        for schema in sorted(schemas):
            await conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
            await conn.execute(text(f'CREATE SCHEMA "{schema}"'))
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()


def pytest_sessionstart(session: pytest.Session) -> None:
    asyncio.run(_prepare())


@pytest.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    # NullPool: у каждого теста свой event loop, переиспользовать соединения нельзя
    eng = create_async_engine(TEST_DATABASE_URL, poolclass=NullPool)
    yield eng
    tables = ", ".join(t.fullname for t in Base.metadata.sorted_tables)
    async with eng.begin() as conn:
        await conn.execute(text(f"TRUNCATE {tables} CASCADE"))
    await eng.dispose()


@pytest.fixture
async def db(engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    async with async_sessionmaker(engine, expire_on_commit=False)() as session:
        yield session


@pytest.fixture
def mailer() -> ConsoleMailer:
    return ConsoleMailer()


@pytest.fixture
async def client(engine: AsyncEngine, mailer: ConsoleMailer) -> AsyncIterator[AsyncClient]:
    sessions = async_sessionmaker(engine, expire_on_commit=False)

    async def _get_session() -> AsyncIterator[AsyncSession]:
        async with sessions() as session:
            yield session

    app.dependency_overrides[get_session] = _get_session
    app.dependency_overrides[get_mailer] = lambda: mailer
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


def last_email_token(mailer: ConsoleMailer) -> str:
    body = mailer.outbox[-1].get_content()
    match = re.search(r"token=([\w\-.]+)", body)
    assert match, body
    return match.group(1)


async def register(client: AsyncClient, email: str, password: str = PASSWORD) -> dict:
    resp = await client.post(
        "/api/v1/auth/register", json={"email": email, "password": password, "pd_consent": True}
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


async def register_verified(
    client: AsyncClient, mailer: ConsoleMailer, email: str, password: str = PASSWORD
) -> dict:
    await register(client, email, password)
    resp = await client.post("/api/v1/auth/verify-email", json={"token": last_email_token(mailer)})
    assert resp.status_code == 200, resp.text
    return resp.json()


async def login(client: AsyncClient, email: str, password: str = PASSWORD) -> dict:
    resp = await client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert resp.status_code == 200, resp.text
    return resp.json()


def bearer(tokens: dict) -> dict[str, str]:
    return {"Authorization": f"Bearer {tokens['access_token']}"}


async def set_role(db: AsyncSession, email: str, role: Role) -> None:
    await db.execute(update(User).where(User.email == email).values(role=role))
    await db.commit()
