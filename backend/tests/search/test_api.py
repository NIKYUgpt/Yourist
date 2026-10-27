from datetime import timedelta
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.core.security import ACCESS, create_token
from app.corpus_main import app as corpus_app
from app.db.session import get_session
from app.modules.search.indexer import IndexWorker
from tests.conftest import bearer, login, register_verified


async def test_search_requires_login(client):
    assert (await client.get("/api/v1/search", params={"q": "прогул"})).status_code == 401


async def test_search_api(client, mailer, indexed):
    await register_verified(client, mailer, "user@example.com")
    headers = bearer(await login(client, "user@example.com"))
    resp = await client.get(
        "/api/v1/search",
        params={"q": "увольнение за прогул", "date": "2026-10-01", "k": 3},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["space"] == "ollama/bge-m3" and body["notes"] == []
    assert [(r["act"], r["valid_from"]) for r in body["redactions"]] == [("ТК РФ", "2026-09-01")]
    assert len(body["hits"]) == 3
    assert any(h["key"] == "81" and "прогула" in h["text"] for h in body["hits"])
    assert set(body["timings_ms"]) >= {"lexical", "embed", "vector"}


def token(role: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {create_token(uuid4(), ACCESS, timedelta(minutes=5), role=role)}"
    }


@pytest.fixture
async def corpus_api(engine, indexer):
    sessions = async_sessionmaker(engine, expire_on_commit=False)

    async def _get_session():
        async with sessions() as session:
            yield session

    corpus_app.state.ai_registry = indexer.registry
    corpus_app.state.index_worker = IndexWorker(indexer, interval_sec=3600)
    corpus_app.dependency_overrides[get_session] = _get_session
    async with AsyncClient(transport=ASGITransport(app=corpus_app), base_url="http://corpus") as c:
        yield c
    corpus_app.dependency_overrides.clear()


async def test_index_endpoints_admin_only(corpus_api, indexed):
    assert (await corpus_api.get("/api/v1/search/index", headers=token("user"))).status_code == 403
    resp = await corpus_api.get("/api/v1/search/index", headers=token("admin"))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["active_space"] == "ollama/bge-m3"
    assert body["spaces"] == {"ollama/bge-m3": body["chunks"]}
    assert body["indexed_versions"] == body["versions"]

    resp = await corpus_api.post("/api/v1/search/index", headers=token("admin"))
    assert resp.status_code == 202
    resp = await corpus_api.post(
        "/api/v1/search/index", params=[("mode", "local"), ("mode", "api")], headers=token("admin")
    )
    assert resp.status_code == 202
    assert corpus_app.state.index_worker._modes == {"local", "api"}
