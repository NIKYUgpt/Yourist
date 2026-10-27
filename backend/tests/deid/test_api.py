from datetime import timedelta
from uuid import uuid4

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from app.core.security import ACCESS, SERVICE, create_token
from app.deid_main import app as deid_app
from app.modules.ai.profiles import ModelRef
from app.modules.deid.api import DeidClient, DeidSettings, DeidUnavailable
from tests.fakes import fake_registry

TEXT = "Меня, Петрова Сергея Ивановича, уволили. Тел. 89161234567"


@pytest.fixture
def registry():
    registry = fake_registry()
    registry.get("ollama").reply = lambda messages: '{"entities": []}'
    deid_app.state.ai_registry = registry
    return registry


def auth(purpose: str, **claims) -> dict[str, str]:
    token = create_token(uuid4(), purpose, timedelta(minutes=1), **claims)
    return {"Authorization": f"Bearer {token}"}


async def test_service_requires_service_token(registry):
    async with AsyncClient(transport=ASGITransport(app=deid_app), base_url="http://deid") as c:
        assert (await c.post("/api/v1/deid", json={"text": TEXT})).status_code == 401
        # токен пользователя (даже администратора) не подходит
        resp = await c.post("/api/v1/deid", json={"text": TEXT}, headers=auth(ACCESS, role="admin"))
        assert resp.status_code == 401
        resp = await c.post("/api/v1/deid", json={"text": TEXT}, headers=auth(SERVICE))
        assert resp.status_code == 200
        assert resp.json()["text"] == "Меня, [ФИО_1], уволили. Тел. [ТЕЛЕФОН_1]"
        assert (await c.get("/api/v1/health")).json()["status"] == "ok"


async def test_client_over_http(registry):
    settings = DeidSettings(deid_url="http://deid")
    client = DeidClient(settings, registry, transport=ASGITransport(app=deid_app))
    result = await client.deidentify(TEXT, ModelRef(provider="ollama", model="qwen3:8b"))
    assert result.mapping["[ФИО_1]"] == "Петрова Сергея Ивановича"
    assert result.llm["provider"] == "ollama"
    assert len(registry.get("ollama").chat_calls) == 1
    await client.aclose()


async def test_client_in_process_without_url(registry):
    client = DeidClient(DeidSettings(deid_url=""), registry)
    result = await client.deidentify(TEXT, None)
    assert result.text.endswith("[ТЕЛЕФОН_1]") and result.llm is None


async def test_client_fails_closed(registry):
    def down(request):
        raise httpx.ConnectError("refused")

    client = DeidClient(
        DeidSettings(deid_url="http://deid"), registry, transport=httpx.MockTransport(down)
    )
    with pytest.raises(DeidUnavailable):
        await client.deidentify(TEXT, None)
