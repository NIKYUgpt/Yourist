import pytest

from app.core.roles import Role
from app.modules.ai.profiles import DEFAULT_CONFIG
from tests.conftest import bearer, login, register_verified, set_role
from tests.fakes import FakeProvider

API = "/api/v1/admin/ai"


@pytest.fixture
async def admin(client, mailer, db):
    await register_verified(client, mailer, "admin@example.com")
    await set_role(db, "admin@example.com", Role.ADMIN)
    return bearer(await login(client, "admin@example.com"))


@pytest.fixture
async def user(client, mailer):
    await register_verified(client, mailer, "user@example.com")
    return bearer(await login(client, "user@example.com"))


def config_payload(**changes) -> dict:
    data = DEFAULT_CONFIG.model_dump(mode="json")
    for path, value in changes.items():
        task, slot = path.split("__")
        data["tasks"][task][slot] = value
    return data


async def test_only_admin(client, user):
    assert (await client.get(f"{API}/config")).status_code == 401
    assert (await client.get(f"{API}/config", headers=user)).status_code == 403
    resp = await client.put(f"{API}/mode", json={"mode": "api"}, headers=user)
    assert resp.status_code == 403


async def test_defaults_are_local(client, admin):
    resp = await client.get(f"{API}/config", headers=admin)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["config"]["mode"] == "local"
    assert body["effective"]["answer"] == {
        "title": "Ответ на вопрос",
        "mode": "local",
        "provider": "ollama",
        "model": "qwen3:8b",
        "configured": True,
    }
    assert body["effective"]["embed"]["model"] == "bge-m3"


async def test_one_switch_moves_all_tasks_to_api(client, admin):
    resp = await client.put(f"{API}/mode", json={"mode": "api"}, headers=admin)
    assert resp.status_code == 200, resp.text
    effective = resp.json()["effective"]
    assert {t: (e["mode"], e["provider"]) for t, e in effective.items()} == {
        "answer": ("api", "gigachat"),
        "verify": ("api", "yandex"),
        "deid": ("api", "yandex"),
        "embed": ("api", "yandex"),
        "stt": ("api", "yandex"),
        "ocr": ("api", "yandex"),
    }
    # настройка сохранилась в БД
    assert (await client.get(f"{API}/config", headers=admin)).json()["config"]["mode"] == "api"


async def test_task_can_override_mode(client, admin):
    payload = config_payload()
    payload["mode"] = "api"
    payload["tasks"]["embed"]["mode"] = "local"  # векторы остаются локальными
    resp = await client.put(f"{API}/config", json=payload, headers=admin)
    assert resp.status_code == 200, resp.text
    effective = resp.json()["effective"]
    assert effective["answer"]["mode"] == "api"
    assert (effective["embed"]["mode"], effective["embed"]["provider"]) == ("local", "ollama")


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"answer__local": {"provider": "yandex", "model": "x"}}, "answer.local"),
        ({"answer__api": {"provider": "ollama", "model": "x"}}, "answer.api"),
        ({"embed__api": {"provider": "deepseek", "model": "x"}}, "не умеет embed"),
        ({"deid__api": {"provider": "deepseek", "model": "x"}}, "deid.api: для исходных данных"),
        ({"stt__api": {"provider": "deepseek", "model": "x"}}, "не умеет stt"),
        ({"ocr__local": {"provider": "whisper", "model": "x"}}, "не умеет ocr"),
        ({"answer__api": {"provider": "nope", "model": "x"}}, "неизвестный провайдер"),
    ],
)
async def test_invalid_config_rejected(client, admin, changes, error):
    resp = await client.put(f"{API}/config", json=config_payload(**changes), headers=admin)
    assert resp.status_code == 422
    assert any(error in e for e in resp.json()["detail"]), resp.json()


async def test_providers_show_configuration_without_secrets(client, admin):
    resp = await client.get(f"{API}/providers", headers=admin)
    assert resp.status_code == 200
    by_name = {p["name"]: p for p in resp.json()}
    assert by_name["openai"]["configured"] is False
    assert by_name["yandex"] == {
        "name": "yandex",
        "title": "yandex",
        "kind": "api",
        "capabilities": ["chat", "embed", "ocr", "stt"],
        "location": "ru",
        "configured": True,
    }


async def test_check_and_models(client, admin, ai_registry):
    ollama: FakeProvider = ai_registry.get("ollama")
    ollama.reply = lambda messages: "Москва"
    resp = await client.post(
        f"{API}/providers/ollama/check", json={"model": "qwen3:8b"}, headers=admin
    )
    assert resp.json()["ok"] is True and resp.json()["detail"] == "Москва"

    resp = await client.post(
        f"{API}/providers/ollama/check",
        json={"model": "bge-m3", "capability": "embed"},
        headers=admin,
    )
    assert resp.json()["detail"] == "вектор размерности 256"

    ollama.fail = "connection refused"
    resp = await client.post(f"{API}/providers/ollama/check", json={"model": "x"}, headers=admin)
    assert resp.status_code == 200 and resp.json()["ok"] is False
    assert "connection refused" in resp.json()["detail"]

    resp = await client.get(f"{API}/providers/yandex/models", headers=admin)
    assert resp.json() == ["fake-small", "fake-large"]
    resp = await client.get(f"{API}/providers/openai/models", headers=admin)
    assert resp.status_code == 502 and "ключи" in resp.json()["detail"]
    resp = await client.post(f"{API}/providers/nope/check", json={"model": "x"}, headers=admin)
    assert resp.status_code == 404
