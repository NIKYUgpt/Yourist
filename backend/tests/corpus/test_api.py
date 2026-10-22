from datetime import timedelta
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.core.security import ACCESS, create_token
from app.corpus_main import app as corpus_app
from app.db.session import get_session
from app.modules.corpus.config import TK_RF_HASH, CorpusSettings
from app.modules.corpus.scheduler import SyncScheduler

API = "/api/v1/corpus"


def auth(role: str) -> dict[str, str]:
    token = create_token(uuid4(), ACCESS, timedelta(minutes=5), role=role)
    return {"Authorization": f"Bearer {token}"}


ADMIN, USER = auth("admin"), auth("user")


@pytest.fixture
async def api(engine, corpus, source):
    sessions = async_sessionmaker(engine, expire_on_commit=False)

    async def _get_session():
        async with sessions() as session:
            yield session

    scheduler = SyncScheduler(corpus, CorpusSettings(sync_enabled=False, default_acts=[]))
    corpus_app.state.corpus_sync = corpus
    corpus_app.state.corpus_client = source
    corpus_app.state.scheduler = scheduler
    corpus_app.dependency_overrides[get_session] = _get_session
    async with AsyncClient(transport=ASGITransport(app=corpus_app), base_url="http://corpus") as c:
        c.scheduler = scheduler
        yield c
    await scheduler.wait_idle()
    corpus_app.dependency_overrides.clear()


@pytest.fixture
async def act_id(api) -> int:
    resp = await api.post(f"{API}/acts", json={"external_hash": TK_RF_HASH}, headers=ADMIN)
    assert resp.status_code == 201, resp.text
    await api.scheduler.wait_idle()  # добавление акта запускает синхронизацию в фоне
    return resp.json()["id"]


async def test_requires_token(api):
    assert (await api.get(f"{API}/acts")).status_code == 401


async def test_only_admin_manages_acts(api):
    payload = {"external_hash": TK_RF_HASH}
    assert (await api.post(f"{API}/acts", json=payload, headers=USER)).status_code == 403
    assert (
        await api.get(f"{API}/search", params={"q": "трудовой"}, headers=USER)
    ).status_code == 403
    assert (await api.get(f"{API}/sync-runs", headers=USER)).status_code == 403


async def test_add_act_validates_hash(api):
    resp = await api.post(f"{API}/acts", json={"external_hash": "nope"}, headers=ADMIN)
    assert resp.status_code == 422


async def test_added_act_is_synced_in_background(api, act_id):
    acts = (await api.get(f"{API}/acts", headers=USER)).json()
    assert [(a["id"], a["number"]) for a in acts] == [(act_id, "197-ФЗ")]

    reds = (await api.get(f"{API}/acts/{act_id}/redactions", headers=USER)).json()
    assert [(r["valid_from"], r["status"]) for r in reds] == [
        ("2026-09-01", "parsed"),
        ("2022-07-14", "parsed"),
    ]
    [run] = (await api.get(f"{API}/sync-runs", headers=ADMIN)).json()
    assert (run["trigger"], run["status"], run["stats"]["parsed"]) == ("added", "success", 2)


async def test_article_on_date_with_text_period(api, act_id):
    url = f"{API}/acts/{act_id}/articles/81"
    old = (await api.get(url, params={"date": "2023-01-01"}, headers=USER)).json()
    new = (await api.get(url, headers=USER)).json()  # сегодня = 22.10.2026

    assert (old["text_valid_from"], old["text_valid_to"]) == ("2022-07-14", "2026-09-01")
    assert (new["text_valid_from"], new["text_valid_to"]) == ("2026-09-01", None)
    assert new["redaction"]["external_id"] == "492286"
    assert new["heading"] == "Статья 81. Расторжение трудового договора по инициативе работодателя"
    subpoint = next(
        f
        for f in new["fragments"]
        if f["ref"] == [["часть", "1"], ["пункт", "6"], ["подпункт", "а"]]
    )
    assert subpoint["text"].startswith("а) прогула")

    # неизменившаяся статья: один текст на всём периоде
    art1 = (await api.get(f"{API}/acts/{act_id}/articles/1", headers=USER)).json()
    assert (art1["text_valid_from"], art1["text_valid_to"]) == ("2022-07-14", None)


async def test_article_not_found_cases(api, act_id):
    missing = await api.get(
        f"{API}/acts/{act_id}/articles/341.1-1", params={"date": "2023-01-01"}, headers=USER
    )
    assert missing.status_code == 404
    assert "341.1-1" in missing.json()["detail"]

    too_early = await api.get(
        f"{API}/acts/{act_id}/articles/1", params={"date": "2020-01-01"}, headers=USER
    )
    assert too_early.status_code == 404
    assert "Нет разобранной редакции" in too_early.json()["detail"]

    assert (await api.get(f"{API}/acts/999/articles/1", headers=USER)).status_code == 404


async def test_contents_and_history(api, act_id):
    contents = (await api.get(f"{API}/acts/{act_id}/articles", headers=USER)).json()
    assert [a["key"] for a in contents["articles"]] == ["1", "7", "56.1", "58", "81", "341.1-1"]
    assert next(a for a in contents["articles"] if a["key"] == "7")["is_repealed"] is True

    history = (await api.get(f"{API}/acts/{act_id}/articles/81/history", headers=USER)).json()
    assert [(p["valid_from"], p["valid_to"]) for p in history] == [
        ("2022-07-14", "2026-09-01"),
        ("2026-09-01", None),
    ]
    unknown = await api.get(f"{API}/acts/{act_id}/articles/9999/history", headers=USER)
    assert unknown.status_code == 404


async def test_changes_between_dates(api, act_id):
    resp = await api.get(
        f"{API}/acts/{act_id}/changes", params={"from": "2023-01-01"}, headers=USER
    )
    body = resp.json()
    assert (body["added"], body["removed"], body["changed"]) == (1, 0, 1)
    assert [(i["key"], i["kind"]) for i in body["items"]] == [
        ("81", "changed"),
        ("341.1-1", "added"),
    ]

    bad = await api.get(
        f"{API}/acts/{act_id}/changes",
        params={"from": "2026-10-01", "to": "2023-01-01"},
        headers=USER,
    )
    assert bad.status_code == 422


async def test_manual_sync_and_search(api, act_id, source):
    resp = await api.post(f"{API}/acts/{act_id}/sync", headers=ADMIN)
    assert resp.status_code == 202
    await api.scheduler.wait_idle()
    runs = (await api.get(f"{API}/sync-runs", params={"act_id": act_id}, headers=ADMIN)).json()
    assert [r["trigger"] for r in runs] == ["manual", "added"]

    assert (await api.post(f"{API}/acts/999/sync", headers=ADMIN)).status_code == 404

    hits = (await api.get(f"{API}/search", params={"q": "трудовой кодекс"}, headers=ADMIN)).json()
    assert hits[0]["external_hash"] == TK_RF_HASH
