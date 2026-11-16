"""Clarification stops before legal search/persistence, and retries keep only masked history."""

import json

import pytest
from sqlalchemy import func, select

from app.modules.ai.service import load_config, save_config
from app.modules.assistant.models import Answer, Dialog, Question
from app.modules.search.retrieval import SearchResult
from tests.assistant.test_identity_transport import PERSON, TEXT, resolution
from tests.conftest import bearer, login, register_verified

API = "/api/v1/assistant"


@pytest.fixture
async def identity_user(client, mailer, db):
    await register_verified(client, mailer, "identity@example.com")
    config = await load_config(db)
    config.deid_llm = False
    config.analyze = False
    config.verify = False
    await save_config(db, config, None)
    return bearer(await login(client, "identity@example.com"))


@pytest.mark.parametrize("same", [False, True])
async def test_identity_required_stops_then_retry_preserves_safe_history(
    client,
    db,
    identity_user,
    ai_registry,
    monkeypatch,
    same,
):
    searches = []

    async def no_sources(session, ai, query, on, **kwargs):
        searches.append(query)
        return SearchResult(query=query, on=on, hits=[], redactions=[])

    monkeypatch.setattr("app.modules.assistant.pipeline.search", no_sources)
    created = await client.post(API + "/dialogs", json={}, headers=identity_user)
    assert created.status_code == 201, created.text
    dialog_id = created.json()["id"]
    payload = {"text": TEXT, "dialog_id": dialog_id}
    response = await client.post(API + "/questions", json=payload, headers=identity_user)
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "person_identity_required"
    assert PERSON not in response.text
    assert not searches and not ai_registry.get("ollama").chat_calls
    assert not ai_registry.get("ollama").embed_calls
    assert await db.scalar(select(func.count()).select_from(Question)) == 0
    dialog = await db.scalar(select(Dialog))
    assert dialog.next_turn == 1

    payload["person_resolution"] = resolution(TEXT, same=same).model_dump(mode="json")
    response = await client.post(API + "/questions", json=payload, headers=identity_user)
    assert response.status_code == 201, response.text
    assert response.json()["turn_no"] == 1
    assert len(searches) == 1 and PERSON not in searches[0]
    question = await db.scalar(select(Question))
    expected = 1 if same else 2
    assert question.text.count("[ФИО_1]") == (2 if same else 1)
    assert ("[ФИО_2]" in question.text) is (expected == 2)
    saved = json.dumps({"text": question.text, "trace": question.trace}, ensure_ascii=False)
    saved += " ".join(await db.scalars(select(Answer.text)))
    assert "Иванов" not in saved and "participant-a" not in saved and "text_sha256" not in saved
    history = await client.get(f"{API}/dialogs/{dialog_id}/messages", headers=identity_user)
    assert history.status_code == 200 and history.json()["total"] == 1
    assert "Иванов" not in history.text and "person_resolution" not in history.text


@pytest.mark.parametrize("kind", ["voice", "photo"])
async def test_media_identity_challenge_prevents_legal_processing(
    client,
    db,
    identity_user,
    ai_registry,
    monkeypatch,
    kind,
):
    async def forbidden(*args, **kwargs):
        raise AssertionError("legal pipeline started before clarification")

    monkeypatch.setattr("app.modules.assistant.pipeline.Pipeline.answer", forbidden)
    ai_registry.get("whisper").transcript = TEXT
    ai_registry.get("ollama").document_text = TEXT
    response = await client.post(
        f"{API}/questions/{kind}",
        headers=identity_user,
        files={
            "file": (
                "source.ogg" if kind == "voice" else "source.jpg",
                b"fixture",
                "audio/ogg" if kind == "voice" else "image/jpeg",
            )
        },
    )
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "person_identity_required"
    assert "Иванов" not in response.text
    assert not ai_registry.get("ollama").chat_calls and not ai_registry.get("ollama").embed_calls
    assert await db.scalar(select(func.count()).select_from(Question)) == 0


@pytest.mark.parametrize("kind", ["text", "voice", "photo"])
async def test_invalid_resolution_is_422_without_echoing_raw_input(
    client,
    identity_user,
    ai_registry,
    kind,
):
    private = "Иванов_секретный_участник"
    invalid = {"text_sha256": "bad", "bindings": [{"start": -1, "end": 2, "participant": private}]}
    if kind == "text":
        response = await client.post(
            API + "/questions",
            json={"text": TEXT, "person_resolution": invalid},
            headers=identity_user,
        )
    else:
        response = await client.post(
            f"{API}/questions/{kind}",
            headers=identity_user,
            files={"file": ("fixture", b"fixture", "image/jpeg")},
            data={"person_resolution": json.dumps(invalid, ensure_ascii=False)},
        )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_person_resolution"
    assert "Иванов" not in response.text and private not in response.text
    assert not ai_registry.get("ollama").chat_calls
    assert not ai_registry.get("whisper").media_calls


async def test_stale_binding_is_422_without_question_record(client, db, identity_user):
    binding = resolution(TEXT).model_dump(mode="json")
    binding["text_sha256"] = "0" * 64
    response = await client.post(
        API + "/questions",
        json={"text": TEXT, "person_resolution": binding},
        headers=identity_user,
    )
    assert response.status_code == 422, response.text
    assert response.json()["detail"]["code"] == "invalid_person_resolution"
    assert "Иванов" not in response.text
    assert await db.scalar(select(func.count()).select_from(Question)) == 0


@pytest.mark.parametrize("kind", ["voice", "photo"])
async def test_malformed_resolution_json_is_safe_422_before_recognition(
    client,
    identity_user,
    ai_registry,
    kind,
):
    response = await client.post(
        f"{API}/questions/{kind}",
        headers=identity_user,
        files={"file": ("fixture.jpg", b"fixture", "image/jpeg")},
        data={"person_resolution": '{"private":"Иванов"'},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_person_resolution"
    assert "Иванов" not in response.text
    assert not ai_registry.get("ollama").media_calls
    assert not ai_registry.get("whisper").media_calls
