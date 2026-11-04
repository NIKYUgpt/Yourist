import json
from datetime import timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.time import utcnow
from app.main import app
from app.modules.ai.service import load_config, save_config
from app.modules.assistant.dialogs import MAX_HISTORY_CHARS, MAX_HISTORY_TURNS, bounded_history
from app.modules.assistant.models import Answer, Dialog, Question
from tests.assistant.test_pipeline import (
    QUESTION,
    good_answer,
    is_analysis,
    is_verify,
    service_reply,
)
from tests.conftest import bearer, login, register_verified

API = "/api/v1/assistant"


@pytest.fixture
async def user(client, mailer):
    await register_verified(client, mailer, "dialog-user@example.com")
    return bearer(await login(client, "dialog-user@example.com"))


def is_context(messages):
    return "поисковый запрос по уточнению" in messages[0]["content"]


@pytest.fixture
def model(ai_registry):
    provider = ai_registry.get("ollama")

    def reply(messages):
        if is_context(messages):
            return '{"query": "Увольнение за прогул, отсутствие менее четырёх часов"}'
        if is_analysis(messages):
            return '{"issues": []}'
        service = service_reply(messages)
        if service is not None:
            return service
        latest = next(m for m in reversed(messages) if "Источники (редакция" in m["content"])
        return good_answer([messages[0], latest])

    provider.reply = reply
    return provider


async def new_dialog(client, user):
    response = await client.post(API + "/dialogs", json={"date": "2026-10-01"}, headers=user)
    assert response.status_code == 201, response.text
    assert "deid_session_id" not in response.json()
    return response.json()["id"]


async def ask(client, user, dialog_id, text, **extra):
    return await client.post(
        API + "/questions", json={"dialog_id": dialog_id, "text": text, **extra}, headers=user
    )


@pytest.mark.parametrize("lost_topic", [False, True])
async def test_dialog_context_and_stable_private_labels(
    client, db, indexed, model, user, lost_topic
):
    if lost_topic:
        original_reply = model.reply
        model.reply = lambda messages: (
            '{"query": "К чему относится слово если?"}'
            if is_context(messages)
            else original_reply(messages)
        )
    dialog_id = await new_dialog(client, user)
    first = await ask(client, user, dialog_id, QUESTION)
    assert first.status_code == 201, first.text
    second = await ask(client, user, dialog_id, "А если только три часа? Телефон 89167654321.")
    assert second.status_code == 201, second.text
    body = second.json()
    assert body["dialog_id"] == dialog_id and body["turn_no"] == 2
    assert body["on_date"] == "2026-10-01"
    assert body["trace"]["context"]["history_turns"] == 1
    assert not body["trace"]["context"]["fallback"]
    # Даже корректная по форме переформулировка может потерять тему: сохраняем
    # дополнительный запрос из слов пользователя, без прежних ответов модели.
    queries = body["trace"]["context"]["retrieval_queries"]
    assert len(queries) == 2
    # Для текущего ответа доступны и исходные значения предыдущей реплики.
    assert "Петрова Сергея Ивановича" in body["answer"]["text"]
    questions = list(await db.scalars(select(Question).order_by(Question.turn_no)))
    assert questions[0].text in queries[-1] and questions[1].text in queries[-1]
    assert "[ТЕЛЕФОН_1]" in questions[0].text
    assert "[ТЕЛЕФОН_2]" in questions[1].text
    assert [q.turn_no for q in questions] == [1, 2]
    stored = json.dumps([{"text": q.text, "trace": q.trace} for q in questions], ensure_ascii=False)
    stored += " ".join(await db.scalars(select(Answer.text)))
    assert "Петров" not in stored and "89161234567" not in stored and "89167654321" not in stored
    all_prompts = json.dumps([c["messages"] for c in model.chat_calls], ensure_ascii=False)
    assert "Петров" not in all_prompts and "89161234567" not in all_prompts
    [rewrite] = [c for c in model.chat_calls if is_context(c["messages"])]
    assert "[ФИО_1]" in rewrite["messages"][-1]["content"]
    verify = [c for c in model.chat_calls if is_verify(c["messages"])][-1]
    assert "Предыдущие реплики" in verify["messages"][-1]["content"]
    assert questions[0].text in verify["messages"][-1]["content"]
    history = (await client.get(f"{API}/dialogs/{dialog_id}/messages", headers=user)).json()
    assert history["total"] == 2 and [q["turn_no"] for q in history["items"]] == [1, 2]
    assert "Петров" not in json.dumps(history, ensure_ascii=False)
    assert all(q["trace"] is None for q in history["items"])
    page = (
        await client.get(f"{API}/dialogs/{dialog_id}/messages?offset=1&limit=1", headers=user)
    ).json()
    assert len(page["items"]) == 1 and page["items"][0]["turn_no"] == 2


async def test_ownership_and_delete_removes_history_and_mapping(
    client, db, indexed, model, user, mailer
):
    dialog_id = await new_dialog(client, user)
    assert (await ask(client, user, dialog_id, QUESTION)).status_code == 201
    row = await db.scalar(select(Dialog))
    sid = row.deid_session_id
    await register_verified(client, mailer, "outsider@example.com")
    other = bearer(await login(client, "outsider@example.com"))
    assert (await client.get(API + "/dialogs", headers=other)).json()["total"] == 0
    for path in [f"/dialogs/{dialog_id}", f"/dialogs/{dialog_id}/messages"]:
        assert (await client.get(API + path, headers=other)).status_code == 404
    assert (await ask(client, other, dialog_id, "А потом?")).status_code == 404
    assert (await client.delete(f"{API}/dialogs/{dialog_id}", headers=other)).status_code == 404
    assert sid in app.state.deid_client.sessions.entries
    qid = (await db.scalar(select(Question))).id
    assert (await client.delete(f"{API}/questions/{qid}", headers=user)).status_code == 409
    assert (await client.delete(f"{API}/dialogs/{dialog_id}", headers=user)).status_code == 204
    assert sid not in app.state.deid_client.sessions.entries
    for cls in (Dialog, Question, Answer):
        assert await db.scalar(select(cls)) is None


@pytest.mark.parametrize("reason", ["expired", "restart", "wrong_date"])
async def test_unusable_session_or_changed_date_fails_before_models(
    client, db, indexed, model, user, reason
):
    dialog_id = await new_dialog(client, user)
    row = await db.scalar(select(Dialog))
    extra = {}
    if reason == "expired":
        row.expires_at = utcnow() - timedelta(seconds=1)
        await db.commit()
    elif reason == "restart":
        app.state.deid_client.sessions.remove(row.deid_session_id, row.user_id)
    else:
        extra["date"] = "2025-01-01"
    response = await ask(client, user, dialog_id, QUESTION, **extra)
    assert response.status_code == (422 if reason == "wrong_date" else 409)
    assert model.chat_calls == []
    assert await db.scalar(select(Question)) is None
    await db.refresh(row)
    assert row.next_turn == 1


async def test_parallel_turn_rejected_by_database_lock(client, db, indexed, model, user, engine):
    dialog_id = await new_dialog(client, user)
    async with AsyncSession(engine) as lock_session:
        await lock_session.scalar(select(Dialog).where(Dialog.id == dialog_id).with_for_update())
        assert (await ask(client, user, dialog_id, QUESTION)).status_code == 409
        assert (await client.delete(f"{API}/dialogs/{dialog_id}", headers=user)).status_code == 409
        assert not model.chat_calls
        await lock_session.rollback()
    assert (await ask(client, user, dialog_id, QUESTION)).status_code == 201


@pytest.mark.parametrize("disabled", [False, True])
async def test_rewrite_failure_or_disabled_analysis_keeps_original_context(
    client, db, indexed, model, user, disabled
):
    dialog_id = await new_dialog(client, user)
    await ask(client, user, dialog_id, QUESTION)
    original_reply = model.reply
    model.reply = lambda m: "not json" if is_context(m) else original_reply(m)
    if disabled:
        config = await load_config(db)
        config.analyze = False
        await save_config(db, config, None)
    response = await ask(client, user, dialog_id, "А если только три часа?")
    assert response.status_code == 201, response.text
    context = response.json()["trace"]["context"]
    assert context["fallback"] and "А если только три часа?" in context["query"]
    assert "[ФИО_1]" in context["query"]


@pytest.mark.parametrize("kind", ["voice", "photo"])
async def test_media_shares_dialog_mapping(client, db, indexed, model, user, ai_registry, kind):
    dialog_id = await new_dialog(client, user)
    await ask(client, user, dialog_id, QUESTION)
    response = await client.post(
        f"{API}/questions/{kind}",
        headers=user,
        data={"dialog_id": dialog_id},
        files={
            "file": ("voice.wav", b"audio", "audio/wav")
            if kind == "voice"
            else ("order.jpg", b"jpeg", "image/jpeg")
        },
    )
    assert response.status_code == 201, response.text
    assert response.json()["source"] == kind and response.json()["turn_no"] == 2
    question = await db.scalar(select(Question).where(Question.turn_no == 2))
    assert "[ФИО_2]" in question.text
    assert "Петров" not in question.text and "Сидоренко" not in question.text


def test_history_is_bounded_and_keeps_recent_questions():
    history = bounded_history([(f"question-{i}", "a" * 5000) for i in range(12)])
    assert sum(m["role"] == "user" for m in history) <= MAX_HISTORY_TURNS
    assert sum(len(m["content"]) for m in history) <= MAX_HISTORY_CHARS
    assert history[-2]["content"] == "question-11"
    assert all(m["content"] != "question-0" for m in history)


async def test_dialogs_require_auth(client):
    assert (await client.post(API + "/dialogs", json={})).status_code == 401
