from sqlalchemy import select

from app.modules.ai.profiles import Mode
from app.modules.ai.service import load_config, save_config
from app.modules.assistant.models import Question
from tests.assistant.test_pipeline import answer_calls, good_answer, service_reply
from tests.conftest import bearer, login, register_verified

API = "/api/v1/assistant/questions"


async def make_user(client, mailer) -> dict[str, str]:
    await register_verified(client, mailer, "voice@example.com")
    return bearer(await login(client, "voice@example.com"))


def setup_models(ai_registry):
    ollama = ai_registry.get("ollama")
    ollama.reply = lambda m: service_reply(m) or good_answer(m)
    return ollama


async def test_voice_question(client, db, mailer, indexed, ai_registry):
    ollama = setup_models(ai_registry)
    headers = await make_user(client, mailer)
    resp = await client.post(
        f"{API}/voice",
        files={"file": ("voice.ogg", b"opus-bytes", "audio/ogg")},
        data={"date": "2026-10-01"},
        headers=headers,
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["source"] == "voice"
    # пользователю — распознанный текст с его данными
    assert body["text"].startswith("Меня зовут Петров Иван Иванович")
    assert body["answer"]["text"].startswith("Да, Петров Иван Иванович можно уволить")
    assert body["trace"]["recognize"]["provider"] == "whisper"
    assert ai_registry.get("whisper").media_calls  # локальный режим — Whisper
    prompt = answer_calls(ollama)[0]["messages"][1]["content"]
    assert "Петров" not in prompt and "[ФИО_1]" in prompt
    stored = await db.scalar(select(Question))
    assert stored.source == "voice" and "Петров" not in stored.text


async def test_photo_question_in_api_mode(client, db, mailer, indexed, ai_registry):
    setup_models(ai_registry)
    config = await load_config(db)
    config.mode = Mode.API
    await save_config(db, config, None)
    ai_registry.get("gigachat").reply = lambda m: service_reply(m) or good_answer(m)
    ai_registry.get("yandex").reply = lambda m: service_reply(m) or good_answer(m)

    headers = await make_user(client, mailer)
    resp = await client.post(
        f"{API}/photo",
        files={"file": ("order.jpg", b"\xff\xd8jpeg", "image/jpeg")},
        data={"text": "Законно ли меня уволили по этому приказу?"},
        headers=headers,
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["source"] == "photo"
    assert "Сидоренко Анны Петровны" in body["text"]  # автору — с данными
    assert ai_registry.get("yandex").media_calls[0] == {
        "kind": "ocr",
        "model": "page",
        "mime": "image/jpeg",
    }
    stored = await db.scalar(select(Question))
    assert "Сидоренко" not in stored.text and "Текст документа" in stored.text


async def test_bad_media_is_422(client, mailer, indexed, ai_registry):
    setup_models(ai_registry)
    headers = await make_user(client, mailer)
    resp = await client.post(
        f"{API}/photo", files={"file": ("a.gif", b"GIF89a", "image/gif")}, headers=headers
    )
    assert resp.status_code == 422 and "JPEG или PNG" in resp.json()["detail"]
    ai_registry.get("whisper").transcript = "   "
    resp = await client.post(
        f"{API}/voice", files={"file": ("v.ogg", b"x", "audio/ogg")}, headers=headers
    )
    assert resp.status_code == 422 and "речь не распознана" in resp.json()["detail"]


async def test_recognizer_down_is_503(client, mailer, indexed, ai_registry):
    setup_models(ai_registry)
    ai_registry.get("whisper").fail = "connection refused"
    headers = await make_user(client, mailer)
    resp = await client.post(
        f"{API}/voice", files={"file": ("v.ogg", b"x", "audio/ogg")}, headers=headers
    )
    assert resp.status_code == 503 and "распознавание недоступно" in resp.json()["detail"]
