import json
from datetime import timedelta
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.security import SERVICE, create_token
from app.deid_main import app as deid_app
from app.modules.ai.profiles import ModelRef
from app.modules.deid.api import DeidClient, DeidSettings, MediaError
from tests.fakes import fake_registry

WHISPER = ModelRef(provider="whisper", model="large-v3-turbo")
OCR = ModelRef(provider="ollama", model="qwen2.5vl:7b")


@pytest.fixture
def registry():
    registry = fake_registry()
    deid_app.state.ai_registry = registry
    return registry


def service() -> dict[str, str]:
    token = create_token(uuid4(), SERVICE, timedelta(minutes=1))
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
async def deid_http(registry):
    async with AsyncClient(transport=ASGITransport(app=deid_app), base_url="http://deid") as c:
        yield c


async def test_audio_is_transcribed_and_masked(deid_http, registry):
    resp = await deid_http.post(
        "/api/v1/deid/audio",
        data={"stt": WHISPER.model_dump_json()},
        files={"file": ("voice.ogg", b"opus-bytes", "audio/ogg")},
        headers=service(),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["text"].startswith("Меня зовут [ФИО_1], меня уволили")
    assert body["mapping"] == {"[ФИО_1]": "Петров Иван Иванович"}
    assert body["recognized"]["provider"] == "whisper"
    assert registry.get("whisper").media_calls == [
        {"kind": "stt", "model": "large-v3-turbo", "size": 10}
    ]


async def test_photo_text_is_joined_with_question_and_masked(deid_http, registry):
    resp = await deid_http.post(
        "/api/v1/deid/image",
        data={"ocr": OCR.model_dump_json(), "text": "Законен ли этот приказ?"},
        files={"file": ("order.jpg", b"\xff\xd8jpeg", "image/jpeg")},
        headers=service(),
    )
    assert resp.status_code == 200, resp.text
    text = resp.json()["text"]
    assert text.startswith("Законен ли этот приказ?\n\nТекст документа (распознан с фото):")
    assert "Сидоренко" not in text and "[ФИО_1]" in text


@pytest.mark.parametrize(
    ("file", "detail"),
    [
        (("doc.pdf", b"%PDF", "application/pdf"), "JPEG или PNG"),
        (("empty.png", b"", "image/png"), "пустой файл"),
    ],
)
async def test_bad_images_rejected(deid_http, file, detail):
    resp = await deid_http.post(
        "/api/v1/deid/image",
        data={"ocr": OCR.model_dump_json()},
        files={"file": file},
        headers=service(),
    )
    assert resp.status_code == 422 and detail in resp.json()["detail"]


async def test_media_requires_service_token(deid_http):
    resp = await deid_http.post(
        "/api/v1/deid/audio",
        data={"stt": WHISPER.model_dump_json()},
        files={"file": ("v.ogg", b"x", "audio/ogg")},
    )
    assert resp.status_code == 401


async def test_recognizer_failure_is_502(deid_http, registry):
    registry.get("whisper").fail = "model not downloaded"
    resp = await deid_http.post(
        "/api/v1/deid/audio",
        data={"stt": WHISPER.model_dump_json()},
        files={"file": ("v.ogg", b"x", "audio/ogg")},
        headers=service(),
    )
    assert resp.status_code == 502 and "model not downloaded" in resp.json()["detail"]


@pytest.mark.parametrize("url", ["http://deid", ""])
async def test_client_over_http_and_in_process(registry, url):
    client = DeidClient(DeidSettings(deid_url=url), registry, transport=ASGITransport(app=deid_app))
    voice = await client.from_audio(b"opus", "voice.ogg", WHISPER, None)
    assert voice.deid.mapping == {"[ФИО_1]": "Петров Иван Иванович"}
    photo = await client.from_image(b"\xff\xd8", "a.jpg", "image/jpeg", "", OCR, None)
    assert photo.deid.text.startswith("Проверь документ на соответствие")
    assert json.loads(json.dumps(photo.recognized))["chars"] > 0
    with pytest.raises(MediaError, match="JPEG или PNG"):
        await client.from_image(b"x", "a.gif", "image/gif", "", OCR, None)
    await client.aclose()
