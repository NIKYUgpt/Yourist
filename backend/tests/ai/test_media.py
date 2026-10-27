"""Распознавание речи и документов: аудио, запросы к провайдерам, Whisper."""

import base64
import io
import json
import math
import struct
import wave

import httpx
import pytest

from app.modules.ai.audio import BYTES_PER_SECOND, AudioError, decode_pcm16, split_pcm
from app.modules.ai.providers.base import ProviderError
from app.modules.ai.providers.whisper import WhisperProvider
from tests.ai.test_providers import registry


def wav(
    seconds: float, rate: int = 8000, channels: int = 2, silence_at: float | None = None
) -> bytes:
    """Тон 440 Гц; если silence_at задан — секунда тишины с этого места."""
    frames = bytearray()
    for i in range(int(seconds * rate)):
        t = i / rate
        quiet = silence_at is not None and silence_at <= t < silence_at + 1
        value = 0 if quiet else int(8000 * math.sin(2 * math.pi * 440 * t))
        frames += struct.pack("<" + "h" * channels, *([value] * channels))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(frames))
    return buf.getvalue()


def test_decode_resamples_to_16k_mono():
    pcm = decode_pcm16(wav(2.0))
    assert abs(len(pcm) - 2 * BYTES_PER_SECOND) < BYTES_PER_SECOND // 10


def test_decode_rejects_garbage_and_long_audio():
    with pytest.raises(AudioError):
        decode_pcm16(b"definitely not audio")
    with pytest.raises(AudioError, match="длиннее"):
        decode_pcm16(wav(3.0), max_seconds=1)


def test_split_cuts_in_silence():
    pcm = decode_pcm16(wav(30.0, silence_at=22.0))
    parts = split_pcm(pcm, max_seconds=25)
    assert len(parts) == 2 and b"".join(parts) == pcm
    # граница — внутри секунды тишины 22–23 с, а не ровно на 25-й секунде
    assert 22 * BYTES_PER_SECOND <= len(parts[0]) <= 23 * BYTES_PER_SECOND


async def test_yandex_speechkit_in_chunks():
    def handler(request):
        return httpx.Response(200, json={"result": f"часть {len(request.content)}"})

    reg, rec = registry(handler)
    result = await reg.get("yandex").transcribe("general", wav(55.0, rate=16000, channels=1))
    assert len(rec.requests) == 3  # 55 с → куски не длиннее 25 с
    req = rec.requests[0]
    assert req.url.host == "stt.api.cloud.yandex.net"
    assert dict(req.url.params) == {
        "lang": "ru-RU",
        "format": "lpcm",
        "sampleRateHertz": "16000",
        "folderId": "b1gfolder",
        "topic": "general",
    }
    assert req.headers["Authorization"] == "Api-Key yc-key"
    assert all(len(r.content) <= 25 * BYTES_PER_SECOND for r in rec.requests)
    assert result.text.count("часть") == 3 and result.provider == "yandex"


async def test_yandex_vision_ocr():
    def handler(request):
        return httpx.Response(
            200, json={"result": {"textAnnotation": {"fullText": " ПРИКАЗ № 15 \n"}}}
        )

    reg, rec = registry(handler)
    result = await reg.get("yandex").ocr("page", b"\xff\xd8jpeg", "image/jpeg")
    req = rec.requests[-1]
    assert str(req.url) == "https://ocr.api.cloud.yandex.net/ocr/v1/recognizeText"
    assert req.headers["x-folder-id"] == "b1gfolder"
    body = json.loads(req.content)
    assert body["mimeType"] == "JPEG" and body["model"] == "page"
    assert base64.b64decode(body["content"]) == b"\xff\xd8jpeg"
    assert result.text == "ПРИКАЗ № 15"


async def test_ollama_ocr_sends_image():
    def handler(request):
        return httpx.Response(200, json={"model": "qwen2.5vl:7b", "message": {"content": "Текст"}})

    reg, rec = registry(handler)
    result = await reg.get("ollama").ocr("qwen2.5vl:7b", b"png-bytes", "image/png")
    message = rec.body()["messages"][0]
    assert message["images"] == [base64.b64encode(b"png-bytes").decode()]
    assert "дословно" in message["content"]
    assert result.text == "Текст"


async def test_unsupported_media_tasks():
    reg, _ = registry(lambda r: httpx.Response(200, json={}))
    with pytest.raises(ProviderError, match="распознавание речи не поддерживается"):
        await reg.get("ollama").transcribe("x", b"audio")
    with pytest.raises(ProviderError, match="изображениях не поддерживается"):
        await reg.get("deepseek").ocr("x", b"img", "image/png")


async def test_whisper_loads_model_once_and_passes_options(monkeypatch):
    calls = {"load": [], "transcribe": []}

    class Segment:
        def __init__(self, text):
            self.text = text

    class FakeModel:
        def transcribe(self, samples, **params):
            calls["transcribe"].append((len(samples), params))
            return iter([Segment(" Меня уволили "), Segment("за прогул. ")]), None

    def fake_model(self, name, load):
        calls["load"].append((name, load))
        return FakeModel()

    monkeypatch.setattr(WhisperProvider, "_model", fake_model)
    provider = WhisperProvider()
    options = {"device": "cpu", "compute_type": "int8", "beam_size": 1}
    result = await provider.transcribe("large-v3-turbo", wav(1.0), options=options)
    assert result.text == "Меня уволили за прогул."
    assert calls["load"] == [("large-v3-turbo", {"device": "cpu", "compute_type": "int8"})]
    samples, params = calls["transcribe"][0]
    assert abs(samples - 16000) < 1600
    assert params == {"language": "ru", "beam_size": 1, "vad_filter": True}


async def test_whisper_reports_bad_audio():
    with pytest.raises(ProviderError, match="не удалось прочитать аудио"):
        await WhisperProvider().transcribe("tiny", b"garbage")
