"""Локальное распознавание речи: Whisper через faster-whisper (CTranslate2).

Модель скачивается с Hugging Face при первом запросе и кэшируется. По умолчанию
работает на процессоре в int8: видеокарта занята моделью ответа, а голосовой
вопрос в 20–30 секунд распознаётся за секунды.
"""

import asyncio
from typing import Any

from app.modules.ai.audio import AudioError, decode_pcm16
from app.modules.ai.providers.base import (
    Capability,
    ChatResult,
    Kind,
    Provider,
    ProviderError,
    ProviderInfo,
    Timer,
)

# параметры загрузки модели; остальные опции профиля уходят в transcribe()
_LOAD_OPTIONS = ("device", "compute_type", "cpu_threads")


class WhisperProvider(Provider):
    info = ProviderInfo(
        name="whisper",
        title="Whisper (локально, faster-whisper)",
        kind=Kind.LOCAL,
        capabilities=frozenset({Capability.STT}),
        location="local",
    )

    def __init__(self, download_root: str | None = None):
        self.download_root = download_root or None
        self._models: dict[tuple, Any] = {}
        # одна модель на процессе и распознавание по одному: память и ядра не резиновые
        self._lock = asyncio.Lock()

    def _model(self, name: str, load: dict[str, Any]):
        from faster_whisper import WhisperModel  # тяжёлый импорт — только при первом запросе

        key = (name, *sorted(load.items()))
        if key not in self._models:
            self._models[key] = WhisperModel(name, download_root=self.download_root, **load)
        return self._models[key]

    async def transcribe(
        self, model: str, audio: bytes, *, options: dict[str, Any] | None = None
    ) -> ChatResult:
        opts = {"device": "cpu", "compute_type": "int8", **(options or {})}
        load = {k: opts.pop(k) for k in _LOAD_OPTIONS if k in opts}
        params = {"language": "ru", "beam_size": 5, "vad_filter": True, **opts}

        def run() -> str:
            import numpy as np

            pcm = decode_pcm16(audio)
            samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            segments, _ = self._model(model, load).transcribe(samples, **params)
            return " ".join(segment.text.strip() for segment in segments)

        timer = Timer()
        async with self._lock:
            try:
                text = await asyncio.to_thread(run)
            except AudioError as exc:
                raise ProviderError(self.info.name, str(exc)) from exc
            except Exception as exc:  # ошибки загрузки модели, CTranslate2
                raise ProviderError(self.info.name, f"{type(exc).__name__}: {exc}") from exc
        return ChatResult(
            text=text.strip(), provider=self.info.name, model=model, latency_ms=timer.ms
        )
