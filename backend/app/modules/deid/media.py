"""Голос и фото документа: распознать и сразу обезличить.

Распознавание выполняется внутри границы персональных данных (сервис deid):
голос и фото документа — такие же исходные данные пользователя, как текст вопроса.
"""

from dataclasses import dataclass
from typing import Any

from app.modules.ai.profiles import ModelRef
from app.modules.ai.providers.base import ProviderError
from app.modules.ai.registry import Registry
from app.modules.deid.service import DeidResult, deidentify

MAX_AUDIO_BYTES = 20 * 1024 * 1024
MAX_IMAGE_BYTES = 15 * 1024 * 1024
IMAGE_TYPES = ("image/jpeg", "image/png")
# распознанный документ длиннее — обрезаем: модели ответа нужен и бюджет на нормы
MAX_DOCUMENT_CHARS = 6000

DEFAULT_PHOTO_QUESTION = (
    "Проверь документ на соответствие трудовому законодательству: законен ли он "
    "и на что обратить внимание?"
)


class MediaRejected(Exception):
    """Ошибка во входных данных: не тот формат, пустая запись, нет текста на фото."""


@dataclass
class Recognized:
    provider: str
    model: str
    latency_ms: int
    chars: int
    truncated: bool = False

    def dump(self) -> dict[str, Any]:
        return self.__dict__.copy()


def with_document(question: str, document: str) -> str:
    return (
        f"{question.strip() or DEFAULT_PHOTO_QUESTION}\n\n"
        f"Текст документа (распознан с фото):\n{document}"
    )


async def from_audio(
    registry: Registry, audio: bytes, stt: ModelRef, llm: ModelRef | None
) -> tuple[DeidResult, Recognized]:
    if not audio:
        raise MediaRejected("пустой файл")
    if len(audio) > MAX_AUDIO_BYTES:
        raise MediaRejected("файл больше 20 МБ")
    result = await registry.get(stt.provider).transcribe(stt.model, audio, options=stt.options)
    if not result.text.strip():
        raise MediaRejected("речь не распознана")
    deid = await deidentify(result.text, _model(registry, llm))
    return deid, Recognized(result.provider, result.model, result.latency_ms, len(result.text))


async def from_image(
    registry: Registry,
    image: bytes,
    mime: str,
    question: str,
    ocr: ModelRef,
    llm: ModelRef | None,
) -> tuple[DeidResult, Recognized]:
    if mime not in IMAGE_TYPES:
        raise MediaRejected("нужна фотография в формате JPEG или PNG")
    if not image:
        raise MediaRejected("пустой файл")
    if len(image) > MAX_IMAGE_BYTES:
        raise MediaRejected("файл больше 15 МБ")
    result = await registry.get(ocr.provider).ocr(ocr.model, image, mime, options=ocr.options)
    document = result.text.strip()
    if not document:
        raise MediaRejected("на фото не найден текст")
    truncated = len(document) > MAX_DOCUMENT_CHARS
    deid = await deidentify(
        with_document(question, document[:MAX_DOCUMENT_CHARS]), _model(registry, llm)
    )
    recognized = Recognized(
        result.provider, result.model, result.latency_ms, len(document), truncated
    )
    return deid, recognized


def _model(registry: Registry, llm: ModelRef | None):
    if llm is None:
        return None
    try:
        return registry.get(llm.provider), llm
    except ProviderError as exc:
        raise MediaRejected(str(exc)) from exc
