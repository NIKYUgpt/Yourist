"""Подменные провайдеры нейросетей: детерминированные, без сети."""

import hashlib
import math
import re
from collections.abc import Callable
from typing import Any

from app.modules.ai.providers.base import (
    Capability,
    ChatResult,
    EmbedKind,
    Kind,
    Message,
    Provider,
    ProviderError,
    ProviderInfo,
    Usage,
)
from app.modules.ai.registry import Registry

DIM = 256
_WORD = re.compile(r"[а-яёa-z0-9]+")


def bag_of_words(text: str) -> list[float]:
    """Вектор «мешка слов» по первым 5 буквам слова: увольнение ~ увольнения."""
    vec = [0.0] * DIM
    for word in _WORD.findall(text.lower()):
        if len(word) < 3:
            continue
        digest = hashlib.md5(word[:5].encode()).digest()
        vec[int.from_bytes(digest[:4], "little") % DIM] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class FakeProvider(Provider):
    def __init__(
        self,
        name: str,
        kind: Kind = Kind.LOCAL,
        location: str = "local",
        capabilities: frozenset[Capability] = frozenset({Capability.CHAT, Capability.EMBED}),
        configured: bool = True,
    ):
        self.info = ProviderInfo(
            name=name, title=name, kind=kind, capabilities=capabilities, location=location
        )
        self._configured = configured
        # ответ чата: функция от сообщений; по умолчанию — эхо последнего сообщения
        self.reply: Callable[[list[Message]], str] = lambda messages: messages[-1]["content"]
        self.chat_calls: list[dict[str, Any]] = []
        self.embed_calls: list[dict[str, Any]] = []
        self.fail: str | None = None

    def configured(self) -> bool:
        return self._configured

    async def chat(
        self,
        model: str,
        messages: list[Message],
        *,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        json_mode: bool = False,
        options: dict[str, Any] | None = None,
    ) -> ChatResult:
        if self.fail:
            raise ProviderError(self.info.name, self.fail)
        self.chat_calls.append(
            {"model": model, "messages": messages, "json_mode": json_mode, "options": options}
        )
        return ChatResult(
            text=self.reply(messages),
            provider=self.info.name,
            model=model,
            latency_ms=1,
            usage=Usage(10, 5),
            finish_reason="stop",
        )

    async def embed(
        self,
        model: str,
        texts: list[str],
        *,
        kind: EmbedKind = EmbedKind.DOCUMENT,
        options: dict[str, Any] | None = None,
    ) -> list[list[float]]:
        if self.fail:
            raise ProviderError(self.info.name, self.fail)
        if Capability.EMBED not in self.info.capabilities:
            return await super().embed(model, texts, kind=kind)
        self.embed_calls.append({"model": model, "texts": texts, "kind": kind})
        return [bag_of_words(t) for t in texts]

    async def models(self) -> list[str]:
        return ["fake-small", "fake-large"]


def fake_registry() -> Registry:
    chat_only = frozenset({Capability.CHAT})
    providers = [
        FakeProvider("ollama"),
        FakeProvider("yandex", Kind.API, "ru"),
        FakeProvider("gigachat", Kind.API, "ru"),
        FakeProvider("deepseek", Kind.API, "foreign", chat_only),
        FakeProvider("openai", Kind.API, "foreign", configured=False),
    ]
    return Registry({p.info.name: p for p in providers})
