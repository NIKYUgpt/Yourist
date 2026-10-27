"""Общий интерфейс провайдеров нейросетей: чат и эмбеддинги.

Каждый провайдер — тонкий клиент HTTP API. Тексты запросов в лог не пишем:
в них может оказаться то, что не поймала деперсонализация.
"""

import asyncio
import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class Kind(enum.StrEnum):
    LOCAL = "local"  # модель работает на нашем оборудовании
    API = "api"  # внешний сервис


class Capability(enum.StrEnum):
    CHAT = "chat"
    EMBED = "embed"


class EmbedKind(enum.StrEnum):
    # часть моделей (YandexGPT) по-разному кодирует документы и поисковые запросы
    DOCUMENT = "document"
    QUERY = "query"


@dataclass(frozen=True)
class ProviderInfo:
    name: str
    title: str
    kind: Kind
    capabilities: frozenset[Capability]
    # где обрабатываются данные: local — у нас, ru — в РФ, foreign — за рубежом
    location: str


@dataclass
class Usage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass
class ChatResult:
    text: str
    provider: str
    model: str
    latency_ms: int
    usage: Usage = field(default_factory=Usage)
    finish_reason: str | None = None


Message = dict[str, str]  # {"role": "system" | "user" | "assistant", "content": "..."}


class ProviderError(Exception):
    def __init__(self, provider: str, message: str, status: int | None = None):
        super().__init__(f"{provider}: {message}")
        self.provider = provider
        self.status = status


class NotSupported(ProviderError):
    pass


def merge_options(body: dict[str, Any], options: dict[str, Any] | None) -> dict[str, Any]:
    """Параметры из профиля поверх стандартных; None убирает параметр из запроса.

    Нужно для моделей с особыми требованиями: например, рассуждающие модели OpenAI
    принимают только температуру по умолчанию — для них в профиле {"temperature": null}.
    """
    for key, value in (options or {}).items():
        if value is None:
            body.pop(key, None)
        else:
            body[key] = value
    return body


class Provider:
    info: ProviderInfo

    def configured(self) -> bool:
        """Заданы ли адрес и ключи; без них провайдер не вызывается."""
        return True

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
        raise NotSupported(self.info.name, "чат не поддерживается")

    async def embed(
        self,
        model: str,
        texts: list[str],
        *,
        kind: EmbedKind = EmbedKind.DOCUMENT,
        options: dict[str, Any] | None = None,
    ) -> list[list[float]]:
        raise NotSupported(self.info.name, "эмбеддинги не поддерживаются")

    async def models(self) -> list[str]:
        """Модели, доступные у провайдера (для выбора в админке)."""
        return []

    async def aclose(self) -> None:
        pass


_RETRY_STATUSES = {408, 409, 429, 500, 502, 503, 504}


class HttpProvider(Provider):
    """Провайдер поверх HTTP: общий клиент, повторы при сбоях, разбор ошибок."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float,
        retries: int,
        proxy: str | None = None,
        verify: Any = True,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.retries = retries
        self.proxy = proxy or None
        self.http = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=15.0),
            proxy=self.proxy,
            verify=verify,
            transport=transport,
        )

    async def _headers(self) -> dict[str, str]:
        return {}

    async def _on_unauthorized(self) -> bool:
        """Вызывается при 401; True — повторить запрос (например, после обновления токена)."""
        return False

    async def request(
        self, method: str, path: str, *, json: Any = None, headers: dict[str, str] | None = None
    ) -> Any:
        url = path if path.startswith("http") else f"{self.base_url}{path}"
        name = self.info.name
        reauthorized = False
        attempt = 0
        while True:
            try:
                resp = await self.http.request(
                    method, url, json=json, headers={**await self._headers(), **(headers or {})}
                )
            except httpx.TransportError as exc:
                if attempt < self.retries:
                    attempt += 1
                    await asyncio.sleep(2**attempt)
                    continue
                raise ProviderError(name, f"нет связи: {type(exc).__name__}: {exc}") from exc

            if resp.status_code == 401 and not reauthorized and await self._on_unauthorized():
                reauthorized = True
                continue
            if resp.status_code in _RETRY_STATUSES and attempt < self.retries:
                attempt += 1
                await asyncio.sleep(2**attempt)
                continue
            if resp.status_code >= 400:
                raise ProviderError(
                    name, f"HTTP {resp.status_code}: {resp.text[:500]}", status=resp.status_code
                )
            try:
                return resp.json()
            except ValueError as exc:
                raise ProviderError(name, f"ответ не JSON: {resp.text[:200]}") from exc

    async def aclose(self) -> None:
        await self.http.aclose()


class Timer:
    def __init__(self) -> None:
        self.start = time.perf_counter()

    @property
    def ms(self) -> int:
        return int((time.perf_counter() - self.start) * 1000)
