"""Провайдеры с OpenAI-совместимым API: /chat/completions, /embeddings, /models.

Такой интерфейс есть у OpenAI, DeepSeek, Qwen (Alibaba Cloud Model Studio),
Gemini (слой совместимости Google), YandexGPT (AI Studio) и GigaChat.
Различаются адрес, авторизация, имена моделей и мелкие детали — они
вынесены в подклассы.
"""

from typing import Any

from app.modules.ai.providers.base import (
    Capability,
    ChatResult,
    EmbedKind,
    HttpProvider,
    Kind,
    Message,
    ProviderError,
    ProviderInfo,
    Timer,
    Usage,
    merge_options,
)

CHAT_AND_EMBED = frozenset({Capability.CHAT, Capability.EMBED})
CHAT_ONLY = frozenset({Capability.CHAT})


class OpenAICompatProvider(HttpProvider):
    # сколько текстов отправлять в одном запросе к /embeddings
    embed_batch = 32
    max_tokens_field = "max_tokens"
    supports_json_mode = True

    def __init__(self, base_url: str, api_key: str, **kwargs: Any):
        super().__init__(base_url, **kwargs)
        self.api_key = api_key

    def configured(self) -> bool:
        return bool(self.api_key)

    async def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def chat_model(self, model: str) -> str:
        return model

    def embed_model(self, model: str, kind: EmbedKind) -> str:
        return model

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
        if not self.configured():
            raise ProviderError(self.info.name, "не заданы ключи доступа")
        body: dict[str, Any] = {
            "model": self.chat_model(model),
            "messages": messages,
            "temperature": temperature,
            self.max_tokens_field: max_tokens,
        }
        if json_mode and self.supports_json_mode:
            body["response_format"] = {"type": "json_object"}
        merge_options(body, options)
        timer = Timer()
        data = await self.request("POST", "/chat/completions", json=body)
        try:
            choice = data["choices"][0]
            text = choice["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(self.info.name, f"неожиданный ответ: {str(data)[:200]}") from exc
        usage = data.get("usage") or {}
        return ChatResult(
            text=text,
            provider=self.info.name,
            model=data.get("model") or model,
            latency_ms=timer.ms,
            usage=Usage(usage.get("prompt_tokens"), usage.get("completion_tokens")),
            finish_reason=choice.get("finish_reason"),
        )

    async def embed(
        self,
        model: str,
        texts: list[str],
        *,
        kind: EmbedKind = EmbedKind.DOCUMENT,
        options: dict[str, Any] | None = None,
    ) -> list[list[float]]:
        if Capability.EMBED not in self.info.capabilities:
            return await super().embed(model, texts, kind=kind, options=options)
        if not self.configured():
            raise ProviderError(self.info.name, "не заданы ключи доступа")
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.embed_batch):
            batch = texts[start : start + self.embed_batch]
            body = merge_options({"model": self.embed_model(model, kind), "input": batch}, options)
            data = await self.request("POST", "/embeddings", json=body)
            items = sorted(data.get("data", []), key=lambda item: item.get("index", 0))
            if len(items) != len(batch):
                raise ProviderError(self.info.name, "число векторов не совпадает с числом текстов")
            vectors.extend(item["embedding"] for item in items)
        return vectors

    async def models(self) -> list[str]:
        data = await self.request("GET", "/models")
        return sorted(m["id"] for m in data.get("data", []))


class OpenAIProvider(OpenAICompatProvider):
    info = ProviderInfo(
        name="openai",
        title="OpenAI (ChatGPT)",
        kind=Kind.API,
        capabilities=CHAT_AND_EMBED,
        location="foreign",
    )
    # новые модели OpenAI не принимают max_tokens
    max_tokens_field = "max_completion_tokens"


class DeepSeekProvider(OpenAICompatProvider):
    info = ProviderInfo(
        name="deepseek",
        title="DeepSeek",
        kind=Kind.API,
        capabilities=CHAT_ONLY,
        location="foreign",
    )


class QwenProvider(OpenAICompatProvider):
    info = ProviderInfo(
        name="qwen",
        title="Qwen (Alibaba Cloud Model Studio)",
        kind=Kind.API,
        capabilities=CHAT_AND_EMBED,
        location="foreign",
    )
    # text-embedding-v4 принимает не больше 10 строк за запрос
    embed_batch = 10


class GeminiProvider(OpenAICompatProvider):
    info = ProviderInfo(
        name="gemini",
        title="Google Gemini",
        kind=Kind.API,
        capabilities=CHAT_AND_EMBED,
        location="foreign",
    )


class YandexProvider(OpenAICompatProvider):
    """YandexGPT через OpenAI-совместимый API AI Studio.

    Модели адресуются URI с каталогом: gpt://<folder>/yandexgpt/latest для генерации,
    emb://<folder>/text-search-doc/latest и …-query/latest для эмбеддингов
    (так же строит URI официальный SDK yandex-ai-studio-sdk).
    """

    info = ProviderInfo(
        name="yandex",
        title="YandexGPT (Yandex AI Studio)",
        kind=Kind.API,
        capabilities=CHAT_AND_EMBED,
        location="ru",
    )
    # /embeddings Яндекса принимает ровно одну строку за запрос (проверено 27.10.2026:
    # на список отвечает 400 «Array input must contain exactly one string»)
    embed_batch = 1
    # модели-пары: отдельная модель для документов (-doc) и для запросов (-query)
    paired_embeddings = ("text-search", "text-embeddings-v2")

    def __init__(self, base_url: str, api_key: str, folder_id: str, **kwargs: Any):
        super().__init__(base_url, api_key, **kwargs)
        self.folder_id = folder_id

    def configured(self) -> bool:
        return bool(self.api_key and self.folder_id)

    async def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Api-Key {self.api_key}",
            "OpenAI-Project": self.folder_id,
            # не разрешаем Яндексу сохранять запросы для улучшения моделей
            "x-data-logging-enabled": "false",
        }

    def _uri(self, prefix: str, model: str) -> str:
        if "://" in model:
            return model
        if "/" not in model:
            model = f"{model}/latest"
        return f"{prefix}://{self.folder_id}/{model}"

    def chat_model(self, model: str) -> str:
        return self._uri("gpt", model)

    def embed_model(self, model: str, kind: EmbedKind) -> str:
        if model in self.paired_embeddings:
            model = f"{model}-doc" if kind == EmbedKind.DOCUMENT else f"{model}-query"
        return self._uri("emb", model)
