"""Локальные модели через Ollama (https://github.com/ollama/ollama/blob/main/docs/api.md)."""

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
)

# параметры запроса верхнего уровня; остальное из профиля уходит в options модели
_TOP_LEVEL = {"think", "keep_alive", "format"}


class OllamaProvider(HttpProvider):
    info = ProviderInfo(
        name="ollama",
        title="Ollama (локальные модели)",
        kind=Kind.LOCAL,
        capabilities=frozenset({Capability.CHAT, Capability.EMBED}),
        location="local",
    )

    def __init__(self, base_url: str, *, num_ctx: int, keep_alive: str, **kwargs: Any):
        super().__init__(base_url, **kwargs)
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive

    def _split(self, options: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
        top, model_opts = {}, {}
        for key, value in (options or {}).items():
            (top if key in _TOP_LEVEL else model_opts)[key] = value
        return top, model_opts

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
        top, model_opts = self._split(options)
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": temperature,
                "num_ctx": self.num_ctx,
                "num_predict": max_tokens,
                **model_opts,
            },
            **top,
        }
        if json_mode:
            body["format"] = "json"
        timer = Timer()
        data = await self.request("POST", "/api/chat", json=body)
        try:
            text = data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise ProviderError(self.info.name, f"неожиданный ответ: {str(data)[:200]}") from exc
        return ChatResult(
            text=text,
            provider=self.info.name,
            model=data.get("model", model),
            latency_ms=timer.ms,
            usage=Usage(data.get("prompt_eval_count"), data.get("eval_count")),
            finish_reason=data.get("done_reason"),
        )

    async def embed(
        self,
        model: str,
        texts: list[str],
        *,
        kind: EmbedKind = EmbedKind.DOCUMENT,
        options: dict[str, Any] | None = None,
    ) -> list[list[float]]:
        top, model_opts = self._split(options)
        body = {"model": model, "input": texts, "keep_alive": self.keep_alive, **top}
        if model_opts:
            body["options"] = model_opts
        data = await self.request("POST", "/api/embed", json=body)
        vectors = data.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise ProviderError(self.info.name, "число векторов не совпадает с числом текстов")
        return vectors

    async def models(self) -> list[str]:
        data = await self.request("GET", "/api/tags")
        return sorted(m["name"] for m in data.get("models", []))
