"""Claude (Anthropic) через Messages API (https://docs.anthropic.com/en/api/messages).

Эмбеддингов у Anthropic нет — для поиска нужен другой провайдер.
"""

from typing import Any

from app.modules.ai.providers.base import (
    Capability,
    ChatResult,
    HttpProvider,
    Kind,
    Message,
    ProviderError,
    ProviderInfo,
    Timer,
    Usage,
    merge_options,
)

API_VERSION = "2023-06-01"


class AnthropicProvider(HttpProvider):
    info = ProviderInfo(
        name="anthropic",
        title="Claude (Anthropic)",
        kind=Kind.API,
        capabilities=frozenset({Capability.CHAT}),
        location="foreign",
    )

    def __init__(self, base_url: str, api_key: str, **kwargs: Any):
        super().__init__(base_url, **kwargs)
        self.api_key = api_key

    def configured(self) -> bool:
        return bool(self.api_key)

    async def _headers(self) -> dict[str, str]:
        return {"x-api-key": self.api_key, "anthropic-version": API_VERSION}

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
            raise ProviderError(self.info.name, "не задан ключ API")
        # системные сообщения у Anthropic — отдельное поле, а не роль в списке
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        body: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [m for m in messages if m["role"] != "system"],
        }
        if system:
            body["system"] = system
        merge_options(body, options)
        timer = Timer()
        data = await self.request("POST", "/v1/messages", json=body)
        try:
            text = "".join(b["text"] for b in data["content"] if b.get("type") == "text")
        except (KeyError, TypeError) as exc:
            raise ProviderError(self.info.name, f"неожиданный ответ: {str(data)[:200]}") from exc
        usage = data.get("usage") or {}
        return ChatResult(
            text=text,
            provider=self.info.name,
            model=data.get("model") or model,
            latency_ms=timer.ms,
            usage=Usage(usage.get("input_tokens"), usage.get("output_tokens")),
            finish_reason=data.get("stop_reason"),
        )

    async def models(self) -> list[str]:
        data = await self.request("GET", "/v1/models")
        return sorted(m["id"] for m in data.get("data", []))
