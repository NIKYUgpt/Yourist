"""Реестр провайдеров: создаётся один раз на процесс из настроек окружения."""

from typing import Any

import httpx

from app.modules.ai.config import ProviderSettings
from app.modules.ai.providers.anthropic import AnthropicProvider
from app.modules.ai.providers.base import Provider, ProviderError, ProviderInfo
from app.modules.ai.providers.gigachat import GigaChatProvider, gigachat_verify
from app.modules.ai.providers.ollama import OllamaProvider
from app.modules.ai.providers.openai_compat import (
    DeepSeekProvider,
    GeminiProvider,
    OpenAIProvider,
    QwenProvider,
    YandexProvider,
)
from app.modules.ai.providers.whisper import WhisperProvider


class Registry:
    def __init__(self, providers: dict[str, Provider]):
        self.providers = providers

    def get(self, name: str) -> Provider:
        try:
            return self.providers[name]
        except KeyError:
            raise ProviderError(name, "провайдер не подключён") from None

    def infos(self) -> list[ProviderInfo]:
        return [p.info for p in self.providers.values()]

    async def aclose(self) -> None:
        for provider in self.providers.values():
            await provider.aclose()


def build_registry(
    settings: ProviderSettings, transport: httpx.AsyncBaseTransport | None = None
) -> Registry:
    def http(name: str, **extra: Any) -> dict[str, Any]:
        proxy = settings.ai_proxy_url if name in settings.ai_proxy_providers else None
        return {
            "timeout": settings.ai_timeout_sec,
            "retries": settings.ai_retries,
            "proxy": proxy,
            "transport": transport,
            **extra,
        }

    providers: list[Provider] = [
        OllamaProvider(
            settings.ollama_url,
            num_ctx=settings.ollama_num_ctx,
            keep_alive=settings.ollama_keep_alive,
            **http("ollama"),
        ),
        WhisperProvider(settings.whisper_download_root),
        YandexProvider(
            settings.yandex_base_url,
            settings.yandex_api_key,
            settings.yandex_folder_id,
            **http("yandex"),
        ),
        GigaChatProvider(
            settings.gigachat_base_url,
            credentials=settings.gigachat_credentials,
            scope=settings.gigachat_scope,
            auth_url=settings.gigachat_auth_url,
            **http(
                "gigachat",
                verify=gigachat_verify(settings.gigachat_ca_bundle, settings.gigachat_verify_ssl),
            ),
        ),
        OpenAIProvider(settings.openai_base_url, settings.openai_api_key, **http("openai")),
        AnthropicProvider(
            settings.anthropic_base_url, settings.anthropic_api_key, **http("anthropic")
        ),
        GeminiProvider(settings.gemini_base_url, settings.gemini_api_key, **http("gemini")),
        DeepSeekProvider(settings.deepseek_base_url, settings.deepseek_api_key, **http("deepseek")),
        QwenProvider(settings.qwen_base_url, settings.qwen_api_key, **http("qwen")),
    ]
    return Registry({p.info.name: p for p in providers})
