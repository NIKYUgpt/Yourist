"""Подключение к провайдерам нейросетей: адреса и ключи.

Секреты живут только в окружении (.env). В БД и в админке — лишь выбор
провайдера и модели для каждой задачи (см. profiles.py).
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class ProviderSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=("../.env", ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    # локальные модели; из контейнера Ollama на хосте доступна по host.docker.internal
    ollama_url: str = "http://127.0.0.1:11434"
    # размер контекста: в 8 ГБ видеопамяти 8B-модель с 8K контекста помещается
    ollama_num_ctx: int = 8192
    ollama_keep_alive: str = "30m"

    # где хранить модели Whisper (по умолчанию — кэш Hugging Face)
    whisper_download_root: str = ""

    yandex_api_key: str = ""
    yandex_folder_id: str = ""
    yandex_base_url: str = "https://llm.api.cloud.yandex.net/v1"

    # «Ключ авторизации» из кабинета GigaChat API (Base64 от Client ID и Client Secret)
    gigachat_credentials: str = ""
    gigachat_scope: str = "GIGACHAT_API_PERS"
    gigachat_base_url: str = "https://api.giga.chat/v1"
    gigachat_auth_url: str = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    # сертификаты Сбера выпущены НУЦ Минцифры («Russian Trusted Root CA»), которого
    # нет в стандартном наборе; путь к файлу сертификата — https://www.gosuslugi.ru/crt
    gigachat_ca_bundle: str = ""
    gigachat_verify_ssl: bool = True

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"

    anthropic_api_key: str = ""
    anthropic_base_url: str = "https://api.anthropic.com"

    gemini_api_key: str = ""
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"

    # Alibaba Cloud Model Studio (DashScope), международный регион
    qwen_api_key: str = ""
    qwen_base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

    # часть зарубежных API недоступна из РФ напрямую: http(s)://… или socks5://…
    ai_proxy_url: str = ""
    ai_proxy_providers: list[str] = ["openai", "anthropic", "gemini"]

    ai_timeout_sec: float = 180.0
    ai_retries: int = 2


@lru_cache
def get_provider_settings() -> ProviderSettings:
    return ProviderSettings()
