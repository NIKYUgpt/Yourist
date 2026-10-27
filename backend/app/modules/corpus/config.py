from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

# Трудовой кодекс РФ на actual.pravo.gov.ru (dochash из карточки документа)
TK_RF_HASH = "03bdee8f44a71247d7ba3342484326dbc3fa292d3caae9f88ada4b1ee38c20d9"


class CorpusSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CORPUS_",
        env_file=("../.env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # недокументированный JSON API, которым пользуется сам сайт
    source_url: str = "http://actual.pravo.gov.ru:8000/api/ebpi"
    user_agent: str = "Yourist/0.5 (+https://github.com/NIKYUgpt/Yourist)"
    request_delay_sec: float = 1.0
    request_timeout_sec: float = 120.0
    request_retries: int = 3

    sync_enabled: bool = True
    sync_interval_hours: float = 24.0
    # поисковый индекс: после синхронизации и по таймеру (подхватывает смену модели)
    index_enabled: bool = True
    index_interval_minutes: float = 10.0
    # акты, которые отслеживаются с первого запуска
    default_acts: list[str] = [TK_RF_HASH]


@lru_cache
def get_corpus_settings() -> CorpusSettings:
    return CorpusSettings()
