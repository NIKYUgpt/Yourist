from functools import lru_cache
from typing import Literal, Self

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_DEV_SECRET = "dev-only-secret-key-do-not-use-in-production"


class Settings(BaseSettings):
    # .env ищем и в корне репозитория, и рядом с backend (второй приоритетнее)
    model_config = SettingsConfigDict(
        env_file=("../.env", ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    env: Literal["dev", "test", "prod"] = "dev"
    frontend_url: str = "http://localhost:3000"
    cors_origins: list[str] = []

    # 127.0.0.1, а не localhost: на Windows localhost сначала пробует IPv6 и теряет ~2 с
    database_url: str = "postgresql+asyncpg://yourist:yourist@127.0.0.1:5432/yourist"

    secret_key: str = _DEV_SECRET
    access_token_ttl_minutes: int = 15
    refresh_token_ttl_days: int = 30
    email_token_ttl_hours: int = 24
    require_verified_email: bool = True

    mail_backend: Literal["console", "smtp"] = "console"
    mail_from: str = "Yourist <no-reply@localhost>"
    smtp_host: str = ""
    smtp_port: int = 465
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_ssl: bool = True

    @model_validator(mode="after")
    def _check_prod_secret(self) -> Self:
        if self.env == "prod" and (self.secret_key == _DEV_SECRET or len(self.secret_key) < 32):
            raise ValueError("В prod SECRET_KEY должен быть случайной строкой от 32 символов")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
