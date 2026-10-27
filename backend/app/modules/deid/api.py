"""HTTP-интерфейс сервиса деперсонализации и клиент к нему.

Сервис — граница персональных данных: исходный текст пользователя попадает только
сюда (и, если включён второй проход, в локальную или российскую модель). Сервис
не ходит в БД и ничего не сохраняет; вызывать его может только наш API — по
короткоживущему служебному токену.
"""

from datetime import timedelta
from functools import lru_cache
from typing import Annotated
from uuid import UUID

import httpx
import jwt
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.security import SERVICE, create_token, decode_token
from app.modules.ai.profiles import ModelRef
from app.modules.ai.providers.base import ProviderError
from app.modules.ai.registry import Registry
from app.modules.deid.service import DeidResult, deidentify

# идентификатор основного API как отправителя служебных токенов
API_SERVICE_ID = UUID("00000000-0000-0000-0000-00000000a001")
MAX_TEXT = 8000


class DeidSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=("../.env", ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    # адрес сервиса; пусто — деперсонализация в процессе API (для разработки)
    deid_url: str = ""
    deid_timeout_sec: float = 120.0


@lru_cache
def get_deid_settings() -> DeidSettings:
    return DeidSettings()


class DeidIn(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT)
    llm: ModelRef | None = Field(None, description="Модель для второго прохода; null — без неё")


class DeidOut(BaseModel):
    text: str
    entities: list[dict[str, str]]
    mapping: dict[str, str]
    llm: dict | None


router = APIRouter(tags=["deid"])
bearer = HTTPBearer(auto_error=False)


async def require_service(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> None:
    try:
        if credentials is None:
            raise jwt.InvalidTokenError("no token")
        decode_token(credentials.credentials, SERVICE)
    except jwt.InvalidTokenError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Нужен служебный токен") from None


def get_registry(request: Request) -> Registry:
    return request.app.state.ai_registry


@router.post("/deid", response_model=DeidOut, dependencies=[Depends(require_service)])
async def deid(data: DeidIn, registry: Annotated[Registry, Depends(get_registry)]):
    model = None
    if data.llm is not None:
        try:
            model = (registry.get(data.llm.provider), data.llm)
        except ProviderError as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    return await deidentify(data.text, model)


class DeidUnavailable(Exception):
    pass


class DeidClient:
    """Клиент API к сервису деперсонализации.

    Если сервис недоступен — ошибка, а не отправка исходного текста дальше:
    без деперсонализации вопрос в языковую модель не уходит.
    """

    def __init__(
        self,
        settings: DeidSettings,
        registry: Registry,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.registry = registry
        self.http = (
            httpx.AsyncClient(
                base_url=settings.deid_url.rstrip("/"),
                timeout=settings.deid_timeout_sec,
                transport=transport,
            )
            if settings.deid_url
            else None
        )

    async def deidentify(self, text: str, llm: ModelRef | None) -> DeidResult:
        if self.http is None:
            model = (self.registry.get(llm.provider), llm) if llm else None
            return await deidentify(text, model)
        token = create_token(API_SERVICE_ID, SERVICE, timedelta(minutes=1))
        payload = DeidIn(text=text, llm=llm).model_dump(mode="json")
        try:
            resp = await self.http.post(
                "/api/v1/deid", json=payload, headers={"Authorization": f"Bearer {token}"}
            )
        except httpx.HTTPError as exc:
            raise DeidUnavailable(f"сервис деперсонализации недоступен: {exc}") from exc
        if resp.status_code != 200:
            raise DeidUnavailable(f"сервис деперсонализации: HTTP {resp.status_code}")
        return DeidResult(**resp.json())

    async def aclose(self) -> None:
        if self.http is not None:
            await self.http.aclose()
