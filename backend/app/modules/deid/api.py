"""HTTP-интерфейс сервиса деперсонализации и клиент к нему.

Сервис — граница персональных данных: исходный текст, голос и фото документа
пользователя попадают только сюда (и, если включены, в локальные или российские
модели распознавания и второго прохода). Сервис не ходит в БД и ничего не сохраняет;
вызывать его может только наш API — по короткоживущему служебному токену.
"""

from dataclasses import dataclass
from datetime import timedelta
from functools import lru_cache
from typing import Annotated
from uuid import UUID

import httpx
import jwt
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.security import SERVICE, create_token, decode_token
from app.modules.ai.profiles import ModelRef
from app.modules.ai.providers.base import ProviderError
from app.modules.ai.registry import Registry
from app.modules.deid.media import MediaRejected, from_audio, from_image
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


class MediaOut(DeidOut):
    recognized: dict = Field(description="Чем распознано, сколько символов, обрезан ли текст")


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


def _refs(main: str, llm: str | None) -> tuple[ModelRef, ModelRef | None]:
    try:
        return ModelRef.model_validate_json(main), (
            ModelRef.model_validate_json(llm) if llm else None
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc


async def _media(call) -> MediaOut:
    try:
        deid, recognized = await call
    except MediaRejected as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    except ProviderError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc
    return MediaOut(**deid.__dict__, recognized=recognized.dump())


@router.post("/deid/audio", response_model=MediaOut, dependencies=[Depends(require_service)])
async def deid_audio(
    registry: Annotated[Registry, Depends(get_registry)],
    file: UploadFile = File(...),
    stt: str = Form(..., description="ModelRef распознавания речи (JSON)"),
    llm: str | None = Form(None),
):
    stt_ref, llm_ref = _refs(stt, llm)
    return await _media(from_audio(registry, await file.read(), stt_ref, llm_ref))


@router.post("/deid/image", response_model=MediaOut, dependencies=[Depends(require_service)])
async def deid_image(
    registry: Annotated[Registry, Depends(get_registry)],
    file: UploadFile = File(...),
    ocr: str = Form(..., description="ModelRef распознавания документа (JSON)"),
    llm: str | None = Form(None),
    text: str = Form("", max_length=MAX_TEXT, description="Вопрос пользователя к документу"),
):
    ocr_ref, llm_ref = _refs(ocr, llm)
    image = await file.read()
    return await _media(
        from_image(registry, image, file.content_type or "", text, ocr_ref, llm_ref)
    )


class DeidUnavailable(Exception):
    pass


class MediaError(Exception):
    """Распознать не удалось по вине входных данных (формат, пустая запись)."""


@dataclass
class MediaResult:
    deid: DeidResult
    recognized: dict


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

    async def from_audio(
        self, audio: bytes, filename: str, stt: ModelRef, llm: ModelRef | None
    ) -> MediaResult:
        if self.http is None:
            return await self._local(from_audio(self.registry, audio, stt, llm))
        data = {"stt": stt.model_dump_json(), **({"llm": llm.model_dump_json()} if llm else {})}
        return await self._post_media("/api/v1/deid/audio", data, (filename, audio, None))

    async def from_image(
        self,
        image: bytes,
        filename: str,
        mime: str,
        question: str,
        ocr: ModelRef,
        llm: ModelRef | None,
    ) -> MediaResult:
        if self.http is None:
            return await self._local(from_image(self.registry, image, mime, question, ocr, llm))
        data = {
            "ocr": ocr.model_dump_json(),
            "text": question,
            **({"llm": llm.model_dump_json()} if llm else {}),
        }
        return await self._post_media("/api/v1/deid/image", data, (filename, image, mime))

    async def _local(self, call) -> MediaResult:
        try:
            deid, recognized = await call
        except MediaRejected as exc:
            raise MediaError(str(exc)) from exc
        except ProviderError as exc:
            raise DeidUnavailable(f"распознавание недоступно: {exc}") from exc
        return MediaResult(deid, recognized.dump())

    async def _post_media(self, path: str, data: dict, file: tuple) -> MediaResult:
        token = create_token(API_SERVICE_ID, SERVICE, timedelta(minutes=1))
        try:
            resp = await self.http.post(
                path,
                data=data,
                files={"file": file},
                headers={"Authorization": f"Bearer {token}"},
            )
        except httpx.HTTPError as exc:
            raise DeidUnavailable(f"сервис деперсонализации недоступен: {exc}") from exc
        if resp.status_code == 422:
            raise MediaError(str(resp.json().get("detail", "файл не распознан")))
        if resp.status_code != 200:
            raise DeidUnavailable(f"распознавание недоступно: HTTP {resp.status_code}")
        body = resp.json()
        recognized = body.pop("recognized")
        return MediaResult(DeidResult(**body), recognized)

    async def aclose(self) -> None:
        if self.http is not None:
            await self.http.aclose()
