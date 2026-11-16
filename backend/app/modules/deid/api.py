"""HTTP-интерфейс сервиса деперсонализации и клиент к нему.

Сервис — граница персональных данных: исходный текст, голос и фото документа
пользователя попадают только сюда (и, если включены, в локальные или российские
модели распознавания и второго прохода). Сервис не ходит в БД и ничего не сохраняет;
вызывать его может только наш API — по короткоживущему служебному токену.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import timedelta
from functools import lru_cache
from typing import Annotated
from uuid import UUID

import httpx
import jwt
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.security import SERVICE, create_token, decode_token
from app.modules.ai.profiles import ModelRef
from app.modules.ai.providers.base import ProviderError
from app.modules.ai.registry import Registry
from app.modules.deid.identity import IdentityBindingError, IdentityRequired, PersonResolution
from app.modules.deid.media import MediaRejected, from_audio, from_image
from app.modules.deid.service import DeidResult, deidentify
from app.modules.deid.sessions import (
    SessionBusy,
    SessionExpired,
    SessionInfo,
    SessionLimit,
    SessionStore,
)

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
    deid_session_ttl_seconds: int = Field(86400, ge=60, le=604800)
    deid_max_sessions: int = Field(256, ge=1, le=10000)


def session_store(settings: DeidSettings) -> SessionStore:
    return SessionStore(settings.deid_session_ttl_seconds, settings.deid_max_sessions)


@lru_cache
def get_deid_settings() -> DeidSettings:
    return DeidSettings()


class DeidIn(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT)
    llm: ModelRef | None = Field(None, description="Модель для второго прохода; null — без неё")
    session_id: UUID | None = None
    user_id: UUID | None = None
    person_resolution: PersonResolution | None = None


class SessionIn(BaseModel):
    user_id: UUID


class DeidOut(BaseModel):
    text: str
    entities: list[dict[str, str]]
    mapping: dict[str, str]
    llm: dict | None


class MediaOut(DeidOut):
    recognized: dict = Field(description="Чем распознано, сколько символов, обрезан ли текст")


router = APIRouter(tags=["deid"])
bearer = HTTPBearer(auto_error=False)


IDENTITY_ERROR = {
    "code": "invalid_person_resolution",
    "message": "Некорректная привязка участников",
}


def parse_person_resolution(value: str | None) -> PersonResolution | None:
    if value is None:
        return None
    try:
        return PersonResolution.model_validate_json(value)
    except ValueError:
        raise HTTPException(422, IDENTITY_ERROR) from None


def install_identity_handlers(app) -> None:
    async def required(request, exc):
        return JSONResponse(status_code=409, content={"detail": exc.detail})

    async def invalid(request, exc):
        return JSONResponse(status_code=422, content={"detail": IDENTITY_ERROR})

    async def validation(request, exc):
        if any("person_resolution" in error["loc"] for error in exc.errors()):
            return await invalid(request, exc)
        return await request_validation_exception_handler(request, exc)

    app.add_exception_handler(IdentityRequired, required)
    app.add_exception_handler(IdentityBindingError, invalid)
    app.add_exception_handler(RequestValidationError, validation)


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


def get_sessions(request: Request) -> SessionStore:
    return request.app.state.deid_sessions


@asynccontextmanager
async def session_mapping(
    store: SessionStore, session_id: UUID | None, user_id: UUID | None
) -> AsyncIterator[dict[str, str] | None]:
    if session_id is None and user_id is None:
        yield None
        return
    if session_id is None or user_id is None:
        raise HTTPException(422, "Нужны и session_id, и user_id")
    try:
        async with store.use(session_id, user_id) as mapping:
            yield mapping
    except (SessionExpired, SessionBusy) as exc:
        raise HTTPException(409, str(exc)) from exc
    except SessionLimit as exc:
        raise HTTPException(429, str(exc)) from exc


@router.post("/deid/sessions", response_model=SessionInfo, dependencies=[Depends(require_service)])
async def create_session(data: SessionIn, store: Annotated[SessionStore, Depends(get_sessions)]):
    try:
        return store.create(data.user_id)
    except SessionLimit as exc:
        raise HTTPException(429, str(exc)) from exc


@router.delete(
    "/deid/sessions/{session_id}", status_code=204, dependencies=[Depends(require_service)]
)
async def delete_session(
    session_id: UUID, user_id: UUID, store: Annotated[SessionStore, Depends(get_sessions)]
):
    store.remove(session_id, user_id)


@router.post("/deid", response_model=DeidOut, dependencies=[Depends(require_service)])
async def deid(
    data: DeidIn,
    registry: Annotated[Registry, Depends(get_registry)],
    store: Annotated[SessionStore, Depends(get_sessions)],
):
    model = None
    if data.llm is not None:
        try:
            model = (registry.get(data.llm.provider), data.llm)
        except ProviderError as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    async with session_mapping(store, data.session_id, data.user_id) as mapping:
        result = await deidentify(
            data.text, model, mapping=mapping, person_resolution=data.person_resolution
        )
    return result


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
    store: Annotated[SessionStore, Depends(get_sessions)],
    file: UploadFile = File(...),
    stt: str = Form(..., description="ModelRef распознавания речи (JSON)"),
    llm: str | None = Form(None),
    session_id: UUID | None = Form(None),
    user_id: UUID | None = Form(None),
    person_resolution: str | None = Form(None),
):
    stt_ref, llm_ref = _refs(stt, llm)
    resolution = parse_person_resolution(person_resolution)
    async with session_mapping(store, session_id, user_id) as mapping:
        result = await _media(
            from_audio(
                registry,
                await file.read(),
                stt_ref,
                llm_ref,
                mapping=mapping,
                person_resolution=resolution,
            )
        )
    return result


@router.post("/deid/image", response_model=MediaOut, dependencies=[Depends(require_service)])
async def deid_image(
    registry: Annotated[Registry, Depends(get_registry)],
    store: Annotated[SessionStore, Depends(get_sessions)],
    file: UploadFile = File(...),
    ocr: str = Form(..., description="ModelRef распознавания документа (JSON)"),
    llm: str | None = Form(None),
    text: str = Form("", max_length=MAX_TEXT, description="Вопрос пользователя к документу"),
    session_id: UUID | None = Form(None),
    user_id: UUID | None = Form(None),
    person_resolution: str | None = Form(None),
):
    ocr_ref, llm_ref = _refs(ocr, llm)
    resolution = parse_person_resolution(person_resolution)
    image = await file.read()
    async with session_mapping(store, session_id, user_id) as mapping:
        result = await _media(
            from_image(
                registry,
                image,
                file.content_type or "",
                text,
                ocr_ref,
                llm_ref,
                mapping=mapping,
                person_resolution=resolution,
            )
        )
    return result


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
        self.sessions = session_store(settings)
        self.http = (
            httpx.AsyncClient(
                base_url=settings.deid_url.rstrip("/"),
                timeout=settings.deid_timeout_sec,
                transport=transport,
            )
            if settings.deid_url
            else None
        )

    async def create_session(self, user_id: UUID) -> SessionInfo:
        if self.http is None:
            self.sessions.start()
            return self.sessions.create(user_id)
        resp = await self._session_request(
            "POST", "/api/v1/deid/sessions", json={"user_id": str(user_id)}
        )
        from pydantic import TypeAdapter

        return TypeAdapter(SessionInfo).validate_python(resp.json())

    async def delete_session(self, session_id: UUID, user_id: UUID) -> None:
        if self.http is None:
            self.sessions.remove(session_id, user_id)
            return
        await self._session_request(
            "DELETE", f"/api/v1/deid/sessions/{session_id}", params={"user_id": str(user_id)}
        )

    async def _session_request(self, method: str, path: str, **kwargs) -> httpx.Response:
        token = create_token(API_SERVICE_ID, SERVICE, timedelta(minutes=1))
        try:
            resp = await self.http.request(
                method, path, headers={"Authorization": f"Bearer {token}"}, **kwargs
            )
        except httpx.HTTPError as exc:
            raise DeidUnavailable("Сервис обезличивания недоступен") from exc
        self._session_error(resp)
        if resp.status_code not in (200, 204):
            raise DeidUnavailable(f"сервис деперсонализации: HTTP {resp.status_code}")
        return resp

    @staticmethod
    def _session_error(resp: httpx.Response) -> None:
        if resp.status_code in (409, 422):
            try:
                detail = resp.json().get("detail")
            except (ValueError, AttributeError):
                detail = None
            if isinstance(detail, dict):
                if resp.status_code == 409 and detail.get("code") == "person_identity_required":
                    raise IdentityRequired(detail)
                if resp.status_code == 422 and detail.get("code") == "invalid_person_resolution":
                    raise IdentityBindingError("Некорректная привязка участников")
        if resp.status_code == 409:
            raise SessionExpired("Сеанс обезличивания недоступен; начните новый диалог")
        if resp.status_code == 429:
            raise SessionLimit("Достигнут предел данных или сеансов обезличивания")

    async def deidentify(
        self,
        text: str,
        llm: ModelRef | None,
        *,
        session_id: UUID | None = None,
        user_id: UUID | None = None,
        person_resolution: PersonResolution | None = None,
    ) -> DeidResult:
        resolution = (
            {"person_resolution": person_resolution} if person_resolution is not None else {}
        )
        if self.http is None:
            model = (self.registry.get(llm.provider), llm) if llm else None
            if session_id is not None:
                async with self.sessions.use(session_id, user_id) as mapping:
                    result = await deidentify(text, model, mapping=mapping, **resolution)
                return result
            return await deidentify(text, model, **resolution)
        token = create_token(API_SERVICE_ID, SERVICE, timedelta(minutes=1))
        payload = DeidIn(
            text=text,
            llm=llm,
            session_id=session_id,
            user_id=user_id,
            person_resolution=person_resolution,
        ).model_dump(mode="json")
        try:
            resp = await self.http.post(
                "/api/v1/deid", json=payload, headers={"Authorization": f"Bearer {token}"}
            )
        except httpx.HTTPError as exc:
            raise DeidUnavailable(f"сервис деперсонализации недоступен: {exc}") from exc
        self._session_error(resp)
        if resp.status_code != 200:
            raise DeidUnavailable(f"сервис деперсонализации: HTTP {resp.status_code}")
        return DeidResult(**resp.json())

    async def from_audio(
        self,
        audio: bytes,
        filename: str,
        stt: ModelRef,
        llm: ModelRef | None,
        *,
        session_id: UUID | None = None,
        user_id: UUID | None = None,
        person_resolution: PersonResolution | None = None,
    ) -> MediaResult:
        resolution = (
            {"person_resolution": person_resolution} if person_resolution is not None else {}
        )
        if self.http is None:
            if session_id is not None:
                async with self.sessions.use(session_id, user_id) as mapping:
                    result = await self._local(
                        from_audio(self.registry, audio, stt, llm, mapping=mapping, **resolution)
                    )
                return result
            return await self._local(from_audio(self.registry, audio, stt, llm, **resolution))
        data = {"stt": stt.model_dump_json(), **({"llm": llm.model_dump_json()} if llm else {})}
        if person_resolution is not None:
            data["person_resolution"] = person_resolution.model_dump_json()
        if session_id is not None:
            data.update(session_id=str(session_id), user_id=str(user_id))
        return await self._post_media("/api/v1/deid/audio", data, (filename, audio, None))

    async def from_image(
        self,
        image: bytes,
        filename: str,
        mime: str,
        question: str,
        ocr: ModelRef,
        llm: ModelRef | None,
        *,
        session_id: UUID | None = None,
        user_id: UUID | None = None,
        person_resolution: PersonResolution | None = None,
    ) -> MediaResult:
        resolution = (
            {"person_resolution": person_resolution} if person_resolution is not None else {}
        )
        if self.http is None:
            if session_id is not None:
                async with self.sessions.use(session_id, user_id) as mapping:
                    result = await self._local(
                        from_image(
                            self.registry,
                            image,
                            mime,
                            question,
                            ocr,
                            llm,
                            mapping=mapping,
                            **resolution,
                        )
                    )
                return result
            return await self._local(
                from_image(
                    self.registry,
                    image,
                    mime,
                    question,
                    ocr,
                    llm,
                    **resolution,
                )
            )
        data = {
            "ocr": ocr.model_dump_json(),
            "text": question,
            **({"llm": llm.model_dump_json()} if llm else {}),
        }
        if person_resolution is not None:
            data["person_resolution"] = person_resolution.model_dump_json()
        if session_id is not None:
            data.update(session_id=str(session_id), user_id=str(user_id))
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
        self._session_error(resp)
        if resp.status_code == 422:
            raise MediaError(str(resp.json().get("detail", "файл не распознан")))
        if resp.status_code != 200:
            raise DeidUnavailable(f"распознавание недоступно: HTTP {resp.status_code}")
        body = resp.json()
        recognized = body.pop("recognized")
        return MediaResult(DeidResult(**body), recognized)

    async def aclose(self) -> None:
        await self.sessions.aclose()
        if self.http is not None:
            await self.http.aclose()
