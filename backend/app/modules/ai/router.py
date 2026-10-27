"""Админка нейросетей: переключатель local/api, модели по задачам, проверка провайдеров."""

from fastapi import APIRouter, Depends, HTTPException, status

from app.db.session import SessionDep
from app.modules.ai.profiles import TASK_TITLES, AIConfig, Task, validate_config
from app.modules.ai.providers.base import Capability, EmbedKind, ProviderError, Timer
from app.modules.ai.registry import Registry
from app.modules.ai.schemas import (
    CheckIn,
    CheckOut,
    ConfigOut,
    EffectiveOut,
    ModeIn,
    ProviderOut,
)
from app.modules.ai.service import RegistryDep, load_config, save_config
from app.modules.auth.deps import require_roles
from app.modules.auth.models import Role, User

router = APIRouter(prefix="/admin/ai", tags=["admin: ai"])
AdminDep = Depends(require_roles(Role.ADMIN))


def _config_out(config: AIConfig, registry: Registry) -> ConfigOut:
    effective = {}
    for task in Task:
        mode, ref = config.resolve(task)
        provider = registry.providers.get(ref.provider)
        effective[task] = EffectiveOut(
            title=TASK_TITLES[task],
            mode=mode,
            provider=ref.provider,
            model=ref.model,
            configured=provider is not None and provider.configured(),
        )
    return ConfigOut(config=config, effective=effective)


def _validated(config: AIConfig, registry: Registry) -> AIConfig:
    errors = validate_config(config, {i.name: i for i in registry.infos()})
    if errors:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, errors)
    return config


@router.get("/providers", response_model=list[ProviderOut], dependencies=[AdminDep])
async def providers(registry: RegistryDep):
    return [
        ProviderOut(
            name=p.info.name,
            title=p.info.title,
            kind=p.info.kind,
            capabilities=sorted(p.info.capabilities),
            location=p.info.location,
            configured=p.configured(),
        )
        for p in registry.providers.values()
    ]


@router.get("/providers/{name}/models", response_model=list[str], dependencies=[AdminDep])
async def provider_models(name: str, registry: RegistryDep):
    try:
        provider = registry.get(name)
        if not provider.configured():
            raise ProviderError(name, "не заданы ключи доступа")
        return await provider.models()
    except ProviderError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc


@router.post("/providers/{name}/check", response_model=CheckOut, dependencies=[AdminDep])
async def check_provider(name: str, data: CheckIn, registry: RegistryDep):
    """Пробный запрос к модели: работают ли ключи, сеть и имя модели."""
    try:
        provider = registry.get(name)
    except ProviderError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    timer = Timer()
    try:
        if data.capability == Capability.EMBED:
            vectors = await provider.embed(data.model, ["проверка связи"], kind=EmbedKind.QUERY)
            detail = f"вектор размерности {len(vectors[0])}"
        else:
            result = await provider.chat(
                data.model,
                [{"role": "user", "content": "Ответь одним словом: столица России?"}],
                max_tokens=20,
            )
            detail = result.text.strip()[:200]
    except ProviderError as exc:
        return CheckOut(ok=False, latency_ms=timer.ms, detail=str(exc))
    return CheckOut(ok=True, latency_ms=timer.ms, detail=detail)


@router.get("/config", response_model=ConfigOut, dependencies=[AdminDep])
async def get_config(session: SessionDep, registry: RegistryDep):
    return _config_out(await load_config(session), registry)


@router.put("/config", response_model=ConfigOut)
async def put_config(
    data: AIConfig,
    session: SessionDep,
    registry: RegistryDep,
    admin: User = AdminDep,
):
    await save_config(session, _validated(data, registry), admin.id)
    return _config_out(data, registry)


@router.put("/mode", response_model=ConfigOut)
async def put_mode(
    data: ModeIn, session: SessionDep, registry: RegistryDep, admin: User = AdminDep
):
    """Общий переключатель: все задачи на локальных моделях или на API."""
    config = await load_config(session)
    config.mode = data.mode
    await save_config(session, _validated(config, registry), admin.id)
    return _config_out(config, registry)
