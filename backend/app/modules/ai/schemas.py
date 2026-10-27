from pydantic import BaseModel, Field

from app.modules.ai.profiles import AIConfig, Mode, Task
from app.modules.ai.providers.base import Capability, Kind


class ProviderOut(BaseModel):
    name: str
    title: str
    kind: Kind
    capabilities: list[Capability]
    location: str = Field(description="local — у нас, ru — в РФ, foreign — за рубежом")
    configured: bool = Field(description="Заданы ли ключи в окружении")


class EffectiveOut(BaseModel):
    title: str
    mode: Mode
    provider: str
    model: str
    configured: bool


class ConfigOut(BaseModel):
    config: AIConfig
    effective: dict[Task, EffectiveOut] = Field(description="Что реально используется сейчас")


class ModeIn(BaseModel):
    mode: Mode


class CheckIn(BaseModel):
    model: str = Field(min_length=1, max_length=200)
    capability: Capability = Capability.CHAT


class CheckOut(BaseModel):
    ok: bool
    latency_ms: int | None = None
    detail: str
