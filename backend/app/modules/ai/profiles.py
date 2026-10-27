"""Какая модель решает какую задачу: локальный вариант, API-вариант и переключатель.

Переключатель mode — та самая «одна настройка в админке»: local — все задачи
на локальных моделях, api — на внешних сервисах. При необходимости отдельной
задаче можно задать свой режим (TaskProfile.mode).
"""

import enum
from typing import Any

from pydantic import BaseModel, Field

from app.modules.ai.providers.base import Capability, Kind, ProviderInfo


class Mode(enum.StrEnum):
    LOCAL = "local"
    API = "api"


class Task(enum.StrEnum):
    ANSWER = "answer"  # ответ на вопрос пользователя
    VERIFY = "verify"  # проверка ответа второй моделью
    DEID = "deid"  # поиск персональных данных, пропущенных правилами
    EMBED = "embed"  # векторы для поиска по нормам


TASK_TITLES = {
    Task.ANSWER: "Ответ на вопрос",
    Task.VERIFY: "Проверка ответа",
    Task.DEID: "Деперсонализация",
    Task.EMBED: "Векторный поиск",
}


def required_capability(task: Task) -> Capability:
    return Capability.EMBED if task == Task.EMBED else Capability.CHAT


class ModelRef(BaseModel):
    provider: str
    model: str = Field(min_length=1, max_length=200)
    # параметры запроса сверх стандартных; null убирает параметр (см. merge_options)
    options: dict[str, Any] = {}


class TaskProfile(BaseModel):
    local: ModelRef
    api: ModelRef
    mode: Mode | None = Field(None, description="Свой режим задачи; null — общий переключатель")


class AIConfig(BaseModel):
    mode: Mode = Mode.LOCAL
    tasks: dict[Task, TaskProfile]
    # второй проход деперсонализации моделью (после правил и словаря)
    deid_llm: bool = True
    # проверка ответа второй моделью (Task.VERIFY)
    verify: bool = True

    def resolve(self, task: Task) -> tuple[Mode, ModelRef]:
        profile = self.tasks[task]
        mode = profile.mode or self.mode
        return mode, profile.local if mode == Mode.LOCAL else profile.api


# Имена моделей у API-провайдеров меняются; актуальный список — в админке
# (GET /admin/ai/providers/{name}/models)
DEFAULT_CONFIG = AIConfig(
    mode=Mode.LOCAL,
    tasks={
        Task.ANSWER: TaskProfile(
            local=ModelRef(provider="ollama", model="qwen3:8b", options={"think": False}),
            api=ModelRef(provider="gigachat", model="GigaChat-2-Max"),
        ),
        # проверяющая модель — другого разработчика, чтобы ошибки не совпадали
        Task.VERIFY: TaskProfile(
            local=ModelRef(provider="ollama", model="qwen3:8b", options={"think": False}),
            api=ModelRef(provider="yandex", model="yandexgpt"),
        ),
        Task.DEID: TaskProfile(
            local=ModelRef(provider="ollama", model="qwen3:8b", options={"think": False}),
            api=ModelRef(provider="yandex", model="yandexgpt"),
        ),
        Task.EMBED: TaskProfile(
            # векторы считаем на процессоре (запрос ~0,2 с): иначе на 8 ГБ видеопамяти
            # модель ответа и модель векторов вытесняют друг друга на каждом вопросе
            local=ModelRef(provider="ollama", model="bge-m3", options={"num_gpu": 0}),
            api=ModelRef(provider="yandex", model="text-search"),
        ),
    },
)


def with_defaults(data: dict[str, Any] | None) -> AIConfig:
    """Сохранённая конфигурация, дополненная задачами, которых в ней ещё нет."""
    merged = DEFAULT_CONFIG.model_dump(mode="json")
    if data:
        for key in ("mode", "deid_llm", "verify"):
            merged[key] = data.get(key, merged[key])
        merged["tasks"].update(data.get("tasks", {}))
    return AIConfig.model_validate(merged)


def validate_config(config: AIConfig, providers: dict[str, ProviderInfo]) -> list[str]:
    errors = []
    for task in Task:
        profile = config.tasks.get(task)
        if profile is None:
            errors.append(f"{task}: задача не настроена")
            continue
        for slot, ref in (("local", profile.local), ("api", profile.api)):
            where = f"{task}.{slot}"
            info = providers.get(ref.provider)
            if info is None:
                errors.append(f"{where}: неизвестный провайдер {ref.provider!r}")
                continue
            expected = Kind.LOCAL if slot == "local" else Kind.API
            if info.kind != expected:
                errors.append(f"{where}: провайдер {ref.provider!r} не {expected.value}")
            if required_capability(task) not in info.capabilities:
                errors.append(f"{where}: {ref.provider!r} не умеет {required_capability(task)}")
            # в задачу деперсонализации попадает исходный текст пользователя —
            # за пределы РФ его не отправляем
            if task == Task.DEID and info.location == "foreign":
                errors.append(f"{where}: для деперсонализации нужен локальный или российский")
    return errors


def embedding_key(ref: ModelRef) -> str:
    """Имя пространства векторов: векторы разных моделей несравнимы."""
    return f"{ref.provider}/{ref.model}"
