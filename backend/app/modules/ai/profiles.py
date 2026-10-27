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
    STT = "stt"  # распознавание голосового вопроса
    OCR = "ocr"  # распознавание текста на фото документа


TASK_TITLES = {
    Task.ANSWER: "Ответ на вопрос",
    Task.VERIFY: "Проверка ответа",
    Task.DEID: "Деперсонализация",
    Task.EMBED: "Векторный поиск",
    Task.STT: "Распознавание речи",
    Task.OCR: "Распознавание документов",
}

# задачи, в которые попадают исходные данные пользователя (текст, голос, фото документа):
# их не отправляем зарубежным провайдерам
RAW_INPUT_TASKS = (Task.DEID, Task.STT, Task.OCR)


def required_capability(task: Task) -> Capability:
    special = {Task.EMBED: Capability.EMBED, Task.STT: Capability.STT, Task.OCR: Capability.OCR}
    return special.get(task, Capability.CHAT)


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
        # выбор по сравнению 27–28.10.2026 (backend/evaluation/results): YandexGPT-5 Lite —
        # лучшая точность и скорость среди локальных моделей; в Ollama — с исправленным
        # шаблоном диалога (deploy/ollama/yandexgpt-5-lite.Modelfile)
        Task.ANSWER: TaskProfile(
            local=ModelRef(provider="ollama", model="yandexgpt-5-lite:8b"),
            api=ModelRef(provider="yandex", model="yandexgpt-5.1"),
        ),
        # проверяющая модель — другого разработчика, чтобы ошибки не совпадали: перекрёстная
        # проверка ловила больше подменённых фактов, чем самопроверка (93% против 79%)
        Task.VERIFY: TaskProfile(
            local=ModelRef(provider="ollama", model="qwen3:8b", options={"think": False}),
            api=ModelRef(provider="yandex", model="deepseek-v4-flash"),
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
        Task.STT: TaskProfile(
            # large-v3-turbo в int8 на процессоре: точный русский без видеокарты
            local=ModelRef(
                provider="whisper",
                model="large-v3-turbo",
                options={"device": "cpu", "compute_type": "int8"},
            ),
            api=ModelRef(provider="yandex", model="general"),
        ),
        Task.OCR: TaskProfile(
            local=ModelRef(provider="ollama", model="qwen2.5vl:7b"),
            api=ModelRef(provider="yandex", model="page"),
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
            # в эти задачи попадают исходные данные пользователя — за пределы РФ не отправляем
            if task in RAW_INPUT_TASKS and info.location == "foreign":
                errors.append(f"{where}: для исходных данных нужен локальный или российский")
    return errors


def embedding_key(ref: ModelRef) -> str:
    """Имя пространства векторов: векторы разных моделей несравнимы."""
    return f"{ref.provider}/{ref.model}"
