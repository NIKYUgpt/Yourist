"""Второй проход языковой моделью: то, что пропустили правила и словарь.

Модель получает текст, где найденное уже заменено метками, и возвращает список
подстрок. Модель может только добавить маскирование: подстроки, которых нет
в тексте дословно, и общие слова («работодатель», «директор») отбрасываются.
"""

import json
import re

from app.modules.ai.profiles import ModelRef
from app.modules.ai.providers.base import Provider, ProviderError
from app.modules.deid.entities import LLM, TYPES, Entity

SYSTEM = (
    """Ты — модуль деперсонализации. Найди в тексте персональные данные людей,
которые ещё не заменены метками вида [ФИО_1]: фамилии, имена, отчества (в том числе
в косвенных падежах и уменьшительные), адреса, телефоны, email, номера документов,
даты рождения, названия конкретных организаций, ссылки на профили.
Не считай персональными данными: должности (директор, бухгалтер), общие слова
(работодатель, работник, сотрудник), названия законов и органов власти, города без адреса,
даты событий (увольнения, приёма на работу), суммы денег.
Ответ — только JSON: {"entities": [{"text": "точная подстрока из текста", "type": "ТИП"}]}
Типы: """
    + ", ".join(TYPES)
    + """.
Если ничего не найдено: {"entities": []}"""
)

_GENERIC = frozenset(
    (
        "работодатель",
        "работник",
        "сотрудник",
        "директор",
        "руководитель",
        "начальник",
        "бухгалтер",
        "кадровик",
        "организация",
        "компания",
        "фирма",
        "суд",
        "инспекция",
        "прокуратура",
        "россия",
        "рф",
        "москва",
    )
)
_PLACEHOLDER = re.compile(r"\[[А-ЯЁA-Z_]+_\d+\]")


def _parse(raw: str) -> list[dict]:
    # модель иногда оборачивает JSON в ```json … ``` или добавляет пояснения
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    items = data.get("entities") if isinstance(data, dict) else None
    return [i for i in items or [] if isinstance(i, dict)]


async def find(masked_text: str, provider: Provider, ref: ModelRef) -> tuple[list[Entity], dict]:
    """Возвращает находки и сведения о вызове модели (для журнала)."""
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": masked_text},
    ]
    try:
        result = await provider.chat(
            ref.model,
            messages,
            temperature=0.0,
            max_tokens=512,
            json_mode=True,
            options=ref.options,
        )
    except ProviderError as exc:
        return [], {"provider": ref.provider, "model": ref.model, "error": str(exc)}

    found: list[Entity] = []
    for item in _parse(result.text):
        value = str(item.get("text", "")).strip()
        kind = str(item.get("type", "")).strip().upper()
        if kind not in TYPES or len(value) < 3 or value.lower() in _GENERIC:
            continue
        if _PLACEHOLDER.search(value):
            continue
        for m in re.finditer(re.escape(value), masked_text):
            found.append(Entity(m.start(), m.end(), kind, LLM))
    info = {
        "provider": result.provider,
        "model": result.model,
        "latency_ms": result.latency_ms,
        "found": len(found),
    }
    return found, info
