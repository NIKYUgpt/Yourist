"""Деперсонализация текста: правила → словарь имён → (по настройке) языковая модель.

Найденное заменяется метками [ФИО_1], [ТЕЛЕФОН_1] … Одно и то же значение получает
одну метку во всех местах текста — в том числе ФИО в разных падежах («Иванов»,
«Иванову»), чтобы модель понимала, что речь об одном человеке. Таблица
«метка → значение» возвращается вызывающему и нигде не сохраняется.
"""

import re
from dataclasses import dataclass, field

from app.modules.ai.profiles import ModelRef
from app.modules.ai.providers.base import Provider
from app.modules.deid import llm, names, rules
from app.modules.deid.entities import PERSON, Entity

_PLACEHOLDER = re.compile(r"\[([А-ЯЁA-Z_]+)_(\d+)\]")


@dataclass
class DeidResult:
    text: str
    # без исходных значений: тип, метка, кто нашёл
    entities: list[dict[str, str]] = field(default_factory=list)
    mapping: dict[str, str] = field(default_factory=dict)
    llm: dict | None = None


def _select(found: list[Entity], taken: list[tuple[int, int]] | None = None) -> list[Entity]:
    """Непересекающиеся находки; при пересечении побеждает более длинная."""
    chosen: list[Entity] = []
    spans = list(taken or [])
    for ent in sorted(found, key=lambda e: (-(e.end - e.start), e.start)):
        if all(ent.end <= s or ent.start >= e for s, e in spans):
            chosen.append(ent)
            spans.append((ent.start, ent.end))
    return sorted(chosen, key=lambda e: e.start)


def _key(kind: str, value: str) -> str:
    if kind == PERSON:
        # ФИО в разных падежах — к начальной форме: «Иванову Ивану» → «иванов иван»
        morph = names.morph()
        words = re.findall(r"[А-ЯЁа-яё]+", value)
        return " ".join(morph.parse(w)[0].normal_form for w in words) or value.lower()
    return re.sub(r"[\s\-()]", "", value.lower())


class _Masker:
    def __init__(self) -> None:
        self.mapping: dict[str, str] = {}
        self.by_key: dict[tuple[str, str], str] = {}
        self.counters: dict[str, int] = {}
        self.entities: list[dict[str, str]] = []

    def placeholder(self, kind: str, value: str) -> str:
        key = (kind, _key(kind, value))
        if key not in self.by_key:
            self.counters[kind] = self.counters.get(kind, 0) + 1
            label = f"[{kind}_{self.counters[kind]}]"
            self.by_key[key] = label
            self.mapping[label] = value
        return self.by_key[key]

    def apply(self, text: str, found: list[Entity]) -> str:
        parts, pos = [], 0
        for ent in found:
            label = self.placeholder(ent.type, text[ent.start : ent.end])
            parts += [text[pos : ent.start], label]
            pos = ent.end
            self.entities.append({"type": ent.type, "placeholder": label, "source": ent.source})
        parts.append(text[pos:])
        return "".join(parts)


async def deidentify(text: str, model: tuple[Provider, ModelRef] | None = None) -> DeidResult:
    """model — языковая модель для второго прохода; None — только правила и словарь."""
    masker = _Masker()
    masked = masker.apply(text, _select([*rules.find(text), *names.find(text)]))
    llm_info = None
    if model is not None:
        extra, llm_info = await llm.find(masked, *model)
        # метки, поставленные на первом проходе, модель не трогает
        taken = [(m.start(), m.end()) for m in _PLACEHOLDER.finditer(masked)]
        masked = masker.apply(masked, _select(extra, taken))
    return DeidResult(text=masked, entities=masker.entities, mapping=masker.mapping, llm=llm_info)


def reidentify(text: str, mapping: dict[str, str]) -> str:
    """Вернуть исходные значения в ответ — только для показа пользователю."""
    return _PLACEHOLDER.sub(lambda m: mapping.get(m.group(0), m.group(0)), text)


def summary(entities: list[dict[str, str]]) -> dict[str, int]:
    """Сколько найдено каждого вида — это можно хранить, значения — нет."""
    counts: dict[str, int] = {}
    for ent in entities:
        counts[ent["type"]] = counts.get(ent["type"], 0) + 1
    return counts
