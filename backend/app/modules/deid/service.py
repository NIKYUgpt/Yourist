"""Обезличивание без неявного объединения людей по совпадению ФИО.

Повторное имя требует явной привязки участника до ответа. Подтверждённые связи
хранятся в RAM сеанса; наружу модели получают только обезличенные метки.
"""

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass, field

from app.modules.ai.profiles import ModelRef
from app.modules.ai.providers.base import Provider
from app.modules.deid import llm, names, rules
from app.modules.deid.entities import PERSON, Entity
from app.modules.deid.identity import (
    IdentityBindingError,
    IdentityRequired,
    PersonMapping,
    PersonResolution,
)

_PLACEHOLDER = re.compile(r"\[([А-ЯЁA-Z_]+)_(\d+)\]")


@dataclass
class DeidResult:
    text: str
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
        morph = names.morph()
        words = re.findall(r"[А-ЯЁа-яё]+", value)
        return " ".join(morph.parse(w)[0].normal_form for w in words) or value.lower()
    return re.sub(r"[\s\-()]", "", value.lower())


class _Masker:
    def __init__(self, mapping: dict[str, str] | None = None) -> None:
        self.mapping = dict(mapping or {})
        self.initial_labels = set(self.mapping)
        self.actors = dict(getattr(mapping, "actors", {}))
        self.by_key: dict[tuple[str, str], set[str]] = defaultdict(set)
        self.counters: dict[str, int] = {}
        self.entities: list[dict[str, str]] = []
        self.mentions: dict[str, list[dict]] = defaultdict(list)
        self.unbound: set[tuple[int, int]] = set()
        for label, value in self.mapping.items():
            match = _PLACEHOLDER.fullmatch(label)
            if match:
                kind, number = match.groups()
                self.by_key[kind, _key(kind, value)].add(label)
                self.counters[kind] = max(self.counters.get(kind, 0), int(number))
        for alias, labels in getattr(mapping, "aliases", {}).items():
            self.by_key[PERSON, alias].update(labels)
        self.previous = {key: set(labels) for key, labels in self.by_key.items()}

    def reserve(self, text: str) -> None:
        for match in _PLACEHOLDER.finditer(text):
            kind, number = match.groups()
            self.counters[kind] = max(self.counters.get(kind, 0), int(number))

    def new_label(self, kind: str, value: str) -> str:
        self.counters[kind] = self.counters.get(kind, 0) + 1
        label = f"[{kind}_{self.counters[kind]}]"
        self.mapping[label] = value
        return label

    def placeholder(self, kind: str, value: str, participant: str | None) -> str:
        key = (kind, _key(kind, value))
        labels = self.by_key[key]
        if kind != PERSON:
            label = sorted(labels)[0] if labels else self.new_label(kind, value)
        elif participant is None:
            label = self.new_label(kind, value)
        elif participant.startswith("["):
            if participant not in self.initial_labels:
                raise IdentityBindingError("Участник недоступен в этом сеансе")
            label = participant
        elif participant in self.actors:
            label = self.actors[participant]
        else:
            label = self.new_label(kind, value)
            self.actors[participant] = label
        labels.add(label)
        return label

    def apply(self, text, found, origins, bindings):
        parts, positions, pos = [], [], 0
        for ent in found:
            start, end = origins[ent.start][0], origins[ent.end - 1][1]
            participant = bindings.get((start, end)) if ent.type == PERSON else None
            value = text[ent.start : ent.end]
            label = self.placeholder(ent.type, value, participant)
            parts += [text[pos : ent.start], label]
            positions += origins[pos : ent.start] + [(start, end)] * len(label)
            pos = ent.end
            self.entities.append({"type": ent.type, "placeholder": label, "source": ent.source})
            if ent.type == PERSON:
                self.mentions[_key(PERSON, value)].append(
                    {"start": start, "end": end, "placeholder": label}
                )
                if participant is None:
                    self.unbound.add((start, end))
        parts.append(text[pos:])
        positions += origins[pos:]
        return "".join(parts), positions

    def require_identity(self, original: str, masked: str) -> None:
        groups = []
        for key, mentions in self.mentions.items():
            previous = sorted(self.previous.get((PERSON, key), set()))
            has_unbound = any((m["start"], m["end"]) in self.unbound for m in mentions)
            if has_unbound and (len(mentions) > 1 or previous):
                groups.append({"mentions": mentions, "candidates": previous})
        if groups:
            raise IdentityRequired(
                {
                    "code": "person_identity_required",
                    "message": "Уточните, какие упоминания относятся к одному человеку, "
                    "а какие к разным. Повторите вопрос с привязками участников.",
                    "text_sha256": hashlib.sha256(original.encode("utf-8")).hexdigest(),
                    "masked_text": masked,
                    "groups": groups,
                }
            )


def _bindings(text, resolution, found):
    if resolution is None:
        return {}, found
    if resolution.text_sha256 != hashlib.sha256(text.encode("utf-8")).hexdigest():
        raise IdentityBindingError("Текст изменился; получите новое уточнение участников")
    extra = []
    result = {}
    for binding in resolution.bindings:
        start, end = binding.start, binding.end
        if end > len(text) or not re.fullmatch(
            r"[^\W\d_]+\.?(?:(?:[ \t]+|-)[^\W\d_]+\.?)*", text[start:end]
        ):
            raise IdentityBindingError("Некорректный диапазон упоминания")
        overlapping = [e for e in found if start < e.end and e.start < end]
        if overlapping and not (
            len(overlapping) == 1
            and overlapping[0].type == PERSON
            and (overlapping[0].start, overlapping[0].end) == (start, end)
        ):
            raise IdentityBindingError("Привязка пересекает другую находку")
        if not overlapping:
            extra.append(Entity(start, end, PERSON, "user"))
        result[start, end] = binding.participant
    return result, sorted([*found, *extra], key=lambda e: e.start)


async def deidentify(
    text: str,
    model: tuple[Provider, ModelRef] | None = None,
    *,
    mapping: dict[str, str] | None = None,
    person_resolution: PersonResolution | None = None,
) -> DeidResult:
    masker = _Masker(mapping)
    masker.reserve(text)
    bindings, found = _bindings(
        text, person_resolution, _select([*rules.find(text), *names.find(text)])
    )
    masked, origins = masker.apply(text, found, [(i, i + 1) for i in range(len(text))], bindings)
    masker.require_identity(text, masked)
    llm_info = None
    if model is not None:
        extra, llm_info = await llm.find(masked, *model)
        taken = [(m.start(), m.end()) for m in _PLACEHOLDER.finditer(masked)]
        applied = _select(extra, taken)
        masked, _ = masker.apply(masked, applied, origins, bindings)
        if "error" not in llm_info:
            llm_info["found"] = len(applied)
        masker.require_identity(text, masked)
    if mapping is not None:
        mapping.update(masker.mapping)
        if isinstance(mapping, PersonMapping):
            mapping.actors = masker.actors.copy()
            mapping.aliases = {
                key: set(labels) for (kind, key), labels in masker.by_key.items() if kind == PERSON
            }
    return DeidResult(text=masked, entities=masker.entities, mapping=masker.mapping, llm=llm_info)


def reidentify(text: str, mapping: dict[str, str]) -> str:
    """Вернуть исходные значения в ответ — только для показа пользователю."""
    return _PLACEHOLDER.sub(lambda m: mapping.get(m.group(0), m.group(0)), text)


def summary(entities: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ent in entities:
        counts[ent["type"]] = counts.get(ent["type"], 0) + 1
    return counts
