"""Эксперимент D: согласование только внутри разрешённых меток.

Морфология описывает форму слова, не личность или пол человека. Неоднозначные
разборы сохраняют исходную подстановку. Модели, сеть и постоянное хранение не нужны.
"""

import re
from dataclasses import dataclass, field

from app.modules.deid.names import morph
from app.modules.deid.service import _PLACEHOLDER, reidentify

CASES = frozenset({"nomn", "gent", "datv", "accs", "ablt", "loct"})
PREPOSITIONS = {
    "для": "gent",
    "у": "gent",
    "от": "gent",
    "без": "gent",
    "до": "gent",
    "после": "gent",
    "к": "datv",
    "перед": "ablt",
    "между": "ablt",
    "о": "loct",
    "об": "loct",
}
_NAME_WORD = re.compile(r"[А-ЯЁ][а-яё]+")


@dataclass(frozen=True)
class Replacement:
    start: int
    end: int
    label: str
    value: str


@dataclass
class Restoration:
    text: str
    changed: int = 0
    unknown: int = 0
    fallback: bool = False
    # Только тип причины и позиция, исходные значения сюда не попадают.
    notes: list[dict] = field(default_factory=list)


def inflect_person(value: str, case: str) -> str | None:
    if case not in CASES:
        return None
    words = list(_NAME_WORD.finditer(value))
    # Инициалы, дефисы, несклоняемые/неизвестные формы пока требуют отдельной методики.
    if not words or " ".join(m.group() for m in words) != value:
        return None
    parses = [
        [
            p
            for p in morph().parse(m.group())
            if any(t in p.tag for t in ("Name", "Surn", "Patr")) and "sing" in p.tag and p.is_known
        ]
        for m in words
    ]
    if any(not group for group in parses):
        return None
    # Согласованность грамматических разборов, без сохранения признака в профиле лица.
    genders = set.intersection(*[{p.tag.gender for p in group if p.tag.gender} for group in parses])
    if len(genders) != 1:
        return None
    gender = next(iter(genders))
    output = []
    for group in parses:
        variants = {
            form.word
            for p in group
            if p.tag.gender == gender
            if (form := p.inflect({case})) is not None
        }
        if len(variants) != 1:
            return None
        output.append(next(iter(variants)).capitalize())
    return " ".join(output)


def case_from_context(text: str, start: int) -> str | None:
    match = re.search(r"\b([А-Яа-яЁё]+)\s+$", text[:start])
    return PREPOSITIONS.get(match.group(1).lower()) if match else None


def apply_replacements(text: str, mapping: dict[str, str], edits: list[Replacement]) -> str:
    """Нельзя менять границы, метку, соседние слова или вставлять чужое значение.

    Для коррекции допускается только проверенная морфологическая форма того же
    значения. Даже будущий LLM-корректор не получает права переписывать правовой текст.
    """
    allowed = {
        (m.start(), m.end()): m.group() for m in _PLACEHOLDER.finditer(text) if m.group() in mapping
    }
    if len({(e.start, e.end) for e in edits}) != len(edits):
        raise ValueError("duplicate replacement spans")
    by_span = {}
    for edit in edits:
        if allowed.get((edit.start, edit.end)) != edit.label:
            raise ValueError("replacement outside mapped placeholder")
        original = mapping[edit.label]
        valid = {original}
        if edit.label.startswith("[ФИО_"):
            valid.update(v for case in CASES if (v := inflect_person(original, case)) is not None)
        if edit.value not in valid:
            raise ValueError("replacement changes entity identity")
        by_span[(edit.start, edit.end)] = edit.value
    parts, pos = [], 0
    for match in _PLACEHOLDER.finditer(text):
        parts.append(text[pos : match.start()])
        parts.append(
            by_span.get((match.start(), match.end()), mapping.get(match.group(), match.group()))
        )
        pos = match.end()
    parts.append(text[pos:])
    return "".join(parts)


def restore_grammar(text: str, mapping: dict[str, str]) -> Restoration:
    baseline = reidentify(text, mapping)
    result = Restoration(baseline)
    edits = []
    for match in _PLACEHOLDER.finditer(text):
        label = match.group()
        if not label.startswith("[ФИО_") or label not in mapping:
            continue
        case = case_from_context(text, match.start())
        value = inflect_person(mapping[label], case) if case else None
        if value is None:
            result.unknown += 1
            result.notes.append({"offset": match.start(), "reason": "unknown_grammar"})
        elif value != mapping[label]:
            edits.append(Replacement(match.start(), match.end(), label, value))
    try:
        result.text = apply_replacements(text, mapping, edits)
        result.changed = len(edits)
    except (ValueError, TypeError):
        result.fallback = True
        result.notes.append({"reason": "invalid_replacement"})
    return result
