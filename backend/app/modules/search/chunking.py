"""Нарезка статьи на фрагменты для поиска.

Фрагмент — несколько соседних частей (пунктов) статьи, около тысячи символов.
Длинную часть режем по её пунктам и подпунктам, повторяя в каждом фрагменте
вводные слова («Трудовой договор может быть расторгнут работодателем в случаях:»),
иначе пункт теряет смысл. Части в тексте кодекса не пронумерованы — подписываем
их «[ч. 2]», чтобы модель могла сослаться на конкретную часть статьи.
"""

import re
from dataclasses import dataclass, field
from typing import Any

# версия алгоритма: при изменении нарезки фрагменты пересобираются
CHUNKER_VERSION = 1
TARGET_CHARS = 1200
MAX_CHARS = 1800

UNIT_ABBR = {"часть": "ч.", "пункт": "п.", "подпункт": "подп.", "абзац": "абз."}
_SENTENCE_END = re.compile(r"(?<=[.;:])\s+")

Ref = list[list[str]]  # [["часть", "1"], ["пункт", "6"]]


def unit_label(unit: list[str]) -> str:
    name, num = unit
    abbr = UNIT_ABBR.get(name, name)
    return f"{abbr} «{num}»" if name == "подпункт" else f"{abbr} {num}"


def cite(key: str, ref: Ref | None = None, act: str | None = None) -> str:
    """Ссылка в принятом порядке — от мелкой единицы к крупной: «подп. «а» п. 6 ч. 1 ст. 81»."""
    parts = [unit_label(u) for u in reversed(ref or [])] + [f"ст. {key}"]
    if act:
        parts.append(act)
    return " ".join(parts)


@dataclass
class _Node:
    ref: Ref
    text: str
    children: list["_Node"] = field(default_factory=list)


@dataclass
class _Segment:
    ref: Ref
    intros: tuple[str, ...]
    text: str


@dataclass
class ChunkDraft:
    ord: int
    refs: list[Ref]
    text: str


def _tree(fragments: list[dict[str, Any]]) -> list[_Node]:
    roots: list[_Node] = []
    stack: list[_Node] = []
    for frag in fragments:
        node = _Node(ref=frag["ref"], text=frag["text"])
        while stack and not (
            len(stack[-1].ref) < len(node.ref) and node.ref[: len(stack[-1].ref)] == stack[-1].ref
        ):
            stack.pop()
        (stack[-1].children if stack else roots).append(node)
        stack.append(node)
    return roots


def _split_long(text: str) -> list[str]:
    """Текст без вложенных единиц: по строкам, длинные строки — по предложениям."""
    pieces: list[str] = []
    for line in text.split("\n"):
        if len(line) <= MAX_CHARS:
            pieces.append(line)
            continue
        current = ""
        for sentence in _SENTENCE_END.split(line):
            if current and len(current) + len(sentence) + 1 > TARGET_CHARS:
                pieces.append(current)
                current = ""
            current = f"{current} {sentence}".strip()
            while len(current) > MAX_CHARS:
                pieces.append(current[:MAX_CHARS])
                current = current[MAX_CHARS:]
        if current:
            pieces.append(current)
    return [p for p in pieces if p.strip()]


def _segments(node: _Node, intros: tuple[str, ...]) -> list[_Segment]:
    if len(node.text) <= MAX_CHARS:
        return [_Segment(node.ref, intros, node.text)]
    if node.children:
        start = node.text.find(node.children[0].text)
        intro = node.text[:start].strip() if start > 0 else ""
        inner = intros + ((intro,) if intro else ())
        return [seg for child in node.children for seg in _segments(child, inner)]
    return [_Segment(node.ref, intros, piece) for piece in _split_long(node.text)]


def _part_label(ref: Ref) -> str | None:
    return f"[{unit_label(ref[0])}]" if ref and ref[0][0] == "часть" else None


def _render(segments: list[_Segment]) -> str:
    lines: list[str] = []
    shown_intros: tuple[str, ...] = ()
    current_part = None
    for seg in segments:
        label = _part_label(seg.ref)
        new_part = label is not None and label != current_part
        if new_part:
            current_part = label
        if seg.intros != shown_intros:
            # вводные слова родительских единиц, которых ещё не было в этом фрагменте
            common = 0
            while (
                common < min(len(seg.intros), len(shown_intros))
                and seg.intros[common] == shown_intros[common]
            ):
                common += 1
            for intro in seg.intros[common:]:
                lines.append(f"{label} {intro}" if new_part and label else intro)
                new_part = False
            shown_intros = seg.intros
        lines.append(f"{label} {seg.text}" if new_part and label else seg.text)
    return "\n".join(lines)


def chunk_article(
    text: str, fragments: list[dict[str, Any]], is_repealed: bool
) -> list[ChunkDraft]:
    if is_repealed or not text.strip():
        return []
    roots = _tree(fragments)
    if roots:
        # вводная фраза статьи перед первым пунктом («… имеют право:») в пункты не входит
        start = text.find(roots[0].text)
        intro = (text[:start].strip(),) if start > 0 else ()
        segments = [seg for root in roots for seg in _segments(root, intro)]
    else:
        segments = [_Segment([], (), piece) for piece in _split_long(text)]

    chunks: list[ChunkDraft] = []
    batch: list[_Segment] = []
    size = 0
    for seg in segments:
        extra = len(seg.text) + sum(
            len(i) for i in seg.intros if not batch or i not in batch[-1].intros
        )
        if batch and size + extra > TARGET_CHARS:
            chunks.append(_draft(len(chunks), batch))
            batch, size = [], 0
            extra = len(seg.text) + sum(len(i) for i in seg.intros)
        batch.append(seg)
        size += extra
    if batch:
        chunks.append(_draft(len(chunks), batch))
    return chunks


def _draft(ord_: int, segments: list[_Segment]) -> ChunkDraft:
    refs: list[Ref] = []
    for seg in segments:
        if seg.ref and seg.ref not in refs:
            refs.append(seg.ref)
    return ChunkDraft(ord=ord_, refs=refs, text=_render(segments))
