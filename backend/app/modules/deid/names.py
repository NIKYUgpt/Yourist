"""Поиск ФИО по морфологическому словарю (pymorphy3, словари OpenCorpora).

Словарь помечает имена (Name), фамилии (Surn) и отчества (Patr) и даёт вероятность
каждого разбора. Отсюда правила:
- цепочка слов с заглавной буквы, где есть имя или отчество, — ФИО («Зайцев Иван Петрович»);
- фамилия рядом с инициалами — ФИО («Иванов И. И.», «И.И. Иванов»);
- одиночное слово — только если словарь уверен, что это имя или фамилия; в начале
  предложения порог выше: «Вера» и «Надежда» там скорее обычные слова.
"""

import re
from dataclasses import dataclass
from functools import lru_cache

import pymorphy3

from app.modules.deid.entities import DICTIONARY, PERSON, Entity

_WORD = re.compile(r"[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?")
_INITIALS = re.compile(r"[А-ЯЁ]\.\s?(?:[А-ЯЁ]\.)?")
_PERSON_TAGS = ("Name", "Surn", "Patr")
# слова, которые словарь иногда принимает за имена, а в обращениях юристам — обычные
_NOT_NAMES = frozenset(
    {"Роса", "Мир", "Закон", "Суд", "Кодекс", "Работодатель", "Работник", "Директор", "Слава"}
)
SINGLE_WORD_SCORE = 0.5
SENTENCE_START_SCORE = 0.9


@lru_cache(maxsize=1)
def morph() -> pymorphy3.MorphAnalyzer:
    return pymorphy3.MorphAnalyzer()


@dataclass
class _Token:
    start: int
    end: int
    text: str
    person: float  # суммарная вероятность разборов «имя/фамилия/отчество»
    patronymic: bool
    first_name: float


def _token(m: re.Match[str]) -> _Token:
    word = m.group(0)
    parses = morph().parse(word.split("-")[0])
    person = sum(p.score for p in parses if any(t in p.tag for t in _PERSON_TAGS))
    first_name = sum(p.score for p in parses if "Name" in p.tag)
    patronymic = any("Patr" in p.tag for p in parses)
    if word in _NOT_NAMES:
        person = first_name = 0.0
    return _Token(m.start(), m.end(), word, person, patronymic, first_name)


def _sentence_start(text: str, pos: int) -> bool:
    before = text[:pos].rstrip(" «\"'(—-")
    return not before or before[-1] in ".!?\n:"


def find(text: str) -> list[Entity]:
    tokens = [_token(m) for m in _WORD.finditer(text)]
    found: list[Entity] = []
    used: set[int] = set()

    # цепочки соседних слов с заглавной буквы
    i = 0
    while i < len(tokens):
        j = i
        while j + 1 < len(tokens) and text[tokens[j].end : tokens[j + 1].start].strip() == "":
            j += 1
        chain = tokens[i : j + 1]
        if len(chain) >= 2:
            anchor = any(t.patronymic or t.first_name >= SINGLE_WORD_SCORE for t in chain)
            if anchor and all(t.person > 0 for t in chain):
                found.append(Entity(chain[0].start, chain[-1].end, PERSON, DICTIONARY))
                used.update(range(i, j + 1))
        i = j + 1

    # фамилия с инициалами до или после
    for idx, tok in enumerate(tokens):
        if idx in used or tok.text in _NOT_NAMES:
            continue
        after = _INITIALS.match(text, tok.end + 1) if text[tok.end : tok.end + 1] == " " else None
        before = None
        for m in _INITIALS.finditer(text, max(0, tok.start - 6), tok.start):
            if text[m.end() : tok.start].strip() == "":
                before = m
        # два инициала — надёжный признак; один — только при слове-фамилии по словарю
        if after and (after.group(0).count(".") == 2 or tok.person > 0):
            found.append(Entity(tok.start, after.end(), PERSON, DICTIONARY))
            used.add(idx)
        elif before and tok.person > 0:
            found.append(Entity(before.start(), tok.end, PERSON, DICTIONARY))
            used.add(idx)

    # одиночные слова
    for idx, tok in enumerate(tokens):
        if idx in used:
            continue
        threshold = SENTENCE_START_SCORE if _sentence_start(text, tok.start) else SINGLE_WORD_SCORE
        if tok.person >= threshold:
            found.append(Entity(tok.start, tok.end, PERSON, DICTIONARY))
    return found
