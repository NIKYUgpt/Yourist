"""Подмена ключевого факта в верном ответе — для проверки проверяющей модели.

Берём место ответа, где найден эталонный факт, и меняем в нём число или слово
на правдоподобное неверное: «28 дней» → «42 дня», «трёх месяцев» → «четырёх
месяцев», «не допускается» → «допускается». Правдоподобие важно: заведомо
абсурдное значение («с 33 часов») проверяющая модель поймает слишком легко.
"""

import re

# фразы (порядок важен: «не вправе» раньше «вправе»)
_PHRASES = [
    (r"не\s+допускается", "допускается"),
    (r"не\s+может", "может"),
    (r"не\s+вправе", "вправе"),
    (r"нельзя", "можно"),
    (r"имеете\s+право", "не имеете права"),
    (r"имеет\s+право", "не имеет права"),
    (r"(?<!не )вправе", "не вправе"),
    (r"средн\w+\s+месячн\w+\s+заработ\w*", "двухнедельного заработка"),
    (r"(в\s+)?последний\s+(рабочий\s+)?день", "в течение недели после увольнения"),
    (r"(в\s+)?день\s+увольнения", "в течение месяца после увольнения"),
    (r"сто\s+пятидесят\w*", "трёхсотой"),
    (r"1\s*/\s*150", "1/300"),
    (r"одного\s+месяца", "двух месяцев"),
    (r"один\s+месяц", "два месяца"),
    (r"одного\s+года", "двух лет"),
    (r"один\s+год", "два года"),
]
_WORDS = {
    "два": "три",
    "две": "три",
    "двух": "трёх",
    "три": "четыре",
    "трёх": "четырёх",
    "трех": "четырех",
    "четыре": "шесть",
    "четырёх": "шести",
    "четырех": "шести",
    "пять": "семь",
    "пяти": "семи",
    "шесть": "девять",
    "шести": "девяти",
    "шестнадцати": "четырнадцати",
    "шестнадцать": "четырнадцать",
    "полуторном": "одинарном",
    "полуторного": "одинарного",
    "двойном": "полуторном",
    "двойного": "полуторного",
    "полмесяца": "месяц",
    "ликвидации": "сокращения штата",
    "выговор": "штраф",
}
_TOKEN = re.compile(r"\d+|[А-Яа-яЁё]+")


def _wrong_number(value: str, before: str, after: str) -> str:
    n = int(value)
    if re.match(r"\s*час", after) and re.search(r"(^|\s)(с|до|от)\s*$", before) and n <= 24:
        return str(n - 1 if n + 1 > 24 else n + 1)  # время суток: 22 → 21, а не 33
    if re.match(r"\s*(лет|года)", after):
        return str(n + 2)  # возраст и сроки в годах: 16 → 18
    return str(n + max(1, n // 2))


def _match_case(original: str, replacement: str) -> str:
    return replacement.capitalize() if original[:1].isupper() else replacement


def _replace(flat: str, a: int, b: int, new: str) -> tuple[str, str]:
    new = _match_case(flat[a:b], new)
    return flat[:a] + new + flat[b:], f"«{flat[a:b]}» → «{new}»"


def _around(flat: str, start: int, end: int) -> tuple[str, str] | None:
    # шаблон может захватить часть слова («двойн») — расширяем до границ слов
    while start > 0 and flat[start - 1].isalnum():
        start -= 1
    while end < len(flat) and flat[end].isalnum():
        end += 1
    lo, hi = max(0, start - 25), min(len(flat), end + 25)
    for pattern, wrong in _PHRASES:
        for p in re.finditer(pattern, flat[lo:hi], re.IGNORECASE):
            a, b = lo + p.start(), lo + p.end()
            if b > start and a < end + 25:
                return _replace(flat, a, b, wrong)
    for token in _TOKEN.finditer(flat, start, end):
        word = token.group(0)
        a, b = token.span()
        if word.isdigit():
            return _replace(
                flat, a, b, _wrong_number(word, flat[max(0, a - 6) : a], flat[b : b + 10])
            )
        if word.lower() in _WORDS:
            return _replace(flat, a, b, _WORDS[word.lower()])
    return None


def corrupt(answer: str, facts: list[dict[str, str]]) -> tuple[str, str] | None:
    """Возвращает (испорченный ответ, что заменено) или None, если подменить нечего."""
    flat = " ".join(answer.replace("\xa0", " ").replace(" ", " ").split())
    for fact in facts:
        for m in re.finditer(fact["pattern"], flat, re.IGNORECASE):
            changed = _around(flat, *m.span())
            if changed:
                return changed
    # рядом с фактом менять нечего — переворачиваем вывод в первом предложении
    first = re.match(r"[^.!?]*", flat)
    for pattern, wrong in _PHRASES[:7]:
        p = re.search(pattern, first.group(0), re.IGNORECASE) if first else None
        if p:
            return _replace(flat, p.start(), p.end(), wrong)
    return None
