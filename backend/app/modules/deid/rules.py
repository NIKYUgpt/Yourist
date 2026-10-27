"""Поиск персональных данных правилами: шаблоны и контрольные суммы.

Контрольные суммы (ИНН, СНИЛС, ОГРН, номер карты) отсекают случайные числа:
«12 месяцев» или номер статьи не примут за документ. Обычные даты не маскируются —
сроки важны для правового ответа; скрывается только дата рождения.
"""

import re
from collections.abc import Callable, Iterator

from app.modules.deid.entities import (
    ACCOUNT,
    ADDRESS,
    BIRTH_DATE,
    CARD,
    EMAIL,
    INN,
    OGRN,
    ORGANIZATION,
    PASSPORT,
    PHONE,
    PROFILE,
    RULE,
    SNILS,
    Entity,
)


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def inn_valid(value: str) -> bool:
    d = [int(c) for c in _digits(value)]

    def check(weights: list[int], n: int) -> int:
        return sum(w * x for w, x in zip(weights, d[:n], strict=False)) % 11 % 10

    if len(d) == 10:
        return check([2, 4, 10, 3, 5, 9, 4, 6, 8], 9) == d[9]
    if len(d) == 12:
        return (
            check([7, 2, 4, 10, 3, 5, 9, 4, 6, 8], 10) == d[10]
            and check([3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8], 11) == d[11]
        )
    return False


def snils_valid(value: str) -> bool:
    d = _digits(value)
    if len(d) != 11:
        return False
    total = sum(int(c) * (9 - i) for i, c in enumerate(d[:9]))
    control = total if total < 100 else 0 if total in (100, 101) else total % 101 % 100
    return control == int(d[9:])


def ogrn_valid(value: str) -> bool:
    d = _digits(value)
    if len(d) == 13:
        return int(d[:12]) % 11 % 10 == int(d[12])
    if len(d) == 15:  # ОГРНИП
        return int(d[:14]) % 13 % 10 == int(d[14])
    return False


def luhn_valid(value: str) -> bool:
    d = [int(c) for c in _digits(value)]
    if not 13 <= len(d) <= 19:
        return False
    total = 0
    for i, x in enumerate(reversed(d)):
        if i % 2:
            x *= 2
            if x > 9:
                x -= 9
        total += x
    return total % 10 == 0


_I = re.IGNORECASE
Validator = Callable[[str, re.Match[str]], bool]


def _checksum(fn: Callable[[str], bool]) -> Validator:
    return lambda text, m: fn(m.group(0))


def _passport(text: str, m: re.Match[str]) -> bool:
    # слово «паспорт» рядом или привычная запись «45 06 123456»
    near = text[max(0, m.start() - 40) : m.start()]
    return bool(re.search(r"паспорт", near, _I)) or bool(
        re.fullmatch(r"\d{2}\s\d{2}\s№?\s?\d{6}|\d{4}\s№?\s?\d{6}", m.group(0))
    )


# порядок важен: при пересечении побеждает правило, стоящее выше
_PATTERNS: list[tuple[str, re.Pattern[str], Validator | None]] = [
    (EMAIL, re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[a-zа-я]{2,}\b", _I), None),
    (
        PROFILE,
        re.compile(
            r"(?:https?://)?(?:www\.)?(?:vk\.com|t\.me|ok\.ru|instagram\.com|facebook\.com)"
            r"/[\w.\-/]*[\w/]",
            _I,
        ),
        None,
    ),
    (PROFILE, re.compile(r"(?<![\w@])@[A-Za-z]\w{3,31}\b"), None),
    (
        PHONE,
        re.compile(
            r"(?<![\d+])(?:\+7|8|7)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}(?!\d)"
        ),
        None,
    ),
    (
        SNILS,
        re.compile(r"(?<!\d)\d{3}[- ]?\d{3}[- ]?\d{3}[- ]?\d{2}(?!\d)"),
        _checksum(snils_valid),
    ),
    (CARD, re.compile(r"(?<!\d)(?:\d{4}[ -]?){3}\d{1,7}(?!\d)"), _checksum(luhn_valid)),
    (ACCOUNT, re.compile(r"(?<!\d)\d{20}(?!\d)"), None),
    (PASSPORT, re.compile(r"(?<!\d)\d{2}\s?\d{2}\s?№?\s?\d{6}(?!\d)"), _passport),
    (OGRN, re.compile(r"(?<!\d)(?:\d{15}|\d{13})(?!\d)"), _checksum(ogrn_valid)),
    (INN, re.compile(r"(?<!\d)(?:\d{12}|\d{10})(?!\d)"), _checksum(inn_valid)),
    (
        ORGANIZATION,
        re.compile(
            r"\b(?:ООО|ОАО|ЗАО|ПАО|АО|НАО|АНО|НКО|ГУП|МУП|ФГУП|ГБУ|МБУ|ФГБУ)\s*"
            r"[«\"“„][^»\"”“]{1,80}[»\"”“]"
        ),
        None,
    ),
    (
        ADDRESS,
        re.compile(
            r"\b(?:ул\.|улиц[аеуы]|пр-т|просп\.|проспект[ае]?|пер\.|переул(?:ок|ке)|"
            r"бул\.|бульвар[ае]?|ш\.|шоссе|наб\.|набережн(?:ая|ой)|пл\.|площад[ьи]|мкр\.?)"
            r"\s*[А-ЯЁ0-9][\w\-ёЁ. ]{0,40}?,?\s*(?:д\.|дом)\s*\d+[а-яА-Я/]*"
            r"(?:,?\s*(?:корп\.|к\.|стр\.|строение|корпус)\s*\d+[а-я]?)?"
            r"(?:,?\s*(?:кв\.|квартира|оф\.|офис)\s*\d+)?",
        ),
        None,
    ),
]

# дата рождения: «родился 12.03.1985», «дата рождения: 12 марта 1985 г.», «1985 г.р.»
_DATE = (
    r"\d{1,2}[./]\d{1,2}[./]\d{2,4}|\d{1,2}\s+(?:января|февраля|марта|апреля|мая|июня|июля|"
    r"августа|сентября|октября|ноября|декабря)\s+\d{4}(?:\s*г(?:ода|\.)?)?"
)
_BIRTH = [
    re.compile(rf"(?:родил(?:ся|ась)|дата\s+рождения|д\.\s?р\.)\W{{0,3}}\s*({_DATE})", _I),
    re.compile(rf"({_DATE}|\d{{4}}\s*)\s*(?:г\.\s?р\.|года\s+рождения)", _I),
]


def find(text: str) -> Iterator[Entity]:
    for kind, pattern, validator in _PATTERNS:
        for m in pattern.finditer(text):
            if validator is None or validator(text, m):
                yield Entity(m.start(), m.end(), kind, RULE)
    for pattern in _BIRTH:
        for m in pattern.finditer(text):
            yield Entity(m.start(1), m.end(1), BIRTH_DATE, RULE)
