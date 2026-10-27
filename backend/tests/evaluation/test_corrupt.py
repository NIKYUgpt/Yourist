import pytest

from app.evaluation.corrupt import corrupt

FACT = [{"name": "x", "pattern": "(3|тр[её]х|три)[\s-]*(х\s*)?месяц"}]


@pytest.mark.parametrize(
    ("answer", "facts", "expected"),
    [
        ("Срок — не более трёх месяцев [1].", FACT, "Срок — не более четырёх месяцев [1]."),
        ("Не более 3 месяцев.", FACT, "Не более 4 месяцев."),
        (
            "Отпуск — 28 календарных дней.",
            [{"name": "28", "pattern": "28\s*дн|28\s*календарн"}],
            "Отпуск — 42 календарных дней.",
        ),
        (
            "Ночное время — с 22 часов до 6 часов.",
            [{"name": "22-6", "pattern": "22\D{1,20}6"}],
            "Ночное время — с 23 часов до 6 часов.",
        ),
        (
            "Работать можно с 16 лет.",
            [{"name": "16", "pattern": "16|шестнадцат"}],
            "Работать можно с 18 лет.",
        ),
        (
            "Нет, увольнение не допускается, кроме ликвидации.",
            [{"name": "нельзя", "pattern": "не\s+допускается"}],
            "Нет, увольнение допускается, кроме ликвидации.",
        ),
        (
            "Оплата — не менее чем в двойном размере.",
            [{"name": "двойн", "pattern": "двойн"}],
            "Оплата — не менее чем в полуторном размере.",
        ),
    ],
)
def test_corrupt_changes_the_fact(answer, facts, expected):
    new, change = corrupt(answer, facts)
    assert new == expected and "→" in change


def test_inverts_conclusion_when_fact_has_nothing_to_change():
    facts = [{"name": "приостановить", "pattern": "приостанов"}]
    answer = "Да, вы имеете право не выходить на работу. Приостановление запрещено врачам."
    assert corrupt(answer, facts)[0].startswith("Да, вы не имеете права не выходить")


def test_nothing_to_corrupt():
    assert corrupt("Обратитесь к юристу.", [{"name": "x", "pattern": "юрист"}]) is None
