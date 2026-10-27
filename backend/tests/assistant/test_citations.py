from datetime import date

import pytest

from app.modules.assistant.citations import check, extract, markers
from app.modules.search.retrieval import search
from tests.search.conftest import BEFORE_AGENCIES, NOW


def one(text: str, default: str = "ТК РФ"):
    [c] = extract(text, default)
    return c.act, c.key, c.part, c.point, c.subpoint


def test_extract_units_and_act():
    assert one("по п. 6 ч. 1 ст. 81 ТК РФ") == ("ТК РФ", "81", "1", "6", None)
    assert one("подп. «а» п. 6 ч. 1 ст. 81") == ("ТК РФ", "81", "1", "6", "а")
    assert one("согласно части первой статьи 81 Трудового кодекса РФ") == (
        "ТК РФ",
        "81",
        "1",
        None,
        None,
    )
    assert one("ст. 341.1-1") == ("ТК РФ", "341.1-1", None, None, None)
    assert one("ст. 15 ГК РФ") == ("ГК РФ", "15", None, None, None)
    assert one("ст. 9 Федерального закона от 27.07.2006 № 152-ФЗ")[0] == (
        "Федеральный закон № 152-ФЗ"
    )


def test_extract_lists_and_ranges():
    keys = [c.key for c in extract("см. ст. 80 и 81, а также статьи 84.1, 140", "ТК РФ")]
    assert keys == ["80", "81", "84.1", "140"]
    assert [c.key for c in extract("ст. 80-81", "ТК РФ")] == ["80", "81"]
    # единицы перед перечнем относятся только к первой статье
    first, second = extract("ч. 1 ст. 80 и 81", "ТК РФ")
    assert (first.part, second.part) == ("1", None)


def test_markers():
    assert markers("Да [1], но [2, 3] и [4–6]. [ФИО_1] — не ссылка") == [1, 2, 3, 4, 5, 6]
    # модели часто дописывают норму прямо в скобки
    assert markers("28 дней [1, ч. 1 ст. 115 ТК РФ], выплата [3, ч. 6].") == [1, 3]
    assert markers("без ссылок") == []


@pytest.fixture
async def sources(db, indexed, gateway):
    return (await search(db, gateway, "срочный трудовой договор срок", NOW, k=4)).hits


def codes(report) -> list[str]:
    return [i.code for i in report.issues]


async def test_correct_answer_passes(db, sources):
    n = next(i for i, h in enumerate(sources, 1) if h.key == "58")
    answer = f"Срочный договор заключается на срок не более пяти лет [{n}] (ч. 1 ст. 58 ТК РФ)."
    report = await check(db, answer, sources, NOW)
    assert report.verdict == "passed", report.issues
    [c] = report.citations
    assert (c.key, c.part, c.status) == ("58", "1", "ok")
    assert n in c.sources  # ст. 58 может быть в нескольких источниках


@pytest.mark.parametrize(
    ("answer", "code", "verdict"),
    [
        ("Так сказано в ст. 999 ТК РФ.", "not_found", "failed"),
        ("См. ст. 7 ТК РФ.", "repealed", "failed"),
        ("Это следует из источника [12].", "bad_marker", "failed"),
        ("Заёмный труд запрещён (ст. 56.1 ТК РФ).", "not_in_sources", "warnings"),
        ("Так сказано в ч. 9 ст. 58 ТК РФ.", "unit_not_found", "warnings"),
        ("Работник вправе уволиться в любой момент.", "no_citations", "warnings"),
    ],
)
async def test_problems_found(db, sources, answer, code, verdict):
    report = await check(db, answer, sources, NOW)
    assert code in codes(report)
    assert report.verdict == verdict


async def test_external_acts_are_not_errors(db, sources):
    report = await check(db, "Вред возмещается по ст. 1064 ГК РФ.", sources, NOW)
    [c] = report.citations
    assert (c.status, c.severity) == ("external", "info")
    assert report.verdict == "passed"


async def test_quotes_must_be_verbatim(db, sources):
    good = "«Трудовые договоры могут заключаться: 1) на неопределенный срок» (ст. 58 ТК РФ)"
    bad = "«Срочный договор можно продлевать без ограничений по желанию работодателя» (ст. 58)"
    assert "quote_not_found" not in codes(await check(db, good, sources, NOW))
    assert "quote_not_found" in codes(await check(db, bad, sources, NOW))


async def test_known_future_change_is_reported(db, indexed, gateway):
    sources = (
        await search(db, gateway, "расторжение по инициативе работодателя", BEFORE_AGENCIES)
    ).hits
    report = await check(db, "Это п. 7.1 ч. 1 ст. 81 ТК РФ.", sources, BEFORE_AGENCIES)
    [c] = report.citations
    assert c.changes_on == [date(2026, 9, 1)]
    assert "изменится с 01.09.2026" in c.message
