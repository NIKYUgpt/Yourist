import json
from pathlib import Path

import pytest

from app.modules.corpus.parser import ParseError, caption_text, parse_redaction

FIXTURES = Path(__file__).parent / "fixtures"


def load(name: str):
    html = (FIXTURES / f"{name}.html").read_text("utf-8")
    toc = json.loads((FIXTURES / f"{name}.toc.json").read_text("utf-8"))
    return parse_redaction(html, toc)


@pytest.fixture(scope="module")
def old():
    return {a.key: a for a in load("tk_2022_07_14").articles}


@pytest.fixture(scope="module")
def new():
    return {a.key: a for a in load("tk_2026_09_01").articles}


def test_all_sample_articles_parsed(old, new):
    assert list(old) == ["1", "7", "56.1", "58", "81"]
    assert list(new) == ["1", "7", "56.1", "58", "81", "341.1-1"]


def test_article_structure_and_path(new):
    art = new["81"]
    assert art.heading == "Статья 81. Расторжение трудового договора по инициативе работодателя"
    assert art.title == "Расторжение трудового договора по инициативе работодателя"
    assert art.path == [
        "ЧАСТЬ ТРЕТЬЯ",
        "Раздел III. ТРУДОВОЙ ДОГОВОР",
        "Глава 13. ПРЕКРАЩЕНИЕ ТРУДОВОГО ДОГОВОРА",
    ]
    assert art.text.startswith("Трудовой договор может быть расторгнут работодателем в случаях:")


def test_fragments_for_parts_points_and_subpoints(new):
    frags = {tuple(map(tuple, f.ref)): f.text for f in new["81"].fragments}
    assert len([ref for ref in frags if len(ref) == 1]) == 7  # части 1–7
    subpoint = frags[(("часть", "1"), ("пункт", "6"), ("подпункт", "а"))]
    assert subpoint.startswith("а) прогула, то есть отсутствия на рабочем месте")
    # текст части включает её пункты
    assert subpoint in frags[(("часть", "1"),)]


def test_editorial_notes_are_separated_from_text(new):
    art = new["81"]
    assert "(В редакции" not in art.text
    assert "(В редакции Федерального закона от 30.06.2006 № 90-ФЗ)" in art.notes


def test_superscript_and_hyphen_numbers(new):
    assert new["56.1"].heading == "Статья 56.1. Запрещение заемного труда"
    assert new["341.1-1"].heading.startswith("Статья 341.1-1. Организации, имеющие право")


def test_repealed_article_detected_from_note(old, new):
    for doc in (old, new):
        art = doc["7"]
        assert art.repealed
        assert art.text.startswith("(Статья утратила силу")


def test_missing_end_paragraph_falls_back_to_next_node(old):
    doc = load("tk_2026_09_01")
    assert any("ст. 58" in w and "граница по следующему узлу" in w for w in doc.warnings)
    art58 = next(a for a in doc.articles if a.key == "58")
    # в редакции 2022 года граница указана верно — тексты должны совпасть
    assert art58.text == old["58"].text


def test_content_hash_detects_real_change(old, new):
    for key in ("1", "7", "56.1", "58"):
        assert old[key].content_hash() == new[key].content_hash(), key
    assert old["81"].content_hash() != new["81"].content_hash()
    added = "а также в случае принятия судом решения о прекращении уголовного преследования"
    assert added not in old["81"].text
    assert added in new["81"].text


@pytest.mark.parametrize(
    ("caption", "expected"),
    [
        (
            "Статья 56<sup>1</sup>. Запрещение заемного труда",
            "Статья 56.1. Запрещение заемного труда",
        ),
        ("Статья 341<sup>1-1</sup>. Организации", "Статья 341.1-1. Организации"),
        ("Глава\xa049<sup>1</sup>. ОСОБЕННОСТИ", "Глава 49.1. ОСОБЕННОСТИ"),
        ('$подпункт "а"', '$подпункт "а"'),
    ],
)
def test_caption_text(caption, expected):
    assert caption_text(caption) == expected


def test_note_paragraphs_and_superscripts_in_body():
    html = (
        "<html><body>"
        '<p class="H" id="p1">Статья 22<span class="W9">1</span>. Пример</p>'
        '<p id="p2">См. статью 19<span class="W9">1</span> '
        '<span class="mark">(В редакции закона)</span>и далее.</p>'
        '<p class="P" id="p3"><span class="mark">(Дополнение статьей - закон)</span></p>'
        "</body></html>"
    )
    toc = [
        {
            "unit": "статья",
            "lvl": 0,
            "np": "p1",
            "npe": "p3",
            "caption": "Статья 22<sup>1</sup>. Пример",
        }
    ]
    [art] = parse_redaction(html, toc).articles
    assert art.key == "22.1"
    assert art.text == "См. статью 19.1 и далее."
    assert art.notes == ["(В редакции закона)", "(Дополнение статьей - закон)"]


def test_document_without_articles_raises():
    with pytest.raises(ParseError):
        parse_redaction("<html><body><p id='p1'>Текст</p></body></html>", [])
