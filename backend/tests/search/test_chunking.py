from dataclasses import asdict

import pytest

from app.modules.corpus.parser import parse_redaction
from app.modules.search.chunking import MAX_CHARS, chunk_article, cite
from tests.corpus.conftest import load_fixture


@pytest.fixture(scope="module")
def articles():
    doc = parse_redaction(*load_fixture("tk_2026_09_01"))
    return {a.key: a for a in doc.articles}


def chunks_of(article):
    return chunk_article(article.text, [asdict(f) for f in article.fragments], article.repealed)


def test_cite_goes_from_smallest_unit():
    ref = [["часть", "1"], ["пункт", "6"], ["подпункт", "а"]]
    assert cite("81", ref, "ТК РФ") == "подп. «а» п. 6 ч. 1 ст. 81 ТК РФ"
    assert cite("56.1") == "ст. 56.1"


def test_parts_are_labelled(articles):
    chunks = chunks_of(articles["58"])
    assert chunks[0].text.startswith("[ч. 1] Трудовые договоры могут заключаться:")
    assert "\n[ч. 2] Срочный трудовой договор" in chunks[0].text
    assert [r for c in chunks for r in c.refs] == [[["часть", str(i)]] for i in range(1, 7)]


def test_long_part_is_split_by_points_with_intro(articles):
    chunks = chunks_of(articles["81"])
    assert len(chunks) > 1
    intro = "Трудовой договор может быть расторгнут работодателем в случаях:"
    # вводные слова части 1 повторяются в каждом фрагменте с её пунктами
    part1 = [c for c in chunks if any(r[0] == ["часть", "1"] for r in c.refs)]
    assert len(part1) > 1 and all(intro in c.text for c in part1)
    prog = next(c for c in chunks if "а) прогула" in c.text)
    assert "6) однократного грубого нарушения работником трудовых обязанностей:" in prog.text
    assert all(len(c.text) <= MAX_CHARS + 200 for c in chunks)
    # каждая часть статьи попала хотя бы в один фрагмент
    covered = {r[0][1] for c in chunks for r in c.refs}
    assert covered == {str(i) for i in range(1, 8)}


def test_article_intro_kept(articles):
    chunks = chunks_of(articles["341.1-1"])
    first_line = articles["341.1-1"].text.split("\n")[0]
    assert all(c.text.startswith(first_line) for c in chunks)


def test_repealed_article_has_no_chunks(articles):
    assert articles["7"].repealed or not articles["7"].text
    assert chunks_of(articles["7"]) == []


def test_text_without_fragments_split_by_lines():
    text = "\n".join(f"Абзац {i}. " + "слово " * 150 for i in range(6))
    chunks = chunk_article(text, [], False)
    assert len(chunks) > 1
    assert all(c.refs == [] for c in chunks)
    assert "".join(c.text.replace("\n", "") for c in chunks) == text.replace("\n", "")
