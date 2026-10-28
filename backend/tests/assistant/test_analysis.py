import json

import pytest

from app.modules.ai.providers.base import ChatResult, ProviderError
from app.modules.assistant.analysis import (
    MAX_ISSUES,
    analyze,
    is_compound,
    parse,
    sentences,
)

COMPOUND = (
    "Меня уволили по сокращению штата. Уведомление о сокращении я получил за 2 недели, "
    "а при расчёте не выплатили выходное пособие. Какие у меня права и куда обращаться?"
)


def test_long_multi_sentence_question_is_compound():
    assert is_compound(COMPOUND)
    assert is_compound("Работаю без договора. " * 20)  # длинный — всегда


def test_short_question_is_not_compound():
    assert not is_compound("Можно ли уволить за прогул?")
    # три предложения, но коротко — одна проблема
    assert not is_compound("Меня уволили. Я опоздал. Законно ли?")
    assert not is_compound("Зарплату задерживают уже больше двух недель. Могу ли я не выходить?")


def test_document_question_is_not_decomposed():
    question = "Законно ли это?\n\nТекст документа (распознан с фото):\n" + "Приказ. " * 100
    assert not is_compound(question)


def test_sentences_ignore_abbreviations():
    text = "Меня уволили по ст. 81 ТК. Предупредили за 2 нед. до этого. [ФИО_1] сказал, что можно."
    assert sentences(text) == [
        "Меня уволили по ст. 81 ТК.",
        "Предупредили за 2 нед. до этого.",
        "[ФИО_1] сказал, что можно.",
    ]


def test_parse_issues():
    raw = json.dumps(
        {
            "issues": [
                {"issue": "Срок предупреждения", "query": "предупреждение о сокращении [ФИО_1]"},
                {"issue": "Срок предупреждения", "query": "предупреждение о сокращении"},
                {"issue": "Пособие", "query": ""},
                {"issue": "", "query": ""},
                "мусор",
            ]
        },
        ensure_ascii=False,
    )
    issues = parse("<think></think>" + raw)
    # метки убраны из запроса, повтор отброшен, пустой запрос заменён формулировкой
    assert [(i.issue, i.query) for i in issues] == [
        ("Срок предупреждения", "предупреждение о сокращении"),
        ("Пособие", "Пособие"),
    ]


def test_parse_limits_issue_count():
    raw = json.dumps({"issues": [{"issue": f"п{i}", "query": f"запрос {i}"} for i in range(9)]})
    assert len(parse(raw)) == MAX_ISSUES


@pytest.mark.parametrize("raw", ["не JSON", '{"issues": "текст"}', "[1, 2]"])
def test_parse_rejects_garbage(raw):
    with pytest.raises(ValueError):
        parse(raw)


async def test_analyze_reports_errors_instead_of_raising():
    async def failing(messages, **kwargs):
        raise ProviderError("ollama", "timeout")

    analysis = await analyze(failing, COMPOUND)
    assert analysis.issues == [] and not analysis.compound
    assert "timeout" in analysis.error

    async def garbage(messages, **kwargs):
        return ChatResult("не знаю", "ollama", "qwen3:8b", latency_ms=5)

    analysis = await analyze(garbage, COMPOUND)
    assert analysis.error.startswith("ответ модели разбора не разобран")
    assert (analysis.provider, analysis.latency_ms) == ("ollama", 5)


async def test_analyze_asks_for_json():
    calls = []

    async def chat(messages, **kwargs):
        calls.append((messages, kwargs))
        return ChatResult(
            '{"issues": [{"issue": "a", "query": "a"}, {"issue": "b", "query": "b"}]}',
            "ollama",
            "qwen3:8b",
            latency_ms=1,
        )

    analysis = await analyze(chat, COMPOUND)
    assert analysis.compound and [i.query for i in analysis.issues] == ["a", "b"]
    [(messages, kwargs)] = calls
    assert kwargs["json_mode"] is True and kwargs["temperature"] == 0.0
    assert messages[-1]["content"] == f"Вопрос: {COMPOUND}"
