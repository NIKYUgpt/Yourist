import pytest

from app.modules.ai.providers.base import ChatResult, ProviderError
from app.modules.assistant.context import contextualize

HISTORY = [
    {"role": "user", "content": "[ФИО_1] уволили за прогул."},
    {"role": "assistant", "content": "Непроверенный старый вывод."},
]


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        '{"query": 42}',
        '{"query": ""}',
        '{"query": "[ФИО_2]"}',
        '{"query": "' + "x" * 1001 + '"}',
    ],
)
async def test_invalid_rewrite_falls_back_without_assistant_claims(raw):
    async def chat(*args, **kwargs):
        return ChatResult(raw, "fake", "fake", latency_ms=1)

    result = await contextualize(chat, "А если три часа?", HISTORY)
    assert result.fallback
    assert "[ФИО_1]" in result.query and "А если три часа?" in result.query
    assert "Непроверенный" not in result.query


async def test_model_unavailable_has_safe_fallback():
    async def chat(*args, **kwargs):
        raise ProviderError("fake", "down")

    result = await contextualize(chat, "А если три часа?", HISTORY)
    assert result.fallback and result.error == "context_model_unavailable"


async def test_valid_rewrite_preserves_current_question_in_input():
    async def chat(messages, **kwargs):
        assert "А если три часа?" in messages[-1]["content"]
        assert "[ФИО_1]" in messages[-1]["content"]
        assert kwargs["json_mode"]
        return ChatResult(
            '{"query": "[ФИО_1] отсутствовал три часа, увольнение за прогул"}',
            "fake",
            "fake",
            latency_ms=2,
        )

    result = await contextualize(chat, "А если три часа?", HISTORY)
    assert not result.fallback and "три часа" in result.query
