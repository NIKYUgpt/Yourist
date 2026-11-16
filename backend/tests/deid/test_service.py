import json

import pytest

from app.modules.ai.profiles import ModelRef
from app.modules.deid.identity import IdentityRequired
from app.modules.deid.service import deidentify, reidentify, summary
from tests.deid.identity_helpers import resolution
from tests.fakes import FakeProvider

LLM = ModelRef(provider="ollama", model="qwen3:8b")


async def test_same_person_gets_one_placeholder():
    text = "Иванов Иван Иванович уволил меня. Я просил Иванова Ивана Ивановича подождать."
    with pytest.raises(IdentityRequired):
        await deidentify(text)
    result = await deidentify(
        text,
        person_resolution=resolution(
            text, ("Иванов Иван Иванович", "worker"), ("Иванова Ивана Ивановича", "worker")
        ),
    )
    assert result.text == "[ФИО_1] уволил меня. Я просил [ФИО_1] подождать."
    assert result.mapping == {"[ФИО_1]": "Иванов Иван Иванович"}
    assert summary(result.entities) == {"ФИО": 2}
    assert result.llm is None


async def test_reidentify_restores_values():
    result = await deidentify("Звоните +7 916 123-45-67, спросите Петрова Сергея.")
    answer = "Позвоните [ТЕЛЕФОН_1] и уточните у [ФИО_1]. [НЕИЗВЕСТНО_1] не трогаем."
    assert reidentify(answer, result.mapping) == (
        "Позвоните +7 916 123-45-67 и уточните у Петрова Сергея. [НЕИЗВЕСТНО_1] не трогаем."
    )


async def test_llm_pass_adds_only_verbatim_findings():
    provider = FakeProvider("ollama")
    provider.reply = lambda messages: (
        "```json\n"
        + json.dumps(
            {
                "entities": [
                    {"text": "шурик", "type": "ФИО"},  # со строчной буквы — словарь не ловит
                    {"text": "работодатель", "type": "ФИО"},  # общее слово — отбрасываем
                    {"text": "Сидоров", "type": "ФИО"},  # нет в тексте — отбрасываем
                    {"text": "[ФИО_1]", "type": "ФИО"},  # метка — не трогаем
                    {"text": "склад на Заречной", "type": "АДРЕС"},
                ]
            },
            ensure_ascii=False,
        )
        + "\n```"
    )
    text = (
        "Иванов Иван Иванович и шурик уволили, работодатель молчит. Работали на склад на Заречной."
    )
    result = await deidentify(text, (provider, LLM))
    assert result.text == ("[ФИО_1] и [ФИО_2] уволили, работодатель молчит. Работали на [АДРЕС_1].")
    # модель видит уже замаскированный текст, а не исходный
    sent = provider.chat_calls[0]["messages"][-1]["content"]
    assert "Иванов" not in sent and "[ФИО_1]" in sent
    assert provider.chat_calls[0]["json_mode"] is True
    assert result.llm["found"] == 2
    sources = {e["placeholder"]: e["source"] for e in result.entities}
    assert sources == {"[ФИО_1]": "dictionary", "[ФИО_2]": "llm", "[АДРЕС_1]": "llm"}


async def test_llm_failure_keeps_rule_results():
    provider = FakeProvider("ollama")
    provider.fail = "connection refused"
    result = await deidentify("Звоните 89161234567", (provider, LLM))
    assert result.text == "Звоните [ТЕЛЕФОН_1]"
    assert "connection refused" in result.llm["error"]


async def test_llm_garbage_is_ignored():
    provider = FakeProvider("ollama")
    provider.reply = lambda messages: "не могу ответить"
    result = await deidentify("Просто вопрос про отпуск.", (provider, LLM))
    assert result.text == "Просто вопрос про отпуск." and result.llm["found"] == 0


async def test_llm_hits_inside_placeholders_are_not_counted():
    provider = FakeProvider("ollama")
    # модель вернула кусок метки — совпадение внутри [ФИО_1] отбрасывается
    provider.reply = lambda messages: '{"entities": [{"text": "ФИО_1", "type": "ФИО"}]}'
    result = await deidentify("Иванов Иван Иванович уволен.", (provider, LLM))
    assert result.text == "[ФИО_1] уволен."
    assert result.llm["found"] == 0


@pytest.mark.parametrize("reverse", [False, True])
async def test_known_namesakes_never_resolve_to_last_mapping_entry(reverse):
    entries = [
        ("[ФИО_1]", "Иванов Иван Иванович"),
        ("[ФИО_2]", "Иванов Иван Иванович"),
    ]
    mapping = dict(reversed(entries) if reverse else entries)
    original = mapping.copy()
    text = "[ФИО_1] и [ФИО_2] работают отдельно. Копию передали Иванову Ивану Ивановичу."
    with pytest.raises(IdentityRequired) as required:
        await deidentify(text, mapping=mapping)
    assert required.value.detail["groups"][0]["candidates"] == ["[ФИО_1]", "[ФИО_2]"]
    assert mapping == original
    result = await deidentify(
        text,
        mapping=mapping,
        person_resolution=resolution(text, ("Иванову Ивану Ивановичу", "[ФИО_2]")),
    )
    assert result.text == "[ФИО_1] и [ФИО_2] работают отдельно. Копию передали [ФИО_2]."
    assert mapping == original
    assert reidentify(result.text, result.mapping) == (
        "Иванов Иван Иванович и Иванов Иван Иванович работают отдельно. "
        "Копию передали Иванов Иван Иванович."
    )
    assert "Иванов" not in str(result.entities)


async def test_ambiguous_person_mentions_stay_separate_within_one_call():
    with pytest.raises(IdentityRequired) as required:
        await deidentify(
            "Иванов Иван Иванович подал заявление. Иванову Ивану Ивановичу выдали копию.",
            mapping={"[ФИО_1]": "Иванов Иван Иванович", "[ФИО_2]": "Иванов Иван Иванович"},
        )
    assert required.value.detail["masked_text"] == "[ФИО_3] подал заявление. [ФИО_4] выдали копию."


async def test_single_known_person_and_phone_still_reuse_stable_labels():
    mapping = {"[ФИО_4]": "Иванов Иван Иванович", "[ТЕЛЕФОН_2]": "89161234567"}
    text = "Копию передали Иванову Ивану Ивановичу. Телефон 89161234567"
    result = await deidentify(
        text,
        mapping=mapping,
        person_resolution=resolution(text, ("Иванову Ивану Ивановичу", "[ФИО_4]")),
    )
    assert result.text == "Копию передали [ФИО_4]. Телефон [ТЕЛЕФОН_2]"
    assert result.mapping == mapping
    assert all("identity" not in entity for entity in result.entities)
