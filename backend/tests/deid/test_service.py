import json

from app.modules.ai.profiles import ModelRef
from app.modules.deid.service import deidentify, reidentify, summary
from tests.fakes import FakeProvider

LLM = ModelRef(provider="ollama", model="qwen3:8b")


async def test_same_person_gets_one_placeholder():
    text = "Иванов Иван Иванович уволил меня. Я просил Иванова Ивана Ивановича подождать."
    result = await deidentify(text)
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
