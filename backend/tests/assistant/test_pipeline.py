import re

import httpx
import pytest
from sqlalchemy import select

from app.core.roles import Role
from app.main import app
from app.modules.ai.service import load_config, save_config
from app.modules.assistant.models import Answer, Question
from app.modules.deid.api import DeidClient, DeidSettings
from tests.conftest import bearer, login, register_verified, set_role

API = "/api/v1/assistant/questions"
QUESTION = (
    "Меня, Петрова Сергея Ивановича, хотят уволить за прогул. "
    "Я отсутствовал 5 часов. Мой телефон 89161234567. Законно ли это?"
)


def source_number(messages, key: str) -> int:
    """Номер источника со статьёй key в промпте — как его увидела бы модель."""
    prompt = messages[-1]["content"] if messages[-1]["role"] == "user" else messages[1]["content"]
    match = re.search(rf"^\[(\d+)\] ТК РФ, [^\n]*ст\. {re.escape(key)}\b", prompt, re.M)
    assert match, prompt[:500]
    return int(match.group(1))


def good_answer(messages) -> str:
    n = source_number(messages[:2], "81")
    return (
        f"Да, [ФИО_1] можно уволить за прогул: это отсутствие более четырёх часов подряд [{n}] "
        "(подп. «а» п. 6 ч. 1 ст. 81 ТК РФ). Это справочная информация."
    )


def is_deid(messages) -> bool:
    return "модуль деперсонализации" in messages[0]["content"]


def is_verify(messages) -> bool:
    return "проверяющий юрист" in messages[0]["content"]


def is_analysis(messages) -> bool:
    return "Раздели вопрос пользователя" in messages[0]["content"]


SUPPORTED = '{"verdict": "supported", "issues": [], "missing": ""}'


def service_reply(messages) -> str | None:
    if is_deid(messages):
        return '{"entities": []}'
    if is_verify(messages):
        return SUPPORTED
    return None


@pytest.fixture
def ollama(ai_registry):
    provider = ai_registry.get("ollama")
    provider.reply = lambda m: service_reply(m) or good_answer(m)
    return provider


@pytest.fixture
async def user(client, mailer):
    await register_verified(client, mailer, "user@example.com")
    return bearer(await login(client, "user@example.com"))


def answer_calls(provider):
    return [
        c
        for c in provider.chat_calls
        if not any(f(c["messages"]) for f in (is_deid, is_verify, is_analysis))
    ]


async def test_full_flow(client, db, indexed, ollama, user):
    resp = await client.post(API, json={"text": QUESTION, "date": "2026-10-01"}, headers=user)
    assert resp.status_code == 201, resp.text
    body = resp.json()

    # пользователю — с его данными
    assert body["text"] == QUESTION
    assert body["answer"]["text"].startswith("Да, Петрова Сергея Ивановича можно уволить")
    assert body["pii"] == {"ФИО": 1, "ТЕЛЕФОН": 1}
    answer = body["answer"]
    assert answer["verdict"] == "passed", answer["issues"]
    [citation] = answer["citations"]
    assert (citation["key"], citation["part"], citation["point"], citation["subpoint"]) == (
        "81",
        "1",
        "6",
        "а",
    )
    assert answer["sources"] and all(s["valid_from"] == "2026-09-01" for s in answer["sources"])
    assert answer["markers"] == [source_number(answer_calls(ollama)[0]["messages"], "81")]
    assert set(body["trace"]) == {"deid", "search", "answer"}

    # модель видела только обезличенный текст
    prompt = answer_calls(ollama)[0]["messages"][1]["content"]
    assert "Петров" not in prompt and "89161234567" not in prompt
    assert "[ФИО_1]" in prompt and "[ТЕЛЕФОН_1]" in prompt

    # в БД — только обезличенный текст
    question = await db.scalar(select(Question))
    stored = await db.scalar(select(Answer))
    assert "Петров" not in question.text and "[ФИО_1]" in question.text
    assert "Петров" not in stored.text and stored.is_final


async def test_wrong_citation_is_corrected(client, db, indexed, ollama, user):
    replies = iter(["Это регулируется ст. 999 ТК РФ [1].", None])

    def reply(messages):
        if service_reply(messages):
            return service_reply(messages)
        first = next(replies)
        return first if first else good_answer(messages)

    ollama.reply = reply
    resp = await client.post(API, json={"text": QUESTION, "date": "2026-10-01"}, headers=user)
    body = resp.json()
    assert body["answer"]["verdict"] == "passed" and body["answer"]["attempt"] == 2
    second = answer_calls(ollama)[1]["messages"]
    assert second[-2]["role"] == "assistant" and "ст. 999" in second[-2]["content"]
    assert "ст. 999 ТК РФ: такой статьи нет" in second[-1]["content"]
    rows = list(await db.scalars(select(Answer).order_by(Answer.attempt)))
    assert [(a.attempt, a.verdict, a.is_final) for a in rows] == [
        (1, "failed", False),
        (2, "passed", True),
    ]


async def test_unfixable_answer_is_returned_with_errors(client, indexed, ollama, user):
    ollama.reply = lambda m: service_reply(m) or "См. ст. 999 ТК РФ."
    body = (await client.post(API, json={"text": QUESTION}, headers=user)).json()
    assert body["answer"]["verdict"] == "failed"
    assert body["answer"]["issues"][0]["code"] == "not_found"
    assert len(answer_calls(ollama)) == 2


async def test_no_sources_skips_model(client, indexed, ollama, user):
    resp = await client.post(API, json={"text": QUESTION, "date": "2001-01-01"}, headers=user)
    body = resp.json()
    assert body["answer"]["verdict"] == "no_sources"
    assert "Не нашёл в базе норм" in body["answer"]["text"]
    assert answer_calls(ollama) == []


async def test_model_failure(client, db, indexed, ollama, user):
    # модель недоступна: второй проход деперсонализации пропускается, ответ — 502
    ollama.fail = "connection refused"
    resp = await client.post(API, json={"text": QUESTION}, headers=user)
    assert resp.status_code == 502 and "connection refused" in resp.json()["detail"]
    question = await db.scalar(select(Question))
    assert question.status == "failed" and "Петров" not in question.text
    assert "connection refused" in question.trace["deid"]["llm"]["error"]


async def test_deid_unavailable_stops_pipeline(client, indexed, ollama, user, ai_registry):
    def down(request):
        raise httpx.ConnectError("refused")

    app.state.deid_client = DeidClient(
        DeidSettings(deid_url="http://deid"), ai_registry, transport=httpx.MockTransport(down)
    )
    resp = await client.post(API, json={"text": QUESTION}, headers=user)
    assert resp.status_code == 503
    assert ollama.chat_calls == []  # исходный текст никуда не ушёл


async def test_deid_llm_can_be_disabled(client, db, indexed, ollama, user):
    config = await load_config(db)
    config.deid_llm = False
    await save_config(db, config, None)
    resp = await client.post(API, json={"text": QUESTION}, headers=user)
    assert resp.status_code == 201
    assert not any(is_deid(c["messages"]) for c in ollama.chat_calls)
    assert resp.json()["trace"]["deid"]["llm"] is None


async def test_history_and_access(client, db, mailer, indexed, ollama, user):
    resp = await client.post(API, json={"text": QUESTION}, headers=user)
    qid = resp.json()["id"]

    history = (await client.get(API, headers=user)).json()
    assert history["total"] == 1
    assert history["items"][0]["verdict"] == "passed"
    assert "[ФИО_1]" in history["items"][0]["text"]  # в истории — обезличенный текст

    own = (await client.get(f"{API}/{qid}", headers=user)).json()
    assert "Петров" not in own["answer"]["text"] and own["trace"] is None

    await register_verified(client, mailer, "other@example.com")
    other = bearer(await login(client, "other@example.com"))
    assert (await client.get(f"{API}/{qid}", headers=other)).status_code == 404
    assert (await client.delete(f"{API}/{qid}", headers=other)).status_code == 404

    await register_verified(client, mailer, "support@example.com")
    await set_role(db, "support@example.com", Role.SUPPORT)
    support = bearer(await login(client, "support@example.com"))
    seen = (await client.get(f"{API}/{qid}", headers=support)).json()
    assert seen["trace"] is not None and "Петров" not in seen["text"]

    assert (await client.delete(f"{API}/{qid}", headers=user)).status_code == 204
    assert (await client.get(f"{API}/{qid}", headers=user)).status_code == 404
    assert await db.scalar(select(Answer)) is None


async def test_requires_login(client):
    assert (await client.post(API, json={"text": QUESTION})).status_code == 401


async def test_verifier_result_is_returned(client, indexed, ollama, user):
    body = (await client.post(API, json={"text": QUESTION}, headers=user)).json()
    verification = body["answer"]["verification"]
    assert verification["verdict"] == "supported" and verification["provider"] == "ollama"
    # проверяющей модели показали те же источники и ответ, но не исходный текст
    [call] = [c for c in ollama.chat_calls if is_verify(c["messages"])]
    prompt = call["messages"][1]["content"]
    assert "Ответ помощника:" in prompt and "[ФИО_1]" in prompt and "Петров" not in prompt
    assert call["json_mode"] is True


async def test_unsupported_answer_is_revised(client, db, indexed, ollama, user):
    verdicts = iter(
        [
            '{"verdict": "unsupported", "issues": [{"claim": "можно уволить", '
            '"problem": "в источнике — более четырёх часов подряд", "severity": "error"}]}',
            SUPPORTED,
        ]
    )

    def reply(messages):
        if is_deid(messages):
            return '{"entities": []}'
        if is_verify(messages):
            return next(verdicts)
        return good_answer(messages)

    ollama.reply = reply
    body = (await client.post(API, json={"text": QUESTION}, headers=user)).json()
    assert body["answer"]["attempt"] == 2
    assert body["answer"]["verification"]["verdict"] == "supported"
    revision = answer_calls(ollama)[1]["messages"][-1]["content"]
    assert "Проверяющий сверил ответ" in revision and "четырёх часов подряд" in revision
    rows = list(await db.scalars(select(Answer).order_by(Answer.attempt)))
    assert [(a.attempt, a.is_final, a.report["verification"]["verdict"]) for a in rows] == [
        (1, False, "unsupported"),
        (2, True, "supported"),
    ]


async def test_verification_can_be_disabled(client, db, indexed, ollama, user):
    config = await load_config(db)
    config.verify = False
    await save_config(db, config, None)
    body = (await client.post(API, json={"text": QUESTION}, headers=user)).json()
    assert body["answer"]["verification"] is None
    assert not any(is_verify(c["messages"]) for c in ollama.chat_calls)


async def test_unreadable_verification_does_not_block_answer(client, indexed, ollama, user):
    ollama.reply = lambda m: (
        '{"entities": []}' if is_deid(m) else "не знаю" if is_verify(m) else good_answer(m)
    )
    body = (await client.post(API, json={"text": QUESTION}, headers=user)).json()
    assert body["answer"]["verdict"] == "passed"
    assert body["answer"]["verification"]["verdict"] == "unknown"
    assert body["answer"]["attempt"] == 1


COMPOUND = (
    "Меня, Петрова Сергея Ивановича, уволили за прогул, хотя я отсутствовал пять часов. "
    "Договор у меня срочный, его продлевали три раза. Законно ли это, можно ли оспорить "
    "увольнение и что делать в первую очередь?"
)
ISSUES = (
    '{"issues": [{"issue": "Увольнение за прогул", "query": "увольнение за прогул"}, '
    '{"issue": "Многократно продлённый срочный договор", "query": "срочный трудовой договор"}]}'
)


async def test_compound_question_is_searched_by_issues(client, db, indexed, ollama, user):
    ollama.reply = lambda m: ISSUES if is_analysis(m) else service_reply(m) or good_answer(m)
    resp = await client.post(API, json={"text": COMPOUND, "date": "2026-10-01"}, headers=user)
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["answer"]["verdict"] == "passed"

    analysis = body["trace"]["analysis"]
    assert [i["query"] for i in analysis["issues"]] == [
        "увольнение за прогул",
        "срочный трудовой договор",
    ]
    assert all(i["sources"] for i in analysis["issues"])
    assert {"81", "58"} <= {s["key"] for s in body["answer"]["sources"]}

    # модель разбора видела только обезличенный вопрос
    [call] = [c for c in ollama.chat_calls if is_analysis(c["messages"])]
    assert call["json_mode"]
    assert "Петров" not in call["messages"][-1]["content"]
    assert "[ФИО_1]" in call["messages"][-1]["content"]

    # исходный вопрос не заменяется: отвечающая и проверяющая модели видят его целиком
    masked = (await db.scalar(select(Question))).text
    prompt = answer_calls(ollama)[0]["messages"][1]["content"]
    assert f"Вопрос: {masked}" in prompt
    assert "В вопросе несколько проблем" in prompt
    assert "1. Увольнение за прогул (источники: [" in prompt
    [check] = [c for c in ollama.chat_calls if is_verify(c["messages"])]
    assert f"Вопрос пользователя: {masked}" in check["messages"][1]["content"]


async def test_failed_analysis_falls_back_to_plain_search(client, indexed, ollama, user):
    ollama.reply = lambda m: "не знаю" if is_analysis(m) else service_reply(m) or good_answer(m)
    body = (await client.post(API, json={"text": COMPOUND}, headers=user)).json()
    assert body["answer"]["verdict"] == "passed"
    assert body["trace"]["analysis"]["error"].startswith("ответ модели разбора не разобран")
    prompt = answer_calls(ollama)[0]["messages"][1]["content"]
    assert "В вопросе несколько проблем" not in prompt


async def test_short_question_and_disabled_analysis_skip_the_model(
    client, db, indexed, ollama, user
):
    await client.post(API, json={"text": QUESTION}, headers=user)
    config = await load_config(db)
    config.analyze = False
    await save_config(db, config, None)
    body = (await client.post(API, json={"text": COMPOUND}, headers=user)).json()
    assert "analysis" not in body["trace"]
    assert not any(is_analysis(c["messages"]) for c in ollama.chat_calls)
