"""Проверяющая модель: следует ли ответ из источников.

Программная проверка ссылок (citations.py) подтверждает, что названные нормы
существуют и были показаны модели, но не то, что вывод из них следует: модель
может сослаться на настоящую статью и неверно её пересказать. Это проверяет
вторая модель — ей дают те же источники и ответ и просят сверить каждое
утверждение. По умолчанию в режиме API это модель другого разработчика, чем
отвечающая, чтобы их ошибки не совпадали.
"""

import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import date

from app.modules.ai.providers.base import ChatResult, Message, ProviderError
from app.modules.assistant.prompts import source_title
from app.modules.search.retrieval import Hit

SUPPORTED, PARTIALLY, UNSUPPORTED, UNKNOWN = "supported", "partially", "unsupported", "unknown"
# чем меньше, тем лучше — для выбора лучшей попытки
RANK = {SUPPORTED: 0, PARTIALLY: 1, UNKNOWN: 1, UNSUPPORTED: 2}

SYSTEM = """Ты — проверяющий юрист по трудовому праву РФ. Тебе дают вопрос пользователя,
нормы-источники и ответ другого помощника. Проверь каждое утверждение ответа о праве:
подтверждается ли оно текстом источников. Обрати особое внимание на числа, сроки,
суммы, условия и исключения.

Правила:
- сверяй только с источниками, своих знаний не добавляй;
- стиль, полноту изложения и вежливые фразы не оценивай;
- метки вида [ФИО_1] — это скрытые персональные данные, это не ошибка;
- если ответ честно говорит, что источников недостаточно, это не ошибка.

Верни только JSON:
{"verdict": "supported" | "partially" | "unsupported",
 "issues": [{"claim": "утверждение из ответа", "problem": "что не так по источникам",
             "severity": "error" | "warning"}],
 "missing": "важное из источников, что ответ упустил, или пустая строка"}

supported — все утверждения подтверждаются источниками;
partially — главный вывод верен, но есть неточные или неподтверждённые детали;
unsupported — главный вывод противоречит источникам или не следует из них."""

Chat = Callable[..., Awaitable[ChatResult]]


@dataclass
class Verification:
    verdict: str
    issues: list[dict[str, str]] = field(default_factory=list)
    missing: str = ""
    provider: str | None = None
    model: str | None = None
    latency_ms: int | None = None
    error: str | None = None


def build_messages(question: str, sources: list[Hit], answer: str, on: date) -> list[Message]:
    blocks = [f"[{n}] {source_title(hit)}\n{hit.text}" for n, hit in enumerate(sources, 1)]
    user = (
        f"Источники (редакция на {on:%d.%m.%Y}):\n\n"
        + "\n\n".join(blocks)
        + f"\n\nВопрос пользователя: {question}\n\nОтвет помощника:\n{answer}"
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def parse(raw: str) -> Verification:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        data = json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        data = None
    if not isinstance(data, dict) or data.get("verdict") not in RANK or data["verdict"] == UNKNOWN:
        return Verification(UNKNOWN, error=f"ответ проверяющей модели не разобран: {raw[:200]}")
    issues = [
        {
            "claim": str(i.get("claim", ""))[:500],
            "problem": str(i.get("problem", ""))[:500],
            "severity": "error" if i.get("severity") == "error" else "warning",
        }
        for i in data.get("issues") or []
        if isinstance(i, dict)
    ]
    return Verification(data["verdict"], issues, str(data.get("missing") or "")[:1000])


async def verify(
    chat: Chat, question: str, sources: list[Hit], answer: str, on: date
) -> Verification:
    try:
        result = await chat(
            build_messages(question, sources, answer, on),
            temperature=0.0,
            max_tokens=1000,
            json_mode=True,
        )
    except ProviderError as exc:
        return Verification(UNKNOWN, error=str(exc))
    verification = parse(re.sub(r"<think>.*?</think>", "", result.text, flags=re.DOTALL))
    verification.provider = result.provider
    verification.model = result.model
    verification.latency_ms = result.latency_ms
    return verification


def needs_revision(v: Verification) -> bool:
    """Переписывать ли ответ по замечаниям проверяющей модели.

    Небольшие модели неверный факт обычно помечают как partially с замечанием уровня
    error, а не как unsupported. На оценке (evaluation/results/2026-10-27-labor-basic,
    verify.md) правило «unsupported или partially с error» у yandexgpt-5-lite:8b ловит
    86% подменённых фактов при 3% ложных срабатываний; один unsupported — лишь 7%.
    """
    if v.verdict == UNSUPPORTED:
        return True
    return v.verdict == PARTIALLY and any(i["severity"] == "error" for i in v.issues)


def revision(verification: Verification) -> str:
    problems = "\n".join(
        f"- «{i['claim']}»: {i['problem']}"
        for i in verification.issues
        if i["claim"] or i["problem"]
    )
    return (
        "Проверяющий сверил ответ с источниками и нашёл проблемы:\n"
        f"{problems or '- главный вывод не следует из источников'}\n"
        "Перепиши ответ так, чтобы каждое утверждение подтверждалось источниками. "
        "Если источников недостаточно — так и скажи."
    )
