"""Разбор сложного вопроса на отдельные правовые проблемы.

Длинный вопрос из нескольких проблем («сократили, предупредили за две недели,
не выплатили пособие — куда обращаться?») одним поиском на 8 фрагментов
покрывается частично: фрагменты достаются самой заметной проблеме, а сроки
обращения в суд или компенсация за задержку выплат в выдачу не попадают.
Модель разбивает вопрос на проблемы, и по каждой ищутся свои нормы
(retrieval.search_issues).

Исходный вопрос при этом не заменяется: отвечающая модель видит его целиком,
а проблемы — как перечень того, что нельзя упустить; проверяющая модель тоже
сверяет ответ с исходным вопросом. Если разбор не удался, вопрос идёт обычным путём.
"""

import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from app.modules.ai.providers.base import ChatResult, Message, ProviderError

MAX_ISSUES = 6
# разбор нужен длинным вопросам из нескольких предложений; вопрос про одну норму
# обычный поиск находит и так (в наборах labor_basic и labor_hard — не больше двух
# предложений и 111 символов, разбор на них не включается)
MIN_SENTENCES = 3
MIN_CHARS = 150
LONG_CHARS = 400

# граница предложения: знак конца и следующее слово с заглавной буквы
# («ст. 81» и «2 нед.» предложение не разрывают)
_BOUNDARY = re.compile(r"(?<=[.!?…])\s+(?=[«\"(\[]?[А-ЯЁA-Z])")
_PLACEHOLDER = re.compile(r"\[[А-ЯЁ_]+_\d+\]")
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_DOCUMENT = "Текст документа (распознан с фото)"

SYSTEM = """Ты — юрист по трудовому праву РФ. Раздели вопрос пользователя на отдельные
правовые проблемы: по каждой нужно искать свою норму закона.

Правила:
- от 1 до 6 проблем; если проблема одна — верни одну;
- каждое отдельное нарушение, выплата, срок или обстоятельство, для которого в законе своя
  норма, — отдельная проблема: «предупредили поздно» и «не выплатили пособие» — две проблемы;
- не пропускай обстоятельства, которые пользователь специально упомянул: возраст,
  беременность, повторные срочные договоры, устное предупреждение, удержания из зарплаты;
- прямые вопросы пользователя («сколько часов можно работать», «сколько должны заплатить»,
  «можно ли отозвать») — отдельные пункты;
- пункт о порядке и сроках обращения в суд или инспекцию труда добавляй, только если
  пользователь спрашивает, куда обращаться, как оспорить или в какой срок подать в суд;
- для каждой проблемы — поисковый запрос из терминов Трудового кодекса;
- не отвечай на вопрос, не называй номера статей и не добавляй проблем, которых в вопросе нет;
- метки вида [ФИО_1] — скрытые персональные данные, в запросы их не переноси.

Пример. Вопрос: «Меня перевели на другую должность без моего согласия, урезали зарплату,
а за прошлый месяц вообще не заплатили. Это законно?»
Ответ:
{"issues": [
 {"issue": "Перевод на другую должность без согласия",
  "query": "перевод на другую работу без согласия работника"},
 {"issue": "Снижение зарплаты",
  "query": "изменение определенных сторонами условий трудового договора заработная плата"},
 {"issue": "Невыплата зарплаты за месяц",
  "query": "нарушение сроков выплаты заработной платы ответственность работодателя"}
]}

Верни только JSON:
{"issues": [{"issue": "проблема своими словами", "query": "поисковый запрос"}]}"""

Chat = Callable[..., Awaitable[ChatResult]]


@dataclass
class Issue:
    issue: str
    query: str


@dataclass
class Analysis:
    issues: list[Issue] = field(default_factory=list)
    provider: str | None = None
    model: str | None = None
    latency_ms: int | None = None
    error: str | None = None

    @property
    def compound(self) -> bool:
        """Проблем две и больше — искать по каждой; одна — обычный поиск."""
        return len(self.issues) >= 2


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _BOUNDARY.split(text.strip()) if s.strip()]


def is_compound(text: str) -> bool:
    """Стоит ли разбирать вопрос: дешёвая проверка до вызова модели."""
    if _DOCUMENT in text:
        return False  # вопрос к документу разбирается по шагам (prompts.DOCUMENT_STEPS)
    if len(text) >= LONG_CHARS:
        return True
    return len(text) >= MIN_CHARS and len(sentences(text)) >= MIN_SENTENCES


def build_messages(question: str) -> list[Message]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Вопрос: {question}"},
    ]


def parse(raw: str) -> list[Issue]:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        data = json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        data = None
    if not isinstance(data, dict) or not isinstance(data.get("issues"), list):
        raise ValueError(f"ответ модели разбора не разобран: {raw[:200]}")
    issues: list[Issue] = []
    for item in data["issues"]:
        if not isinstance(item, dict):
            continue
        issue = " ".join(str(item.get("issue") or "").split())[:300]
        query = " ".join(_PLACEHOLDER.sub(" ", str(item.get("query") or issue)).split())[:300]
        if query and all(query != i.query for i in issues):
            issues.append(Issue(issue or query, query))
    return issues[:MAX_ISSUES]


async def analyze(chat: Chat, question: str) -> Analysis:
    try:
        result = await chat(
            build_messages(question), temperature=0.0, max_tokens=800, json_mode=True
        )
    except ProviderError as exc:
        return Analysis(error=str(exc))
    analysis = Analysis(provider=result.provider, model=result.model, latency_ms=result.latency_ms)
    try:
        analysis.issues = parse(_THINK.sub("", result.text))
    except ValueError as exc:
        analysis.error = str(exc)
    return analysis
