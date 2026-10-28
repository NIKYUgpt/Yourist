"""Конвейер ответа на вопрос.

1. Деперсонализация (отдельный сервис). Если он недоступен — вопрос дальше не идёт.
2. Поиск норм, действующих на дату вопроса. Длинный вопрос из нескольких проблем
   модель сначала разбирает на проблемы, и нормы ищутся по каждой (analysis.py).
3. Ответ модели со ссылками на нумерованные источники.
4. Детерминированная проверка ссылок; при ошибках — одна попытка исправления
   с перечнем ошибок.
5. Проверяющая модель сверяет лучший ответ с источниками; если вывод
   не подтверждается или есть серьёзные замечания — ещё одна попытка с ними.
Из всех попыток остаётся лучшая: меньше ошибок в ссылках, затем лучше вердикт проверки.
"""

import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.ai.profiles import ModelRef, Task
from app.modules.ai.providers.base import ChatResult, ProviderError, Timer
from app.modules.ai.service import AIGateway
from app.modules.assistant import prompts
from app.modules.assistant.analysis import Analysis, analyze, is_compound, sentences
from app.modules.assistant.citations import CheckReport, check
from app.modules.assistant.models import Answer, Question, QuestionSource, QuestionStatus
from app.modules.assistant.verifier import RANK, Verification, needs_revision, revision, verify
from app.modules.deid.api import DeidClient
from app.modules.deid.service import DeidResult, reidentify, summary
from app.modules.search.retrieval import Hit, search, search_issues

# попыток на исправление ссылок; плюс одна — по замечаниям проверяющей модели
MAX_ATTEMPTS = 2
SOURCES_K = 8
MAX_TOKENS = 1500
# ответ по нескольким проблемам длиннее
MAX_TOKENS_COMPOUND = 2000
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


class AnswerFailed(Exception):
    def __init__(self, question: Question, message: str):
        super().__init__(message)
        self.question = question


@dataclass
class Attempt:
    result: ChatResult
    text: str
    report: CheckReport
    verification: Verification | None = None


@dataclass
class Outcome:
    """Результат для ответа пользователю: с восстановленными данными, в БД не пишется."""

    question: Question
    answer: Answer
    mapping: dict[str, str] = field(default_factory=dict)

    def restore(self, text: str) -> str:
        return reidentify(text, self.mapping)


def _source_dump(hit: Hit, n: int) -> dict[str, Any]:
    return {
        "n": n,
        "chunk_id": hit.chunk_id,
        "act": hit.act,
        "key": hit.key,
        "heading": hit.heading,
        "refs": hit.refs,
        "text": hit.text,
        "valid_from": hit.valid_from.isoformat(),
        "title": prompts.source_title(hit),
    }


def _report_dump(report: CheckReport) -> dict[str, Any]:
    data = asdict(report)
    for c in data["citations"]:
        c["changes_on"] = [d.isoformat() for d in c["changes_on"]]
    return data


def _verification_brief(v: Verification | None) -> dict[str, Any] | None:
    if v is None:
        return None
    return {
        "verdict": v.verdict,
        "provider": v.provider,
        "model": v.model,
        "latency_ms": v.latency_ms,
        "issues": len(v.issues),
        "error": v.error,
    }


def _better(a: Attempt, b: Attempt) -> Attempt:
    """Меньше ошибок в ссылках, лучше вердикт проверки, меньше предупреждений;
    при равенстве — более поздняя попытка."""

    def cost(x: Attempt) -> tuple[int, int, int]:
        rank = RANK[x.verification.verdict] if x.verification else 1
        return len(x.report.errors), rank, len(x.report.issues)

    return a if cost(a) < cost(b) else b


class Pipeline:
    def __init__(self, session: AsyncSession, ai: AIGateway, deid: DeidClient):
        self.session = session
        self.ai = ai
        self.deid = deid

    def deid_model(self) -> ModelRef | None:
        """Модель второго прохода деперсонализации, если он включён."""
        return self.ai.resolve(Task.DEID)[1] if self.ai.config.deid_llm else None

    async def ask(self, user_id: uuid.UUID, text: str, on: date) -> Outcome:
        timer = Timer()
        deid: DeidResult = await self.deid.deidentify(text, self.deid_model())
        trace = {"deid": {"ms": timer.ms, "found": summary(deid.entities), "llm": deid.llm}}
        return await self.answer(user_id, deid, on, QuestionSource.TEXT, trace)

    async def answer(
        self,
        user_id: uuid.UUID,
        deid: DeidResult,
        on: date,
        source: QuestionSource,
        trace: dict[str, Any],
    ) -> Outcome:
        """Ответ на уже обезличенный вопрос (текст, распознанный голос или документ)."""
        question = Question(
            user_id=user_id,
            text=deid.text,
            on_date=on,
            source=source,
            pii=summary(deid.entities),
            status=QuestionStatus.ANSWERED,
            trace=trace,
        )
        # в сессию — только в конце: запросы поиска не должны сбрасывать вопрос в БД раньше

        analysis = await self._analyze(deid.text, trace)
        timer = Timer()
        issues: list[tuple[str, list[int]]] = []
        if analysis is not None and analysis.compound:
            # ищем и по каждому предложению вопроса: если модель упустила проблему,
            # её нормы всё равно попадут в источники (evaluation/results/
            # 2026-10-28-labor-compound: статья проблемы в источниках 79% → 84%)
            queries = [i.query for i in analysis.issues] + sentences(deid.text)
            found, linked = await search_issues(self.session, self.ai, deid.text, queries, on)
            linked = linked[: len(analysis.issues)]
            issues = [
                (i.issue, numbers) for i, numbers in zip(analysis.issues, linked, strict=True)
            ]
            for item, numbers in zip(trace["analysis"]["issues"], linked, strict=True):
                item["sources"] = numbers
        else:
            found = await search(self.session, self.ai, deid.text, on, k=SOURCES_K)
        trace["search"] = {
            "ms": timer.ms,
            "space": found.space,
            "notes": found.notes,
            "timings_ms": found.timings_ms,
            "hits": [[h.key, h.refs[:2], h.score] for h in found.hits],
        }
        sources = found.hits
        max_tokens = MAX_TOKENS_COMPOUND if issues else MAX_TOKENS

        if not sources:
            answer = Answer(
                attempt=0,
                is_final=True,
                text=prompts.NO_SOURCES.format(on=f"{on:%d.%m.%Y}"),
                verdict="no_sources",
                report={"citations": [], "markers": [], "issues": [], "verdict": "no_sources"},
                sources=[],
            )
            return await self._save(question, [answer], deid)

        base = prompts.build_messages(deid.text, sources, on, issues)
        messages = base
        attempts: list[Attempt] = []
        try:
            for _ in range(MAX_ATTEMPTS):
                attempt = await self._answer(messages, sources, on, max_tokens)
                attempts.append(attempt)
                if attempt.report.verdict != "failed":
                    break
                messages = [
                    *messages,
                    {"role": "assistant", "content": attempt.text},
                    {"role": "user", "content": prompts.correction(attempt.report)},
                ]
            best = attempts[0]
            for attempt in attempts[1:]:
                best = _better(best, attempt)

            if self.ai.config.verify:
                best.verification = await self._verify(deid.text, sources, best.text, on)
                if needs_revision(best.verification):
                    messages = [
                        *base,
                        {"role": "assistant", "content": best.text},
                        {"role": "user", "content": revision(best.verification)},
                    ]
                    revised = await self._answer(messages, sources, on, max_tokens)
                    revised.verification = await self._verify(deid.text, sources, revised.text, on)
                    attempts.append(revised)
                    best = _better(best, revised)
        except ProviderError as exc:
            if not attempts:
                question.status = QuestionStatus.FAILED
                question.error = str(exc)
                trace["answer"] = {"error": str(exc)}
                self.session.add(question)
                await self.session.commit()
                raise AnswerFailed(question, str(exc)) from exc
            best = attempts[0]
            for attempt in attempts[1:]:
                best = _better(best, attempt)

        trace["answer"] = {
            "attempts": [
                {
                    "provider": a.result.provider,
                    "model": a.result.model,
                    "latency_ms": a.result.latency_ms,
                    "prompt_tokens": a.result.usage.prompt_tokens,
                    "completion_tokens": a.result.usage.completion_tokens,
                    "verdict": a.report.verdict,
                    "verification": _verification_brief(a.verification),
                }
                for a in attempts
            ]
        }
        source_dump = [_source_dump(h, n) for n, h in enumerate(sources, 1)]
        answers = [
            Answer(
                attempt=i,
                is_final=a is best,
                text=a.text,
                verdict=a.report.verdict,
                report={
                    **_report_dump(a.report),
                    "verification": asdict(a.verification) if a.verification else None,
                },
                sources=source_dump,
                provider=a.result.provider,
                model=a.result.model,
                latency_ms=a.result.latency_ms,
                prompt_tokens=a.result.usage.prompt_tokens,
                completion_tokens=a.result.usage.completion_tokens,
            )
            for i, a in enumerate(attempts, 1)
        ]
        return await self._save(question, answers, deid)

    async def _analyze(self, question: str, trace: dict[str, Any]) -> Analysis | None:
        """Разбор длинного вопроса на проблемы; None — разбор не нужен или выключен.

        Сбой разбора не останавливает ответ: вопрос идёт обычным поиском.
        """
        if not self.ai.config.analyze or not is_compound(question):
            return None

        async def chat(messages, **kwargs):
            return await self.ai.chat(Task.ANALYZE, messages, **kwargs)

        analysis = await analyze(chat, question)
        trace["analysis"] = {
            "provider": analysis.provider,
            "model": analysis.model,
            "latency_ms": analysis.latency_ms,
            "issues": [asdict(i) for i in analysis.issues],
            "error": analysis.error,
        }
        return analysis

    async def _answer(
        self,
        messages: list[dict[str, str]],
        sources: list[Hit],
        on: date,
        max_tokens: int = MAX_TOKENS,
    ) -> Attempt:
        result = await self.ai.chat(Task.ANSWER, messages, temperature=0.1, max_tokens=max_tokens)
        text = _THINK.sub("", result.text).strip()
        return Attempt(result, text, await check(self.session, text, sources, on))

    async def _verify(
        self, question: str, sources: list[Hit], answer: str, on: date
    ) -> Verification:
        async def chat(messages, **kwargs):
            return await self.ai.chat(Task.VERIFY, messages, **kwargs)

        return await verify(chat, question, sources, answer, on)

    async def _save(self, question: Question, answers: list[Answer], deid: DeidResult) -> Outcome:
        self.session.add(question)
        await self.session.flush()
        for answer in answers:
            answer.question_id = question.id
        self.session.add_all(answers)
        await self.session.commit()
        final = next(a for a in answers if a.is_final)
        return Outcome(question=question, answer=final, mapping=deid.mapping)
