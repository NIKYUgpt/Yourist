"""Метрики сравнения: без нейросетей, по регулярным выражениям и проверке ссылок.

Факт засчитывается, если его шаблон найден в ответе. Шаблоны нестрогие («3 месяца»,
«трёх месяцев», «три месяца»), но всё равно это оценка снизу: перефразированный
верный ответ может не совпасть с шаблоном. Поэтому ответы сохраняются целиком
для ручной проверки.
"""

import re
from dataclasses import dataclass, field
from typing import Any

from app.modules.assistant.citations import CheckReport
from app.modules.search.retrieval import Hit

_REFUSAL = re.compile(
    r"не\s+относится|только\s+на\s+вопросы|не\s+могу\s+(по)?мочь|вне\s+(моей\s+)?компетенц",
    re.IGNORECASE,
)


@dataclass
class Question:
    id: str
    text: str
    articles: list[str]
    facts: list[dict[str, str]]
    off_topic: bool = False
    on: str | None = None  # своя дата вопроса (проверка учёта редакции)
    # шаблоны неверного вывода: если найден — ответ неверен, даже если факты совпали
    forbidden: list[dict[str, str]] = field(default_factory=list)


@dataclass
class AnswerScore:
    facts_found: list[str] = field(default_factory=list)
    facts_missed: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    cites_expected: bool = False
    refused: bool = False
    empty: bool = False

    @property
    def fact_share(self) -> float:
        if self.contradictions:
            return 0.0
        total = len(self.facts_found) + len(self.facts_missed)
        return len(self.facts_found) / total if total else 1.0

    @property
    def correct(self) -> bool:
        return not self.empty and not self.facts_missed and not self.contradictions


def score_answer(q: Question, text: str, report: CheckReport, sources: list[Hit]) -> AnswerScore:
    score = AnswerScore(empty=not text.strip())
    flat = " ".join(text.replace("\xa0", " ").split())
    for fact in q.facts:
        found = re.search(fact["pattern"], flat, re.IGNORECASE)
        (score.facts_found if found else score.facts_missed).append(fact["name"])
    for bad in q.forbidden:
        if re.search(bad["pattern"], flat, re.IGNORECASE):
            score.contradictions.append(bad["name"])
    cited = {c.key for c in report.citations if c.status != "external"}
    cited |= {sources[n - 1].key for n in report.markers if 1 <= n <= len(sources)}
    score.cites_expected = bool(set(q.articles) & cited)
    score.refused = bool(_REFUSAL.search(flat))
    return score


def retrieval_rank(q: Question, hits: list[Hit]) -> int | None:
    """Позиция первого фрагмента нужной статьи в выдаче (с 1); None — не найдена."""
    for i, hit in enumerate(hits, 1):
        if hit.key in q.articles:
            return i
    return None


def load_questions(data: dict[str, Any]) -> list[Question]:
    return [Question(**q) for q in data["questions"]]
