import uuid
from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field

from app.modules.assistant.models import QuestionStatus


class QuestionIn(BaseModel):
    text: str = Field(min_length=3, max_length=4000)
    on: date | None = Field(
        None, alias="date", description="На какую дату отвечать; по умолчанию — сегодня (МСК)"
    )


class SourceOut(BaseModel):
    n: int
    act: str
    key: str
    heading: str
    title: str = Field(description="Подпись источника: акт, части статьи, заголовок")
    refs: list[list[list[str]]]
    text: str
    valid_from: date = Field(description="С какой даты действует редакция")


class CitationOut(BaseModel):
    text: str = Field(description="Как ссылка написана в ответе")
    act: str
    key: str
    part: str | None
    point: str | None
    subpoint: str | None
    status: str = Field(
        description="ok | not_in_sources | unit_not_found | not_found | repealed | external"
    )
    severity: str = Field(description="ok | info | warning | error")
    message: str
    sources: list[int]
    changes_on: list[date] = Field(description="Даты будущих изменений статьи")


class IssueOut(BaseModel):
    severity: str
    code: str
    message: str


class VerificationOut(BaseModel):
    verdict: str = Field(description="supported | partially | unsupported | unknown")
    issues: list[dict[str, str]] = Field(description="Утверждение, проблема, error/warning")
    missing: str = Field(description="Важное из источников, что ответ упустил")
    provider: str | None
    model: str | None
    latency_ms: int | None
    error: str | None


class AnswerOut(BaseModel):
    text: str
    verdict: str = Field(description="passed | warnings | failed | no_sources")
    attempt: int
    citations: list[CitationOut]
    markers: list[int] = Field(description="Номера источников [n], упомянутые в ответе")
    issues: list[IssueOut]
    verification: VerificationOut | None = Field(
        None, description="Сверка ответа с источниками второй моделью"
    )
    sources: list[SourceOut]
    provider: str | None
    model: str | None
    latency_ms: int | None


class QuestionOut(BaseModel):
    id: uuid.UUID
    created_at: datetime
    on_date: date
    text: str
    status: QuestionStatus
    pii: dict[str, int] = Field(description="Сколько персональных данных скрыто, по видам")
    answer: AnswerOut | None
    trace: dict[str, Any] | None = Field(None, description="Этапы обработки (для отладки)")


class QuestionBrief(BaseModel):
    id: uuid.UUID
    created_at: datetime
    on_date: date
    text: str
    status: QuestionStatus
    verdict: str | None


class QuestionList(BaseModel):
    items: list[QuestionBrief]
    total: int
