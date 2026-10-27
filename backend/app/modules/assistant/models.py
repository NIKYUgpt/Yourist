"""Вопросы пользователей и ответы (схема assistant).

Хранится только деперсонализированный текст: исходный вопрос и таблица меток
живут в памяти на время запроса. Удаление пользователя удаляет его вопросы.
"""

import enum
import uuid
from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Date,
    DateTime,
    Enum,
    ForeignKey,
    Identity,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base

SCHEMA = "assistant"


class QuestionStatus(enum.StrEnum):
    ANSWERED = "answered"
    FAILED = "failed"


class Question(Base):
    __tablename__ = "questions"
    __table_args__ = {"schema": SCHEMA}

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("auth.users.id", ondelete="CASCADE"), index=True
    )
    text: Mapped[str] = mapped_column(Text)  # после деперсонализации
    on_date: Mapped[date] = mapped_column(Date)
    # сколько персональных данных каждого вида нашли — без самих значений
    pii: Mapped[dict[str, int]] = mapped_column(JSONB, default=dict)
    status: Mapped[QuestionStatus] = mapped_column(
        Enum(
            QuestionStatus,
            name="question_status",
            native_enum=False,
            create_constraint=True,
            length=16,
            values_callable=lambda items: [i.value for i in items],
        )
    )
    error: Mapped[str | None] = mapped_column(Text)
    # этапы конвейера: время, модели, заметки поиска — для отладки и оценки
    trace: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Answer(Base):
    __tablename__ = "answers"
    __table_args__ = {"schema": SCHEMA}

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    question_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey(f"{SCHEMA}.questions.id", ondelete="CASCADE"), index=True
    )
    attempt: Mapped[int]
    is_final: Mapped[bool]
    text: Mapped[str] = mapped_column(Text)
    verdict: Mapped[str] = mapped_column(String(16))  # passed | warnings | failed | no_sources
    report: Mapped[dict[str, Any]] = mapped_column(JSONB)
    sources: Mapped[list[dict[str, Any]]] = mapped_column(JSONB)
    provider: Mapped[str | None] = mapped_column(String(32))
    model: Mapped[str | None] = mapped_column(String(200))
    latency_ms: Mapped[int | None]
    prompt_tokens: Mapped[int | None]
    completion_tokens: Mapped[int | None]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
