"""История диалогов. Блокировка строки удерживается до сохранения ответа конвейером."""

from datetime import date, datetime
from uuid import UUID

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.time import utcnow
from app.modules.ai.providers.base import Message
from app.modules.assistant.models import Answer, Dialog, Question, QuestionStatus
from app.modules.assistant.schemas import QuestionOut

MAX_HISTORY_TURNS = 6
MAX_HISTORY_CHARS = 12000
MAX_PREVIOUS_ANSWER_CHARS = 2000


class DialogIn(BaseModel):
    on: date | None = Field(None, alias="date")


class DialogOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    on_date: date
    created_at: datetime
    updated_at: datetime
    expires_at: datetime


class DialogList(BaseModel):
    items: list[DialogOut]
    total: int


class DialogHistory(BaseModel):
    items: list[QuestionOut]
    total: int


async def owned(
    session: AsyncSession, dialog_id: UUID, user_id: UUID, *, lock: bool = False
) -> Dialog:
    stmt = select(Dialog).where(Dialog.id == dialog_id, Dialog.user_id == user_id)
    if lock:
        stmt = stmt.with_for_update(nowait=True)
    try:
        dialog = await session.scalar(stmt)
    except DBAPIError as exc:
        if getattr(exc.orig, "sqlstate", None) == "55P03":
            await session.rollback()
            raise HTTPException(409, "Предыдущее сообщение ещё обрабатывается") from exc
        raise
    if dialog is None:
        raise HTTPException(404, "Диалог не найден")
    return dialog


def bounded_history(pairs: list[tuple[str, str]]) -> list[Message]:
    """pairs — от старых к новым; последние целые вопросы важнее старых ответов."""
    selected = []
    remaining = MAX_HISTORY_CHARS
    for question, answer in reversed(pairs[-MAX_HISTORY_TURNS:]):
        if len(question) > remaining:
            break
        answer = answer[: min(MAX_PREVIOUS_ANSWER_CHARS, remaining - len(question))]
        selected.append((question, answer))
        remaining -= len(question) + len(answer)
    return [
        message
        for question, answer in reversed(selected)
        for message in (
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        )
        if message["content"]
    ]


async def begin_turn(
    session: AsyncSession, dialog_id: UUID, user_id: UUID, on: date | None
) -> tuple[Dialog, list[Message]]:
    dialog = await owned(session, dialog_id, user_id, lock=True)
    if dialog.expires_at <= utcnow():
        raise HTTPException(409, "Сеанс завершён; начните новый диалог")
    if on is not None and on != dialog.on_date:
        raise HTTPException(
            422, "Дата норм фиксирована для диалога; создайте новый для другой даты"
        )
    rows = (
        await session.execute(
            select(Question.text, Answer.text)
            .join(Answer, Answer.question_id == Question.id)
            .where(
                Question.dialog_id == dialog.id,
                Question.status == QuestionStatus.ANSWERED,
                Answer.is_final,
            )
            .order_by(Question.turn_no.desc())
            .limit(MAX_HISTORY_TURNS)
        )
    ).all()
    dialog.next_turn += 1
    dialog.updated_at = utcnow()
    return dialog, bounded_history(list(reversed(rows)))
