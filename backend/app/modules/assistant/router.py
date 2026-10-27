import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy import delete, func, select

from app.db.session import SessionDep
from app.modules.ai.service import AIDep
from app.modules.assistant.models import Answer, Question
from app.modules.assistant.pipeline import AnswerFailed, Pipeline
from app.modules.assistant.schemas import (
    AnswerOut,
    QuestionBrief,
    QuestionIn,
    QuestionList,
    QuestionOut,
)
from app.modules.auth.deps import CurrentUser
from app.modules.auth.models import Role
from app.modules.corpus.sync import today_msk
from app.modules.deid.api import DeidClient, DeidUnavailable

router = APIRouter(prefix="/assistant", tags=["assistant"])
STAFF = (Role.ADMIN, Role.SUPPORT)


def get_deid(request: Request) -> DeidClient:
    return request.app.state.deid_client


DeidDep = Annotated[DeidClient, Depends(get_deid)]


def _answer_out(answer: Answer, restore=lambda text: text) -> AnswerOut:
    report = answer.report
    return AnswerOut(
        text=restore(answer.text),
        verdict=answer.verdict,
        attempt=answer.attempt,
        citations=report.get("citations", []),
        markers=report.get("markers", []),
        issues=report.get("issues", []),
        sources=answer.sources,
        provider=answer.provider,
        model=answer.model,
        latency_ms=answer.latency_ms,
    )


@router.post("/questions", response_model=QuestionOut, status_code=status.HTTP_201_CREATED)
async def ask(data: QuestionIn, user: CurrentUser, session: SessionDep, ai: AIDep, deid: DeidDep):
    """Задать вопрос. Ответ приходит целиком (локальная модель — до минуты)."""
    pipeline = Pipeline(session, ai, deid)
    try:
        outcome = await pipeline.ask(user.id, data.text, data.on or today_msk())
    except DeidUnavailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except AnswerFailed as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Модель не ответила: {exc}") from exc
    q = outcome.question
    # пользователю, задавшему вопрос, — с его данными; в БД и в истории — с метками
    return QuestionOut(
        id=q.id,
        created_at=q.created_at,
        on_date=q.on_date,
        text=data.text,
        status=q.status,
        pii=q.pii,
        answer=_answer_out(outcome.answer, outcome.restore),
        trace=q.trace,
    )


async def _final_answers(session, ids: list[uuid.UUID]) -> dict[uuid.UUID, Answer]:
    rows = await session.scalars(select(Answer).where(Answer.question_id.in_(ids), Answer.is_final))
    return {a.question_id: a for a in rows}


@router.get("/questions", response_model=QuestionList)
async def history(
    user: CurrentUser,
    session: SessionDep,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    where = Question.user_id == user.id
    total = await session.scalar(select(func.count()).select_from(Question).where(where))
    questions = list(
        await session.scalars(
            select(Question)
            .where(where)
            .order_by(Question.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
    )
    finals = await _final_answers(session, [q.id for q in questions])
    items = [
        QuestionBrief(
            id=q.id,
            created_at=q.created_at,
            on_date=q.on_date,
            text=q.text,
            status=q.status,
            verdict=finals[q.id].verdict if q.id in finals else None,
        )
        for q in questions
    ]
    return QuestionList(items=items, total=total)


async def _visible(session, question_id: uuid.UUID, user) -> Question:
    question = await session.get(Question, question_id)
    # чужие вопросы видят только администратор и поддержка (в обезличенном виде)
    if question is None or (question.user_id != user.id and user.role not in STAFF):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Вопрос не найден")
    return question


@router.get("/questions/{question_id}", response_model=QuestionOut)
async def get_question(question_id: uuid.UUID, user: CurrentUser, session: SessionDep):
    q = await _visible(session, question_id, user)
    final = (await _final_answers(session, [q.id])).get(q.id)
    return QuestionOut(
        id=q.id,
        created_at=q.created_at,
        on_date=q.on_date,
        text=q.text,
        status=q.status,
        pii=q.pii,
        answer=_answer_out(final) if final else None,
        trace=q.trace if user.role in STAFF else None,
    )


@router.delete("/questions/{question_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_question(question_id: uuid.UUID, user: CurrentUser, session: SessionDep):
    """Удалить свой вопрос вместе с ответами."""
    question = await session.get(Question, question_id)
    if question is None or question.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Вопрос не найден")
    await session.execute(delete(Question).where(Question.id == question_id))
    await session.commit()
