import uuid
from datetime import date
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from sqlalchemy import delete, func, select

from app.db.session import SessionDep
from app.modules.ai.profiles import Task
from app.modules.ai.providers.base import Timer
from app.modules.ai.service import AIDep
from app.modules.assistant.models import Answer, Question, QuestionSource
from app.modules.assistant.pipeline import AnswerFailed, Outcome, Pipeline
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
from app.modules.deid.api import DeidClient, DeidUnavailable, MediaError
from app.modules.deid.service import summary

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
        verification=report.get("verification"),
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
    return _outcome_out(outcome, data.text)


def _outcome_out(outcome: Outcome, original: str | None = None) -> QuestionOut:
    q = outcome.question
    # пользователю, задавшему вопрос, — с его данными; в БД и в истории — с метками
    return QuestionOut(
        id=q.id,
        created_at=q.created_at,
        on_date=q.on_date,
        source=q.source,
        # для текста — как написал пользователь; для голоса и фото — распознанный текст
        # (одно лицо в разных падежах восстановится в одной форме)
        text=original if original is not None else outcome.restore(q.text),
        status=q.status,
        pii=q.pii,
        answer=_answer_out(outcome.answer, outcome.restore),
        trace=q.trace,
    )


async def _media_question(pipeline: Pipeline, user_id, recognize, source, on) -> QuestionOut:
    timer = Timer()
    try:
        media = await recognize
        trace = {
            "recognize": {"ms": timer.ms, **media.recognized},
            "deid": {"found": summary(media.deid.entities), "llm": media.deid.llm},
        }
        outcome = await pipeline.answer(user_id, media.deid, on or today_msk(), source, trace)
    except MediaError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    except DeidUnavailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except AnswerFailed as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Модель не ответила: {exc}") from exc
    return _outcome_out(outcome)


@router.post("/questions/voice", response_model=QuestionOut, status_code=status.HTTP_201_CREATED)
async def ask_voice(
    user: CurrentUser,
    session: SessionDep,
    ai: AIDep,
    deid: DeidDep,
    file: UploadFile = File(..., description="Голосовое сообщение: OGG/Opus, M4A, MP3, WAV"),
    on: date | None = Form(None, alias="date"),
):
    """Голосовой вопрос: распознаётся и обезличивается в сервисе deid, дальше — как текст."""
    pipeline = Pipeline(session, ai, deid)
    stt = ai.resolve(Task.STT)[1]
    audio = await file.read()
    recognize = deid.from_audio(audio, file.filename or "audio", stt, pipeline.deid_model())
    return await _media_question(pipeline, user.id, recognize, QuestionSource.VOICE, on)


@router.post("/questions/photo", response_model=QuestionOut, status_code=status.HTTP_201_CREATED)
async def ask_photo(
    user: CurrentUser,
    session: SessionDep,
    ai: AIDep,
    deid: DeidDep,
    file: UploadFile = File(..., description="Фото документа: JPEG или PNG"),
    text: str = Form("", max_length=4000, description="Вопрос к документу; можно не задавать"),
    on: date | None = Form(None, alias="date"),
):
    """Вопрос к фото документа (приказа, договора): текст распознаётся и обезличивается
    в сервисе deid и добавляется к вопросу."""
    pipeline = Pipeline(session, ai, deid)
    ocr = ai.resolve(Task.OCR)[1]
    image = await file.read()
    recognize = deid.from_image(
        image, file.filename or "photo", file.content_type or "", text, ocr, pipeline.deid_model()
    )
    return await _media_question(pipeline, user.id, recognize, QuestionSource.PHOTO, on)


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
            source=q.source,
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
        source=q.source,
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
