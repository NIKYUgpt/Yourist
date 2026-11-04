import uuid
from contextlib import suppress
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
from app.modules.assistant.dialogs import (
    DialogHistory,
    DialogIn,
    DialogList,
    DialogOut,
    begin_turn,
    owned,
)
from app.modules.assistant.models import Answer, Dialog, Question, QuestionSource
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


async def _pipeline(session, ai, deid, user, dialog_id, on):
    if dialog_id is None:
        return Pipeline(session, ai, deid), on or today_msk()
    dialog, history = await begin_turn(session, dialog_id, user.id, on)
    return Pipeline(session, ai, deid, dialog=dialog, history=history), dialog.on_date


@router.post("/dialogs", response_model=DialogOut, status_code=201)
async def create_dialog(data: DialogIn, user: CurrentUser, session: SessionDep, deid: DeidDep):
    info = await deid.create_session(user.id)
    dialog = Dialog(
        user_id=user.id,
        deid_session_id=info.id,
        expires_at=info.expires_at,
        on_date=data.on or today_msk(),
    )
    try:
        session.add(dialog)
        await session.commit()
    except Exception:
        await session.rollback()
        with suppress(DeidUnavailable):
            await deid.delete_session(info.id, user.id)
        raise
    return dialog


@router.get("/dialogs", response_model=DialogList)
async def list_dialogs(
    user: CurrentUser,
    session: SessionDep,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    where = Dialog.user_id == user.id
    total = await session.scalar(select(func.count()).select_from(Dialog).where(where))
    rows = await session.scalars(
        select(Dialog)
        .where(where)
        .order_by(Dialog.updated_at.desc(), Dialog.id)
        .limit(limit)
        .offset(offset)
    )
    return DialogList(items=[DialogOut.model_validate(d) for d in rows], total=total)


@router.get("/dialogs/{dialog_id}", response_model=DialogOut)
async def get_dialog(dialog_id: uuid.UUID, user: CurrentUser, session: SessionDep):
    return await owned(session, dialog_id, user.id)


@router.get("/dialogs/{dialog_id}/messages", response_model=DialogHistory)
async def dialog_messages(
    dialog_id: uuid.UUID,
    user: CurrentUser,
    session: SessionDep,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    await owned(session, dialog_id, user.id)
    where = Question.dialog_id == dialog_id
    total = await session.scalar(select(func.count()).select_from(Question).where(where))
    rows = list(
        await session.scalars(
            select(Question).where(where).order_by(Question.turn_no).limit(limit).offset(offset)
        )
    )
    finals = await _final_answers(session, [q.id for q in rows])
    return DialogHistory(items=[_stored_question(q, finals.get(q.id)) for q in rows], total=total)


@router.delete("/dialogs/{dialog_id}", status_code=204)
async def delete_dialog(
    dialog_id: uuid.UUID, user: CurrentUser, session: SessionDep, deid: DeidDep
):
    dialog = await owned(session, dialog_id, user.id, lock=True)
    # При недоступном deid не сообщаем об успешном удалении исходных значений.
    await deid.delete_session(dialog.deid_session_id, user.id)
    await session.delete(dialog)
    await session.commit()


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
    pipeline, on = await _pipeline(session, ai, deid, user, data.dialog_id, data.on)
    try:
        outcome = await pipeline.ask(user.id, data.text, on)
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
        dialog_id=q.dialog_id,
        turn_no=q.turn_no,
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
    dialog_id: uuid.UUID | None = Form(None),
):
    """Голосовой вопрос: распознаётся и обезличивается в сервисе deid, дальше — как текст."""
    pipeline, on = await _pipeline(session, ai, deid, user, dialog_id, on)
    stt = ai.resolve(Task.STT)[1]
    audio = await file.read()
    recognize = deid.from_audio(
        audio, file.filename or "audio", stt, pipeline.deid_model(), **pipeline.deid_context()
    )
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
    dialog_id: uuid.UUID | None = Form(None),
):
    """Вопрос к фото документа (приказа, договора): текст распознаётся и обезличивается
    в сервисе deid и добавляется к вопросу."""
    pipeline, on = await _pipeline(session, ai, deid, user, dialog_id, on)
    ocr = ai.resolve(Task.OCR)[1]
    image = await file.read()
    recognize = deid.from_image(
        image,
        file.filename or "photo",
        file.content_type or "",
        text,
        ocr,
        pipeline.deid_model(),
        **pipeline.deid_context(),
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
            dialog_id=q.dialog_id,
            turn_no=q.turn_no,
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
    return _stored_question(q, final, staff=user.role in STAFF)


def _stored_question(q: Question, final: Answer | None, *, staff: bool = False) -> QuestionOut:
    return QuestionOut(
        id=q.id,
        dialog_id=q.dialog_id,
        turn_no=q.turn_no,
        created_at=q.created_at,
        on_date=q.on_date,
        source=q.source,
        text=q.text,
        status=q.status,
        pii=q.pii,
        answer=_answer_out(final) if final else None,
        trace=q.trace if staff else None,
    )


@router.delete("/questions/{question_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_question(question_id: uuid.UUID, user: CurrentUser, session: SessionDep):
    """Удалить свой вопрос вместе с ответами."""
    question = await session.get(Question, question_id)
    if question is None or question.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Вопрос не найден")
    if question.dialog_id is not None:
        raise HTTPException(409, "Сообщение входит в диалог; удалите диалог целиком")
    await session.execute(delete(Question).where(Question.id == question_id))
    await session.commit()
