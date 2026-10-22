from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy import select

from app.core.principal import PrincipalDep, require_role
from app.core.roles import Role
from app.db.session import SessionDep
from app.modules.corpus import queries
from app.modules.corpus.client import PravoClient, SourceError
from app.modules.corpus.models import Act, Redaction, SyncRun
from app.modules.corpus.scheduler import SyncScheduler
from app.modules.corpus.schemas import (
    ActCreate,
    ActOut,
    ArticleBrief,
    ArticleOut,
    ChangeOut,
    ChangesOut,
    ContentsOut,
    PeriodOut,
    RedactionOut,
    SearchHitOut,
    SyncRunOut,
)
from app.modules.corpus.sync import CorpusSync, today_msk

router = APIRouter(prefix="/corpus", tags=["corpus"])
admin_only = Depends(require_role(Role.ADMIN))


def get_sync(request: Request) -> CorpusSync:
    return request.app.state.corpus_sync


def get_scheduler(request: Request) -> SyncScheduler:
    return request.app.state.scheduler


def get_source(request: Request) -> PravoClient:
    return request.app.state.corpus_client


async def _act(session, act_id: int) -> Act:
    act = await session.get(Act, act_id)
    if act is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Акт не найден")
    return act


def _no_redaction(on: date) -> HTTPException:
    return HTTPException(
        status.HTTP_404_NOT_FOUND, f"Нет разобранной редакции, действовавшей на {on:%d.%m.%Y}"
    )


@router.get("/acts", response_model=list[ActOut])
async def list_acts(session: SessionDep, _: PrincipalDep):
    return list(await session.scalars(select(Act).order_by(Act.id)))


@router.post(
    "/acts", response_model=ActOut, status_code=status.HTTP_201_CREATED, dependencies=[admin_only]
)
async def track_act(
    data: ActCreate,
    sync: CorpusSync = Depends(get_sync),
    scheduler: SyncScheduler = Depends(get_scheduler),
):
    try:
        act = await sync.track_act(data.external_hash)
    except SourceError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Источник: {exc}") from exc
    scheduler.trigger(act.id, reason="added")
    return act


@router.get("/search", response_model=list[SearchHitOut], dependencies=[admin_only])
async def search_source(
    q: str = Query(min_length=2, max_length=200), source: PravoClient = Depends(get_source)
):
    try:
        return await source.search(q)
    except SourceError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Источник: {exc}") from exc


@router.get("/acts/{act_id}", response_model=ActOut)
async def get_act(act_id: int, session: SessionDep, _: PrincipalDep):
    return await _act(session, act_id)


@router.post("/acts/{act_id}/sync", status_code=status.HTTP_202_ACCEPTED, dependencies=[admin_only])
async def sync_act(
    act_id: int, session: SessionDep, scheduler: SyncScheduler = Depends(get_scheduler)
) -> dict[str, str]:
    await _act(session, act_id)
    scheduler.trigger(act_id)
    return {"detail": "Синхронизация запущена"}


@router.get("/acts/{act_id}/redactions", response_model=list[RedactionOut])
async def list_redactions(act_id: int, session: SessionDep, _: PrincipalDep):
    await _act(session, act_id)
    rows = await session.scalars(
        select(Redaction).where(Redaction.act_id == act_id).order_by(Redaction.valid_from.desc())
    )
    return list(rows)


@router.get("/acts/{act_id}/articles", response_model=ContentsOut)
async def contents(
    act_id: int,
    session: SessionDep,
    _: PrincipalDep,
    on: date | None = Query(None, alias="date", description="По умолчанию — сегодня (МСК)"),
):
    await _act(session, act_id)
    on = on or today_msk()
    found = await queries.contents_on(session, act_id, on)
    if found is None:
        raise _no_redaction(on)
    red, versions = found
    return ContentsOut(
        redaction=RedactionOut.model_validate(red),
        articles=[ArticleBrief.model_validate(v) for v in versions],
    )


@router.get("/acts/{act_id}/articles/{key}", response_model=ArticleOut)
async def article(
    act_id: int,
    key: str,
    session: SessionDep,
    _: PrincipalDep,
    on: date | None = Query(None, alias="date", description="По умолчанию — сегодня (МСК)"),
):
    await _act(session, act_id)
    on = on or today_msk()
    found = await queries.article_on(session, act_id, key, on)
    if found is None:
        if await queries.redaction_on(session, act_id, on) is None:
            raise _no_redaction(on)
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"Статьи {key} нет в редакции на {on:%d.%m.%Y}"
        )
    version, red = found
    period = next(
        (
            p
            for p in await queries.article_history(session, act_id, key)
            if p.version is not None
            and p.version.id == version.id
            and p.valid_from <= on
            and (p.valid_to is None or on < p.valid_to)
        ),
        queries.Period(version=version, valid_from=red.valid_from, valid_to=None),
    )
    return ArticleOut(
        key=version.key,
        heading=version.heading,
        title=version.title,
        text=version.text,
        path=version.path,
        notes=version.notes,
        fragments=version.fragments,
        is_repealed=version.is_repealed,
        redaction=RedactionOut.model_validate(red),
        text_valid_from=period.valid_from,
        text_valid_to=period.valid_to,
    )


@router.get("/acts/{act_id}/articles/{key}/history", response_model=list[PeriodOut])
async def history(act_id: int, key: str, session: SessionDep, _: PrincipalDep):
    await _act(session, act_id)
    periods = await queries.article_history(session, act_id, key)
    if not periods:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"Статья {key} не найдена ни в одной редакции"
        )
    return [
        PeriodOut(
            valid_from=p.valid_from,
            valid_to=p.valid_to,
            present=p.version is not None,
            heading=p.version.heading if p.version else None,
            is_repealed=p.version.is_repealed if p.version else None,
            version_id=p.version.id if p.version else None,
        )
        for p in periods
    ]


@router.get("/acts/{act_id}/changes", response_model=ChangesOut)
async def changes(
    act_id: int,
    session: SessionDep,
    _: PrincipalDep,
    date_from: date = Query(alias="from"),
    date_to: date | None = Query(None, alias="to", description="По умолчанию — сегодня (МСК)"),
):
    await _act(session, act_id)
    date_to = date_to or today_msk()
    if date_from > date_to:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Дата «from» позже даты «to»")
    found = await queries.changes_between(session, act_id, date_from, date_to)
    if found is None:
        raise _no_redaction(date_from)
    items = [
        ChangeOut(
            key=i.key,
            kind=i.kind,
            heading=(i.after or i.before).heading,
            diff=i.diff,
            notes_added=i.notes_added,
        )
        for i in found.items
    ]
    count = {
        kind: sum(1 for i in items if i.kind == kind) for kind in ("added", "removed", "changed")
    }
    return ChangesOut(
        before=RedactionOut.model_validate(found.before),
        after=RedactionOut.model_validate(found.after),
        items=items,
        **count,
    )


@router.get("/sync-runs", response_model=list[SyncRunOut], dependencies=[admin_only])
async def sync_runs(
    session: SessionDep,
    act_id: int | None = None,
    limit: int = Query(20, ge=1, le=200),
):
    query = select(SyncRun).order_by(SyncRun.started_at.desc(), SyncRun.id.desc()).limit(limit)
    if act_id is not None:
        query = query.where(SyncRun.act_id == act_id)
    return list(await session.scalars(query))
