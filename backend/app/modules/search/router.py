from dataclasses import asdict
from datetime import date

from fastapi import APIRouter, Depends, Query, Request, status

from app.core.principal import require_role
from app.core.roles import Role
from app.db.session import SessionDep
from app.modules.ai.service import AIDep, AIGateway, get_registry, load_config
from app.modules.auth.deps import CurrentUser
from app.modules.corpus.sync import today_msk
from app.modules.search.indexer import IndexWorker, index_status
from app.modules.search.retrieval import search
from app.modules.search.schemas import IndexStatusOut, SearchOut

# поиск по нормам — в основном API
router = APIRouter(prefix="/search", tags=["search"])
# управление индексом — в сервисе корпуса, где индекс и строится
index_router = APIRouter(prefix="/search/index", tags=["search"])
admin_only = Depends(require_role(Role.ADMIN))


@router.get("", response_model=SearchOut)
async def search_norms(
    session: SessionDep,
    ai: AIDep,
    _: CurrentUser,
    q: str = Query(min_length=3, max_length=2000),
    on: date | None = Query(None, alias="date", description="По умолчанию — сегодня (МСК)"),
    k: int = Query(8, ge=1, le=30),
):
    result = await search(session, ai, q, on or today_msk(), k=k)
    return asdict(result)


def get_worker(request: Request) -> IndexWorker:
    return request.app.state.index_worker


@index_router.get("", response_model=IndexStatusOut, dependencies=[admin_only])
async def get_index_status(
    request: Request, session: SessionDep, worker: IndexWorker = Depends(get_worker)
):
    gateway = AIGateway(get_registry(request), await load_config(session))
    last = asdict(worker.last) if worker.last else None
    return {
        **await index_status(session),
        "active_space": gateway.embedding_key,
        "last_run": last,
    }


@index_router.post("", status_code=status.HTTP_202_ACCEPTED, dependencies=[admin_only])
async def run_index(worker: IndexWorker = Depends(get_worker)) -> dict[str, str]:
    worker.trigger()
    return {"detail": "Обновление индекса запущено"}
