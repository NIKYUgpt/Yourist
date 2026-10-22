import logging

from fastapi import APIRouter, Response, status
from sqlalchemy import text

from app import __version__
from app.api.deps import SessionDep

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(session: SessionDep, response: Response) -> dict[str, str]:
    db = "ok"
    try:
        await session.execute(text("SELECT 1"))
    except Exception:
        logger.exception("База данных недоступна")
        db = "unavailable"
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "ok" if db == "ok" else "degraded", "db": db, "version": __version__}
