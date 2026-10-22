from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select

from app.api.deps import CurrentUser, SessionDep, require_roles
from app.models import Role, User
from app.schemas.user import UserList, UserOut, UserUpdate
from app.services.auth import revoke_all

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/me", response_model=UserOut)
async def me(user: CurrentUser):
    return user


@router.get(
    "", response_model=UserList, dependencies=[Depends(require_roles(Role.ADMIN, Role.SUPPORT))]
)
async def list_users(
    session: SessionDep,
    email: str | None = Query(None, description="Поиск по части email"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    query = select(User)
    if email:
        query = query.where(User.email.contains(email.lower(), autoescape=True))
    total = await session.scalar(select(func.count()).select_from(query.subquery()))
    rows = await session.scalars(query.order_by(User.created_at.desc()).limit(limit).offset(offset))
    return UserList(items=[UserOut.model_validate(u) for u in rows], total=total or 0)


@router.patch("/{user_id}", response_model=UserOut)
async def update_user(
    user_id: UUID,
    data: UserUpdate,
    session: SessionDep,
    admin: User = Depends(require_roles(Role.ADMIN)),
):
    user = await session.get(User, user_id)
    if user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Пользователь не найден")
    # защита от ситуации, когда в системе не остаётся ни одного админа
    if user.id == admin.id and (data.role not in (None, Role.ADMIN) or data.is_active is False):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Нельзя снять с себя права администратора или заблокировать себя",
        )

    if data.role is not None:
        user.role = data.role
    if data.is_active is not None:
        user.is_active = data.is_active
        if not data.is_active:
            await revoke_all(session, user.id)
    await session.commit()
    await session.refresh(user)
    return user
