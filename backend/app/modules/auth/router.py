from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select

from app.core.mail import MailerDep
from app.db.session import SessionDep
from app.modules.auth import service
from app.modules.auth.deps import CurrentUser, require_roles
from app.modules.auth.models import Role, User
from app.modules.auth.schemas import (
    EmailIn,
    EmailTokenIn,
    LoginIn,
    MessageOut,
    RefreshIn,
    RegisterIn,
    TokenPair,
    UserList,
    UserOut,
    UserUpdate,
)

auth_router = APIRouter(prefix="/auth", tags=["auth"])
users_router = APIRouter(prefix="/users", tags=["users"])


@auth_router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def register(data: RegisterIn, session: SessionDep, mailer: MailerDep):
    return await service.register(session, mailer, data)


@auth_router.post("/verify-email", response_model=UserOut)
async def verify_email(data: EmailTokenIn, session: SessionDep):
    return await service.verify_email(session, data.token)


@auth_router.post(
    "/resend-verification", response_model=MessageOut, status_code=status.HTTP_202_ACCEPTED
)
async def resend_verification(data: EmailIn, session: SessionDep, mailer: MailerDep):
    await service.resend_verification(session, mailer, data.email)
    # одинаковый ответ независимо от того, есть ли такой пользователь
    return MessageOut(detail="Если адрес зарегистрирован и не подтверждён, письмо отправлено")


@auth_router.post("/login", response_model=TokenPair)
async def login(data: LoginIn, session: SessionDep):
    user = await service.authenticate(session, data.email, data.password)
    return await service.issue_tokens(session, user)


@auth_router.post("/refresh", response_model=TokenPair)
async def refresh(data: RefreshIn, session: SessionDep):
    return await service.rotate_refresh(session, data.refresh_token)


@auth_router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(data: RefreshIn, session: SessionDep) -> None:
    await service.revoke_refresh(session, data.refresh_token)


@users_router.get("/me", response_model=UserOut)
async def me(user: CurrentUser):
    return user


@users_router.get(
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


@users_router.patch("/{user_id}", response_model=UserOut)
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
            await service.revoke_all(session, user.id)
    await session.commit()
    await session.refresh(user)
    return user


router = APIRouter()
router.include_router(auth_router)
router.include_router(users_router)
