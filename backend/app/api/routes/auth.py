from fastapi import APIRouter, status

from app.api.deps import MailerDep, SessionDep
from app.schemas.auth import (
    EmailIn,
    EmailTokenIn,
    LoginIn,
    MessageOut,
    RefreshIn,
    RegisterIn,
    TokenPair,
)
from app.schemas.user import UserOut
from app.services import auth as service

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def register(data: RegisterIn, session: SessionDep, mailer: MailerDep):
    return await service.register(session, mailer, data)


@router.post("/verify-email", response_model=UserOut)
async def verify_email(data: EmailTokenIn, session: SessionDep):
    return await service.verify_email(session, data.token)


@router.post(
    "/resend-verification", response_model=MessageOut, status_code=status.HTTP_202_ACCEPTED
)
async def resend_verification(data: EmailIn, session: SessionDep, mailer: MailerDep):
    await service.resend_verification(session, mailer, data.email)
    # одинаковый ответ независимо от того, есть ли такой пользователь
    return MessageOut(detail="Если адрес зарегистрирован и не подтверждён, письмо отправлено")


@router.post("/login", response_model=TokenPair)
async def login(data: LoginIn, session: SessionDep):
    user = await service.authenticate(session, data.email, data.password)
    return await service.issue_tokens(session, user)


@router.post("/refresh", response_model=TokenPair)
async def refresh(data: RefreshIn, session: SessionDep):
    return await service.rotate_refresh(session, data.refresh_token)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(data: RefreshIn, session: SessionDep) -> None:
    await service.revoke_refresh(session, data.refresh_token)
