import logging
from datetime import timedelta
from uuid import UUID

import jwt
from fastapi import HTTPException, status
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.mail import Mailer
from app.core.security import (
    burn_password_check,
    create_token,
    decode_token,
    hash_password,
    hash_refresh_token,
    new_refresh_token,
    password_needs_rehash,
    verify_password,
)
from app.core.time import utcnow
from app.modules.auth.models import RefreshToken, User
from app.modules.auth.schemas import RegisterIn, TokenPair

logger = logging.getLogger(__name__)

ACCESS = "access"
VERIFY_EMAIL = "verify_email"

INVALID_CREDENTIALS = "Неверный email или пароль"
INVALID_REFRESH = "Сессия недействительна, войдите заново"


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(status.HTTP_401_UNAUTHORIZED, detail)


async def register(session: AsyncSession, mailer: Mailer, data: RegisterIn) -> User:
    email = data.email.lower()
    if await session.scalar(select(User.id).where(User.email == email)):
        raise HTTPException(status.HTTP_409_CONFLICT, "Пользователь с таким email уже существует")

    user = User(email=email, password_hash=hash_password(data.password), pd_consent_at=utcnow())
    session.add(user)
    try:
        await session.commit()
    except IntegrityError:
        # параллельная регистрация с тем же email
        await session.rollback()
        raise HTTPException(
            status.HTTP_409_CONFLICT, "Пользователь с таким email уже существует"
        ) from None
    await session.refresh(user)
    await send_verification(mailer, user)
    return user


async def send_verification(mailer: Mailer, user: User) -> None:
    settings = get_settings()
    token = create_token(
        user.id, VERIFY_EMAIL, timedelta(hours=settings.email_token_ttl_hours), email=user.email
    )
    link = f"{settings.frontend_url}/verify-email?token={token}"
    body = (
        "Здравствуйте!\n\n"
        f"Чтобы подтвердить email в Yourist, перейдите по ссылке:\n{link}\n\n"
        f"Ссылка действует {settings.email_token_ttl_hours} ч. "
        "Если вы не регистрировались, просто проигнорируйте письмо."
    )
    try:
        await mailer.send(user.email, "Подтверждение email — Yourist", body)
    except Exception:
        # регистрация не должна падать из-за почты: письмо можно запросить повторно
        logger.exception("Не удалось отправить письмо подтверждения на %s", user.email)


async def verify_email(session: AsyncSession, token: str) -> User:
    invalid = HTTPException(
        status.HTTP_400_BAD_REQUEST, "Ссылка подтверждения недействительна или устарела"
    )
    try:
        payload = decode_token(token, VERIFY_EMAIL)
        user_id = UUID(payload["sub"])
    except (jwt.InvalidTokenError, KeyError, ValueError):
        raise invalid from None

    user = await session.get(User, user_id)
    # если email сменили после отправки письма — старая ссылка не должна работать
    if user is None or user.email != payload.get("email"):
        raise invalid
    if user.email_verified_at is None:
        user.email_verified_at = utcnow()
        await session.commit()
        await session.refresh(user)
    return user


async def resend_verification(session: AsyncSession, mailer: Mailer, email: str) -> None:
    user = await session.scalar(select(User).where(User.email == email.lower()))
    if user is not None and user.is_active and not user.is_verified:
        await send_verification(mailer, user)


async def authenticate(session: AsyncSession, email: str, password: str) -> User:
    user = await session.scalar(select(User).where(User.email == email.lower()))
    if user is None:
        burn_password_check(password)
        raise _unauthorized(INVALID_CREDENTIALS)
    if not verify_password(user.password_hash, password):
        raise _unauthorized(INVALID_CREDENTIALS)
    if not user.is_active:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Учётная запись заблокирована")
    if get_settings().require_verified_email and not user.is_verified:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Сначала подтвердите email")
    if password_needs_rehash(user.password_hash):
        user.password_hash = hash_password(password)
    return user


async def issue_tokens(session: AsyncSession, user: User) -> TokenPair:
    settings = get_settings()
    raw, token_hash = new_refresh_token()
    session.add(
        RefreshToken(
            user_id=user.id,
            token_hash=token_hash,
            expires_at=utcnow() + timedelta(days=settings.refresh_token_ttl_days),
        )
    )
    await session.commit()
    access = create_token(user.id, ACCESS, timedelta(minutes=settings.access_token_ttl_minutes))
    return TokenPair(
        access_token=access,
        refresh_token=raw,
        expires_in=settings.access_token_ttl_minutes * 60,
    )


async def rotate_refresh(session: AsyncSession, raw_token: str) -> TokenPair:
    """Меняет refresh-токен на новую пару. Старый токен после этого недействителен."""
    token = await session.scalar(
        select(RefreshToken)
        .where(RefreshToken.token_hash == hash_refresh_token(raw_token))
        .with_for_update()
    )
    now = utcnow()
    if token is None:
        raise _unauthorized(INVALID_REFRESH)
    if token.revoked_at is not None:
        # повторное использование отозванного токена — похоже на кражу,
        # поэтому закрываем все сессии пользователя
        await revoke_all(session, token.user_id)
        await session.commit()
        raise _unauthorized(INVALID_REFRESH)
    if token.expires_at <= now:
        raise _unauthorized(INVALID_REFRESH)

    user = await session.get(User, token.user_id)
    if user is None or not user.is_active:
        raise _unauthorized(INVALID_REFRESH)

    token.revoked_at = now
    return await issue_tokens(session, user)


async def revoke_refresh(session: AsyncSession, raw_token: str) -> None:
    await session.execute(
        update(RefreshToken)
        .where(
            RefreshToken.token_hash == hash_refresh_token(raw_token),
            RefreshToken.revoked_at.is_(None),
        )
        .values(revoked_at=utcnow())
    )
    await session.commit()


async def revoke_all(session: AsyncSession, user_id: UUID) -> None:
    await session.execute(
        update(RefreshToken)
        .where(RefreshToken.user_id == user_id, RefreshToken.revoked_at.is_(None))
        .values(revoked_at=utcnow())
    )
