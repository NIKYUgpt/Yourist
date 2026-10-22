import hashlib
import secrets
from datetime import timedelta
from typing import Any
from uuid import UUID

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

from app.core.config import get_settings
from app.core.time import utcnow

JWT_ALGORITHM = "HS256"

# Argon2id с параметрами по умолчанию из argon2-cffi
_hasher = PasswordHasher()
_dummy_hash: str | None = None


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    try:
        return _hasher.verify(password_hash, password)
    except (VerificationError, InvalidHashError):
        return False


def password_needs_rehash(password_hash: str) -> bool:
    return _hasher.check_needs_rehash(password_hash)


def burn_password_check(password: str) -> None:
    """Проверка пароля «вхолостую» для несуществующего email.

    Без неё по времени ответа можно понять, зарегистрирован ли адрес.
    """
    global _dummy_hash
    if _dummy_hash is None:
        _dummy_hash = _hasher.hash(secrets.token_urlsafe(16))
    verify_password(_dummy_hash, password)


def create_token(subject: UUID, purpose: str, ttl: timedelta, **claims: Any) -> str:
    now = utcnow()
    payload = {"sub": str(subject), "typ": purpose, "iat": now, "exp": now + ttl, **claims}
    return jwt.encode(payload, get_settings().secret_key, algorithm=JWT_ALGORITHM)


def decode_token(token: str, purpose: str) -> dict[str, Any]:
    """Проверяет подпись, срок и назначение токена. Бросает jwt.InvalidTokenError."""
    payload = jwt.decode(token, get_settings().secret_key, algorithms=[JWT_ALGORITHM])
    if payload.get("typ") != purpose:
        raise jwt.InvalidTokenError("unexpected token type")
    return payload


def new_refresh_token() -> tuple[str, str]:
    """Возвращает (токен для клиента, его хеш для БД). Сам токен в БД не хранится."""
    token = secrets.token_urlsafe(48)
    return token, hash_refresh_token(token)


def hash_refresh_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()
