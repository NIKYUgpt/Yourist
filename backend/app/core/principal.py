"""Проверка access-токена без обращения к БД.

Нужна сервисам, у которых нет таблицы пользователей (например, corpus):
подпись, срок и тип токена проверяются локально, роль берётся из claim.
Блокировка пользователя начинает действовать для них по истечении access-токена (15 мин).
"""

from dataclasses import dataclass
from typing import Annotated
from uuid import UUID

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.roles import Role
from app.core.security import ACCESS, decode_token

bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class Principal:
    user_id: UUID
    role: Role


async def get_principal(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> Principal:
    unauthorized = HTTPException(
        status.HTTP_401_UNAUTHORIZED,
        "Требуется авторизация",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if credentials is None:
        raise unauthorized
    try:
        payload = decode_token(credentials.credentials, ACCESS)
        return Principal(user_id=UUID(payload["sub"]), role=Role(payload["role"]))
    except (jwt.InvalidTokenError, KeyError, ValueError):
        raise unauthorized from None


PrincipalDep = Annotated[Principal, Depends(get_principal)]


def require_role(*roles: Role):
    async def dependency(principal: PrincipalDep) -> Principal:
        if principal.role not in roles:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Недостаточно прав")
        return principal

    return dependency
