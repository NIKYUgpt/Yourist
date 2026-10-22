from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from app.models import Role


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    email: str
    role: Role
    is_active: bool
    is_verified: bool
    created_at: datetime


class UserUpdate(BaseModel):
    role: Role | None = None
    is_active: bool | None = None


class UserList(BaseModel):
    items: list[UserOut]
    total: int
