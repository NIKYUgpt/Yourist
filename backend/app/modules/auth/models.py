import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, String, Uuid, func, true
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin

SCHEMA = "auth"


class Role(enum.StrEnum):
    USER = "user"
    SUPPORT = "support"
    ADMIN = "admin"


class User(TimestampMixin, Base):
    __tablename__ = "users"
    __table_args__ = {"schema": SCHEMA}

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    # email храним в нижнем регистре, поэтому обычного unique достаточно
    email: Mapped[str] = mapped_column(String(320), unique=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[Role] = mapped_column(
        Enum(
            Role,
            name="user_role",
            native_enum=False,
            create_constraint=True,
            length=16,
            values_callable=lambda roles: [r.value for r in roles],
        ),
        default=Role.USER,
        server_default=Role.USER.value,
    )
    is_active: Mapped[bool] = mapped_column(default=True, server_default=true())
    email_verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # Момент согласия на обработку ПД (ст. 9 152-ФЗ) — без него регистрация невозможна
    pd_consent_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    @property
    def is_verified(self) -> bool:
        return self.email_verified_at is not None


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"
    __table_args__ = {"schema": SCHEMA}

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey(f"{SCHEMA}.users.id", ondelete="CASCADE"), index=True
    )
    token_hash: Mapped[str] = mapped_column(String(64), unique=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
