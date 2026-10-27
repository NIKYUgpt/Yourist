"""Настройки нейросетей (схема ai). Ключи API здесь не хранятся — только в окружении."""

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import CheckConstraint, DateTime, SmallInteger, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base

SCHEMA = "ai"


class AISettings(Base):
    """Одна строка: текущая конфигурация (profiles.AIConfig)."""

    __tablename__ = "settings"
    __table_args__ = (CheckConstraint("id = 1", name="single_row"), {"schema": SCHEMA})

    id: Mapped[int] = mapped_column(SmallInteger, primary_key=True, default=1)
    config: Mapped[dict[str, Any]] = mapped_column(JSONB)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    # кто менял — для журнала; без внешнего ключа, чтобы не связывать схемы модулей
    updated_by: Mapped[UUID | None]
