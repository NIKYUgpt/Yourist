"""Импортирует модели всех модулей, чтобы Alembic и тесты видели полную metadata."""

from app.db.base import Base
from app.modules.auth import models as auth_models

__all__ = ["Base", "auth_models"]
