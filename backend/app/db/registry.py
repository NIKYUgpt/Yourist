"""Импортирует модели всех модулей, чтобы Alembic и тесты видели полную metadata."""

from app.db.base import Base
from app.modules.auth import models as auth_models
from app.modules.corpus import models as corpus_models

__all__ = ["Base", "auth_models", "corpus_models"]
