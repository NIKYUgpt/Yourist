"""Импортирует модели всех модулей, чтобы Alembic и тесты видели полную metadata."""

from app.db.base import Base
from app.modules.ai import models as ai_models
from app.modules.auth import models as auth_models
from app.modules.corpus import models as corpus_models

__all__ = ["Base", "ai_models", "auth_models", "corpus_models"]
