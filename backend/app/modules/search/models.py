"""Поисковый индекс по корпусу (схема search).

Индекс строится по уникальным версиям статей (corpus.article_versions), а не по
редакциям: одинаковый текст в 30 редакциях режется и векторизуется один раз.
Фильтр по дате — через сопоставление редакции (corpus.redaction_articles).
"""

from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Computed,
    DateTime,
    ForeignKey,
    Identity,
    Index,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base

SCHEMA = "search"


class IndexedVersion(Base):
    """Версия статьи нарезана на фрагменты (в том числе на ноль — утратившая силу)."""

    __tablename__ = "indexed_versions"
    __table_args__ = {"schema": SCHEMA}

    version_id: Mapped[int] = mapped_column(
        ForeignKey("corpus.article_versions.id", ondelete="CASCADE"), primary_key=True
    )
    chunker_version: Mapped[int]
    chunks_count: Mapped[int]
    indexed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("version_id", "ord"),
        Index("ix_chunks_tsv", "tsv", postgresql_using="gin"),
        {"schema": SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    version_id: Mapped[int] = mapped_column(
        ForeignKey("corpus.article_versions.id", ondelete="CASCADE"), index=True
    )
    ord: Mapped[int]
    # «ТК РФ — Глава 13 … — Статья 81. …»: участвует и в векторе, и в полнотекстовом поиске
    context: Mapped[str] = mapped_column(Text)
    text: Mapped[str] = mapped_column(Text)
    refs: Mapped[list[Any]] = mapped_column(JSONB)
    tsv: Mapped[str] = mapped_column(
        TSVECTOR, Computed("to_tsvector('russian', context || ' ' || text)", persisted=True)
    )


class Embedding(Base):
    """Вектор фрагмента в пространстве конкретной модели (space = «ollama/bge-m3»).

    Размерность не фиксирована: у моделей она разная, а векторы разных моделей
    не сравниваются — поиск всегда идёт внутри одного space.
    """

    __tablename__ = "embeddings"
    __table_args__ = (Index("ix_embeddings_space", "space"), {"schema": SCHEMA})

    chunk_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA}.chunks.id", ondelete="CASCADE"), primary_key=True
    )
    space: Mapped[str] = mapped_column(String(200), primary_key=True)
    vector: Mapped[list[float]] = mapped_column(Vector())
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
