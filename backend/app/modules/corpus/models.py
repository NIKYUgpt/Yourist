"""Таблицы корпуса (схема corpus).

Статьи хранятся без повторов: одинаковый текст в разных редакциях — одна строка
в article_versions. Редакция — это сопоставление «номер статьи → версия»
(redaction_articles). Отсюда простые запросы: текст на дату, история статьи,
разница между редакциями.
"""

import enum
from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Date,
    DateTime,
    Enum,
    ForeignKey,
    Identity,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    true,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin

SCHEMA = "corpus"
SOURCE_PRAVO_ACTUAL = "pravo_actual"


def _enum(cls: type[enum.Enum], name: str) -> Enum:
    return Enum(
        cls,
        name=name,
        native_enum=False,
        create_constraint=True,
        length=16,
        values_callable=lambda items: [i.value for i in items],
    )


class RedactionStatus(enum.StrEnum):
    PENDING = "pending"  # нужно скачать и разобрать
    PARSED = "parsed"
    FAILED = "failed"
    WITHDRAWN = "withdrawn"  # исчезла из источника


class SyncStatus(enum.StrEnum):
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class Act(TimestampMixin, Base):
    __tablename__ = "acts"
    __table_args__ = (UniqueConstraint("source", "external_id"), {"schema": SCHEMA})

    id: Mapped[int] = mapped_column(Integer, Identity(), primary_key=True)
    source: Mapped[str] = mapped_column(String(32), default=SOURCE_PRAVO_ACTUAL)
    external_id: Mapped[str] = mapped_column(String(64))
    # идентификатор документа, по которому источник отдаёт редакции
    external_hash: Mapped[str] = mapped_column(String(64))
    title: Mapped[str] = mapped_column(Text)
    short_title: Mapped[str | None] = mapped_column(String(100))
    kind: Mapped[str | None] = mapped_column(String(100))
    number: Mapped[str | None] = mapped_column(String(50))
    adopted_on: Mapped[date | None] = mapped_column(Date)
    source_state: Mapped[str | None] = mapped_column(String(200))
    is_tracked: Mapped[bool] = mapped_column(default=True, server_default=true())
    last_checked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_changed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Redaction(TimestampMixin, Base):
    __tablename__ = "redactions"
    __table_args__ = (
        UniqueConstraint("act_id", "external_id"),
        Index("ix_redactions_act_id_valid_from", "act_id", "valid_from"),
        {"schema": SCHEMA},
    )

    id: Mapped[int] = mapped_column(Integer, Identity(), primary_key=True)
    act_id: Mapped[int] = mapped_column(ForeignKey(f"{SCHEMA}.acts.id", ondelete="CASCADE"))
    external_id: Mapped[str] = mapped_column(String(32))
    valid_from: Mapped[date] = mapped_column(Date)
    source_timestamp: Mapped[str | None] = mapped_column(String(32))
    caption: Mapped[str] = mapped_column(Text, default="")
    is_official: Mapped[bool] = mapped_column(default=True)
    status: Mapped[RedactionStatus] = mapped_column(
        _enum(RedactionStatus, "redaction_status"), default=RedactionStatus.PENDING
    )
    error: Mapped[str | None] = mapped_column(Text)
    # sha256 от HTML и оглавления — по нему видно, что источник пересобрал редакцию
    content_hash: Mapped[str | None] = mapped_column(String(64))
    articles_count: Mapped[int] = mapped_column(default=0, server_default="0")
    warnings: Mapped[list[str]] = mapped_column(JSONB, default=list, server_default="[]")
    fetched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    parsed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class RedactionSource(Base):
    """Сырые данные источника — чтобы пересобрать корпус без повторного скачивания."""

    __tablename__ = "redaction_sources"
    __table_args__ = {"schema": SCHEMA}

    redaction_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA}.redactions.id", ondelete="CASCADE"), primary_key=True
    )
    html: Mapped[str] = mapped_column(Text)
    toc: Mapped[list[dict[str, Any]]] = mapped_column(JSONB)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ArticleVersion(Base):
    __tablename__ = "article_versions"
    __table_args__ = (
        UniqueConstraint("act_id", "key", "content_hash"),
        {"schema": SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    act_id: Mapped[int] = mapped_column(ForeignKey(f"{SCHEMA}.acts.id", ondelete="CASCADE"))
    key: Mapped[str] = mapped_column(String(32))
    heading: Mapped[str] = mapped_column(Text)
    title: Mapped[str] = mapped_column(Text)
    text: Mapped[str] = mapped_column(Text)
    path: Mapped[list[str]] = mapped_column(JSONB)
    fragments: Mapped[list[dict[str, Any]]] = mapped_column(JSONB)
    notes: Mapped[list[str]] = mapped_column(JSONB)
    is_repealed: Mapped[bool]
    content_hash: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class RedactionArticle(Base):
    """Какая версия статьи действует в редакции."""

    __tablename__ = "redaction_articles"
    __table_args__ = {"schema": SCHEMA}

    redaction_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA}.redactions.id", ondelete="CASCADE"), primary_key=True
    )
    key: Mapped[str] = mapped_column(String(32), primary_key=True)
    version_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA}.article_versions.id", ondelete="RESTRICT"), index=True
    )
    position: Mapped[int]


class SyncRun(Base):
    __tablename__ = "sync_runs"
    __table_args__ = {"schema": SCHEMA}

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    act_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA}.acts.id", ondelete="CASCADE"), index=True
    )
    trigger: Mapped[str] = mapped_column(String(16))
    status: Mapped[SyncStatus] = mapped_column(
        _enum(SyncStatus, "sync_status"), default=SyncStatus.RUNNING
    )
    stats: Mapped[dict[str, int]] = mapped_column(JSONB, default=dict, server_default="{}")
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
