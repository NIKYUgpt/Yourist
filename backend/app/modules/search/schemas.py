from datetime import date

from pydantic import BaseModel, Field


class HitOut(BaseModel):
    act: str
    key: str
    heading: str
    refs: list[list[list[str]]] = Field(description="Части и пункты статьи в этом фрагменте")
    text: str
    valid_from: date = Field(description="С какой даты действует редакция, из которой фрагмент")
    score: float
    lexical_rank: int | None
    vector_rank: int | None
    explicit: bool = Field(description="Статья прямо названа в запросе")


class RedactionUsedOut(BaseModel):
    act: str
    redaction_id: int
    valid_from: date


class SearchOut(BaseModel):
    query: str
    on: date
    space: str | None = Field(description="Модель эмбеддингов, по векторам которой искали")
    notes: list[str]
    redactions: list[RedactionUsedOut]
    hits: list[HitOut]
    timings_ms: dict[str, int]


class IndexStatusOut(BaseModel):
    versions: int
    indexed_versions: int
    chunks: int
    spaces: dict[str, int] = Field(description="Сколько фрагментов векторизовано каждой моделью")
    active_space: str
    last_run: dict | None
