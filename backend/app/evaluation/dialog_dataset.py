"""Проверяемый формат синтетической выборки. Разделение — по семействам диалогов."""

from datetime import date
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Article(Record):
    act: str  # стабильный source:external_id, не локальный PK
    key: str


class Turn(Record):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class Case(Record):
    id: str
    family: str
    split: Literal["development", "evaluation"]
    kind: str
    history: list[Turn] = Field(min_length=1)
    question: str = Field(min_length=1, max_length=2000)
    articles: list[Article] = Field(min_length=1)
    rationale: str

    @model_validator(mode="after")
    def coherent(self) -> Self:
        if self.history[0].role != "user":
            raise ValueError("history must start with user")
        if len({(a.act, a.key) for a in self.articles}) != len(self.articles):
            raise ValueError("duplicate gold article")
        return self


class Dataset(Record):
    name: str
    on: date
    annotation: Literal["synthetic_unreviewed", "human_reviewed"]
    description: str
    cases: list[Case] = Field(min_length=1)

    @model_validator(mode="after")
    def split_integrity(self) -> Self:
        ids, families = set(), {}
        for case in self.cases:
            if case.id in ids:
                raise ValueError(f"duplicate case id: {case.id}")
            ids.add(case.id)
            if families.setdefault(case.family, case.split) != case.split:
                raise ValueError(f"family crosses splits: {case.family}")
        return self
