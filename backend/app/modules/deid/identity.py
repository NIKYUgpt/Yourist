"""Явные привязки упоминаний. Идентификаторы и псевдонимы живут только в RAM deid."""

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PersonBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: int = Field(ge=0, strict=True)
    end: int = Field(gt=0, strict=True)
    participant: str = Field(pattern=r"^(?:[A-Za-z0-9_-]{1,64}|\[ФИО_[1-9]\d*\])$")

    @model_validator(mode="after")
    def valid_range(self):
        if self.end <= self.start:
            raise ValueError("invalid mention range")
        return self


class PersonResolution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    bindings: list[PersonBinding] = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def disjoint_ranges(self):
        ranges = sorted((b.start, b.end) for b in self.bindings)
        if any(left[1] > right[0] for left, right in zip(ranges, ranges[1:], strict=False)):
            raise ValueError("overlapping mention bindings")
        return self


class IdentityRequired(Exception):
    def __init__(self, detail: dict):
        self.detail = detail
        super().__init__("Требуется уточнить участников")


class IdentityBindingError(ValueError):
    def __init__(self, message: str):
        self.detail = {"code": "invalid_person_resolution", "message": message}
        super().__init__(message)


class PersonMapping(dict[str, str]):
    """Совместимая таблица подстановок с отдельными подтверждёнными связями."""

    def __init__(self, values=(), *, actors=None, aliases=None):
        super().__init__(values)
        self.actors: dict[str, str] = dict(actors or {})
        self.aliases: dict[str, set[str]] = {k: set(v) for k, v in (aliases or {}).items()}

    def copy(self):
        return PersonMapping(self, actors=self.actors, aliases=self.aliases)

    def clear(self):
        super().clear()
        self.actors.clear()
        self.aliases.clear()

    @property
    def identity_size(self) -> int:
        return sum(len(k) + len(v) for k, v in self.actors.items()) + sum(
            len(k) + sum(map(len, v)) for k, v in self.aliases.items()
        )
