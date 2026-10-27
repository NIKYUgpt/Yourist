from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field

from app.modules.corpus.models import RedactionStatus, SyncStatus


class ActOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    short_title: str | None
    kind: str | None
    number: str | None
    adopted_on: date | None
    source_state: str | None
    external_hash: str
    is_tracked: bool
    last_checked_at: datetime | None
    last_changed_at: datetime | None


class ActCreate(BaseModel):
    external_hash: str = Field(pattern=r"^[0-9a-f]{64}$", description="dochash из поиска источника")


class SearchHitOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    external_id: str
    external_hash: str
    title: str
    passing: str
    state: str


class RedactionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    external_id: str
    valid_from: date
    caption: str
    is_official: bool
    status: RedactionStatus
    articles_count: int
    error: str | None
    fetched_at: datetime | None
    parsed_at: datetime | None


class ArticleBrief(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    key: str
    heading: str
    path: list[str]
    is_repealed: bool


class ContentsOut(BaseModel):
    redaction: RedactionOut
    articles: list[ArticleBrief]


class FragmentOut(BaseModel):
    ref: list[list[str]] = Field(description='Например [["часть", "1"], ["пункт", "6"]]')
    text: str


class ArticleOut(BaseModel):
    key: str
    heading: str
    title: str
    text: str
    path: list[str]
    notes: list[str]
    fragments: list[FragmentOut]
    is_repealed: bool
    redaction: RedactionOut
    # период, в течение которого действует именно этот текст статьи
    text_valid_from: date
    text_valid_to: date | None


class PeriodOut(BaseModel):
    valid_from: date
    valid_to: date | None
    present: bool = Field(description="False — в этот период статьи не было в кодексе")
    heading: str | None
    is_repealed: bool | None
    version_id: int | None


class ChangeOut(BaseModel):
    key: str
    kind: str = Field(description="added | removed | changed")
    heading: str
    diff: list[str] = Field(description="Построчная разница в формате unified diff")
    notes_added: list[str]


class ChangesOut(BaseModel):
    before: RedactionOut
    after: RedactionOut
    added: int
    removed: int
    changed: int
    items: list[ChangeOut]


class SyncRunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    act_id: int
    trigger: str
    status: SyncStatus
    stats: dict[str, int]
    error: str | None
    started_at: datetime
    finished_at: datetime | None
