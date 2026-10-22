import json
from collections import Counter
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from app.modules.corpus.client import ActCard, SearchHit, SourceError, SourceRedaction
from app.modules.corpus.sync import CorpusSync

FIXTURES = Path(__file__).parent / "fixtures"
TODAY = date(2026, 10, 22)
DOC_HASH = "tk-hash"


def load_fixture(name: str) -> tuple[str, list[dict]]:
    html = (FIXTURES / f"{name}.html").read_text("utf-8")
    toc = json.loads((FIXTURES / f"{name}.toc.json").read_text("utf-8"))
    return html, toc


class FakeSource:
    """Источник в памяти: две реальные (урезанные) редакции ТК РФ."""

    def __init__(self) -> None:
        self.calls: Counter[str] = Counter()
        self.failing: set[str] = set()
        self.fail_list = False
        self.items: dict[str, tuple[SourceRedaction, str, list[dict]]] = {}
        for redid, valid_from, name in (
            ("401591", date(2022, 7, 14), "tk_2022_07_14"),
            ("492286", date(2026, 9, 1), "tk_2026_09_01"),
        ):
            html, toc = load_fixture(name)
            red = SourceRedaction(
                external_id=redid,
                valid_from=valid_from,
                source_timestamp=f"{valid_from.isoformat()}T10:00:00",
                caption=f"с {valid_from:%d.%m.%Y}",
                is_actual=valid_from == date(2026, 9, 1),
                is_official=True,
                has_content=True,
            )
            self.items[redid] = (red, html, toc)

    def edit(self, redid: str, old: str, new: str) -> None:
        """Источник «пересобрал» редакцию: меняем текст и метку времени."""
        red, html, toc = self.items[redid]
        assert old in html
        self.items[redid] = (
            replace(red, source_timestamp="regenerated"),
            html.replace(old, new),
            toc,
        )

    async def search(self, query: str, limit: int = 20) -> list[SearchHit]:
        self.calls["search"] += 1
        return [
            SearchHit(
                external_id="64284",
                external_hash="03bdee8f44a71247d7ba3342484326dbc3fa292d3caae9f88ada4b1ee38c20d9",
                title="Трудовой кодекс Российской Федерации",
                passing="Кодекс Российской Федерации от 30.12.2001 № 197-ФЗ",
                state="Действует с изменениями",
            )
        ]

    async def card(self, doc_hash: str) -> ActCard:
        self.calls["card"] += 1
        return ActCard(
            external_id="64284",
            external_hash=doc_hash,
            title="Трудовой кодекс Российской Федерации",
            passing="Кодекс Российской Федерации от 30.12.2001 № 197-ФЗ",
            state="Действует с изменениями",
            kind="Кодекс",
            number="197-ФЗ",
            adopted_on=date(2001, 12, 30),
        )

    async def redactions(self, doc_hash: str) -> list[SourceRedaction]:
        self.calls["redactions"] += 1
        if self.fail_list:
            raise SourceError("redactions: источник недоступен")
        return [red for red, _, _ in self.items.values()]

    async def redaction_html(self, redaction_id: str) -> str:
        self.calls[f"html:{redaction_id}"] += 1
        if redaction_id in self.failing:
            raise SourceError("redtext: HTTP 500")
        return self.items[redaction_id][1]

    async def redaction_toc(self, redaction_id: str) -> list[dict]:
        self.calls[f"toc:{redaction_id}"] += 1
        return self.items[redaction_id][2]


@pytest.fixture
def source() -> FakeSource:
    return FakeSource()


@pytest.fixture
def frozen_today(monkeypatch):
    for target in ("app.modules.corpus.sync.today_msk", "app.modules.corpus.router.today_msk"):
        monkeypatch.setattr(target, lambda: TODAY)
    return TODAY


@pytest.fixture
async def corpus(engine, source, frozen_today) -> CorpusSync:
    return CorpusSync(engine, source)
