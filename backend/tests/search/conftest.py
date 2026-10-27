from datetime import date

import pytest

from app.modules.ai.service import AIGateway, load_config
from app.modules.search.indexer import Indexer
from tests.corpus.conftest import corpus, frozen_today, source  # noqa: F401

DOC_HASH = "tk-hash"


@pytest.fixture
async def synced(corpus):  # noqa: F811
    act = await corpus.track_act(DOC_HASH)
    await corpus.sync_act(act.id)
    return act


@pytest.fixture
def indexer(engine, ai_registry) -> Indexer:
    return Indexer(engine, ai_registry, embed_batch=8)


@pytest.fixture
async def indexed(synced, indexer):
    stats = await indexer.run()
    assert stats is not None and not stats.errors
    return stats


@pytest.fixture
async def gateway(db, ai_registry) -> AIGateway:
    return AIGateway(ai_registry, await load_config(db))


BEFORE_AGENCIES = date(2023, 1, 1)  # ст. 341.1-1 ещё нет (её нет в редакции 2022 года)
NOW = date(2026, 10, 1)
