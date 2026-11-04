import asyncio
import threading
from dataclasses import replace

import pytest

from app.modules.ai.profiles import with_defaults
from app.modules.ai.service import AIGateway, load_config
from app.modules.search import reranker as module
from app.modules.search.reranker import LocalReranker, RerankSettings, RerankUnavailable, order
from app.modules.search.retrieval import search, search_issues
from tests.search.conftest import NOW


def test_order_preserves_explicit_and_stable_ties():
    assert order([1, 9, 9, -5], [False, False, False, True]) == [3, 1, 2, 0]
    with pytest.raises(RerankUnavailable, match="invalid_scores"):
        order([float("nan")], [False])
    assert with_defaults({}).rerank is False
    assert with_defaults({"rerank": True}).rerank is True


@pytest.mark.parametrize("values", [[1], [float("inf"), 2], [float("nan"), 2]])
async def test_invalid_predictions_rejected(monkeypatch, values):
    ranker = LocalReranker(RerankSettings())
    monkeypatch.setattr(ranker, "_predict", lambda q, p: values)
    with pytest.raises(RerankUnavailable, match="invalid_scores"):
        await ranker.score("query", ["one", "two"])
    assert not ranker._lock.locked()


async def test_missing_model_has_no_download():
    ranker = LocalReranker(RerankSettings(rerank_model_path=""))
    with pytest.raises(RerankUnavailable, match="model_not_prepared"):
        await ranker.score("query", ["document"])


async def test_timeout_keeps_busy_until_worker_finishes(monkeypatch):
    release = threading.Event()
    ranker = LocalReranker(RerankSettings(rerank_timeout_seconds=0.02))

    def slow(query, passages):
        release.wait(timeout=2)
        return [1.0]

    monkeypatch.setattr(ranker, "_predict", slow)
    try:
        with pytest.raises(RerankUnavailable, match="timeout"):
            await ranker.score("q", ["p"])
        with pytest.raises(RerankUnavailable, match="busy"):
            await ranker.score("q", ["p"])
    finally:
        release.set()
    for _ in range(100):
        if not ranker._lock.locked():
            break
        await asyncio.sleep(0.01)
    assert await ranker.score("q", ["p"]) == [1.0]


async def test_disabled_and_failure_preserve_baseline(db, indexed, ai_registry, monkeypatch):
    config = await load_config(db)
    gateway = AIGateway(ai_registry, config)
    baseline = await search(db, gateway, "увольнение за прогул", NOW, k=5)
    called = []

    async def fail(query, hits, k, **kwargs):
        called.append(query)
        raise RerankUnavailable("model_not_prepared")

    monkeypatch.setattr(module, "rerank_hits", fail)
    disabled = await search(db, gateway, "увольнение за прогул", NOW, k=5)
    assert not called
    config.rerank = True
    fallback = await search(db, gateway, "увольнение за прогул", NOW, k=5)
    assert [h.chunk_id for h in baseline.hits] == [h.chunk_id for h in fallback.hits]
    assert disabled.hits == baseline.hits
    assert "rerank_fallback:model_not_prepared" in fallback.notes


async def test_real_search_pool_keeps_date_and_explicit_refs(db, indexed, ai_registry, monkeypatch):
    gateway = AIGateway(ai_registry, await load_config(db))
    gateway.config.rerank = True
    ranker = LocalReranker(RerankSettings())
    monkeypatch.setattr(ranker, "_predict", lambda q, p: list(range(len(p))))
    monkeypatch.setattr(module, "get_reranker", lambda: ranker)
    result = await search(db, gateway, "ст. 81 прогул", NOW, k=4)
    assert len(result.hits) <= 4
    assert result.hits[0].explicit and result.hits[0].key == "81"
    assert all(h.valid_from <= NOW and h.rerank_score is not None for h in result.hits)
    assert len({h.chunk_id for h in result.hits}) == len(result.hits)


async def test_compound_links_follow_reordered_sources(db, indexed, ai_registry, monkeypatch):
    gateway = AIGateway(ai_registry, await load_config(db))
    queries = ["увольнение за прогул", "беременная женщина"]
    original, links = await search_issues(db, gateway, "трудовой договор", queries, NOW)
    expected = [{original.hits[n - 1].chunk_id for n in numbers} for numbers in links]
    calls = []

    async def reverse(query, hits, k, **kwargs):
        calls.append(query)
        return [replace(h, rerank_score=float(i)) for i, h in enumerate(reversed(hits))]

    monkeypatch.setattr(module, "rerank_hits", reverse)
    gateway.config.rerank = True
    actual, reordered = await search_issues(db, gateway, "трудовой договор", queries, NOW)
    assert len(calls) == 1
    assert actual.hits[0].chunk_id == original.hits[-1].chunk_id
    assert [{actual.hits[n - 1].chunk_id for n in numbers} for numbers in reordered] == expected
