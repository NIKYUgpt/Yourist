from datetime import date

from sqlalchemy import func, select

from app.modules.ai.profiles import Mode
from app.modules.ai.service import AIGateway, load_config, save_config
from app.modules.corpus.models import ArticleVersion
from app.modules.search import indexer as indexer_module
from app.modules.search.indexer import index_status
from app.modules.search.models import Chunk
from app.modules.search.retrieval import (
    explicit_article_keys,
    lexical_term,
    query_words,
    search,
)
from tests.search.conftest import BEFORE_AGENCIES, NOW


def test_explicit_article_keys():
    q = "Можно ли уволить по ст. 81 и статье 19.1, а также по ст.ст. 341.1-1 ТК?"
    assert explicit_article_keys(q) == ["81", "19.1", "341.1-1"]
    assert explicit_article_keys("Статья 58 про срочный договор") == ["58"]
    assert explicit_article_keys("сколько стоит 81 рубль") == []


def test_long_words_match_by_prefix():
    # «испытательный» и «испытания» — разные основы; по началу слова находятся обе
    assert lexical_term("испытательный") == "испыта:*"
    assert lexical_term("отпуск") == "отпуск"
    assert lexical_term("прогула") == "прогула"


def test_query_words_drop_short_and_stop_words():
    assert query_words("Можно ли меня уволить за прогул? Прогул!") == ["уволить", "прогул"]
    assert query_words("а и в") == []


async def test_index_built_once_per_version(db, indexed, indexer):
    versions = await db.scalar(select(func.count()).select_from(ArticleVersion))
    assert indexed.versions == versions
    assert indexed.chunks > versions  # длинные статьи режутся на несколько фрагментов
    assert indexed.embedded == indexed.chunks
    assert indexed.space == "ollama/bge-m3"

    again = await indexer.run()
    assert (again.versions, again.chunks, again.embedded) == (0, 0, 0)

    status = await index_status(db)
    assert status["indexed_versions"] == versions
    assert status["spaces"] == {"ollama/bge-m3": indexed.chunks}


async def test_switching_embed_model_adds_new_space(db, indexed, indexer, ai_registry):
    config = await load_config(db)
    config.mode = Mode.API
    await save_config(db, config, None)

    stats = await indexer.run()
    assert (stats.versions, stats.space, stats.embedded) == (
        0,
        "yandex/text-search",
        indexed.chunks,
    )
    yandex = ai_registry.get("yandex")
    assert {call["kind"] for call in yandex.embed_calls} == {"document"}
    # векторы старой модели остаются — переключение обратно не требует пересчёта
    assert (await index_status(db))["spaces"] == {
        "ollama/bge-m3": indexed.chunks,
        "yandex/text-search": indexed.chunks,
    }


async def test_new_chunker_version_rebuilds(db, indexed, indexer, monkeypatch):
    monkeypatch.setattr(indexer_module, "CHUNKER_VERSION", indexer_module.CHUNKER_VERSION + 1)
    stats = await indexer.run()
    assert stats.versions == indexed.versions
    assert stats.embedded == stats.chunks  # старые фрагменты удалены вместе с векторами
    assert await db.scalar(select(func.count()).select_from(Chunk)) == stats.chunks


async def test_embedding_failure_is_reported(db, synced, indexer, ai_registry):
    ai_registry.get("ollama").fail = "connection refused"
    stats = await indexer.run()
    assert stats.chunks > 0 and stats.embedded == 0
    assert "connection refused" in stats.errors[0]


async def test_search_respects_date(db, indexed, gateway):
    query = "частное агентство занятости"
    now = await search(db, gateway, query, NOW)
    assert now.hits[0].key == "341.1-1"
    assert now.hits[0].valid_from == date(2026, 9, 1)
    assert now.hits[0].act == "ТК РФ"
    assert now.hits[0].lexical_rank is not None and now.hits[0].vector_rank is not None

    before = await search(db, gateway, query, BEFORE_AGENCIES)
    assert before.redactions[0].valid_from == date(2022, 7, 14)
    assert all(h.key != "341.1-1" for h in before.hits)


async def test_search_returns_version_valid_on_date(db, indexed, gateway):
    query = "непредставления сведений о доходах конфликт интересов"
    old = await search(db, gateway, query, BEFORE_AGENCIES)
    new = await search(db, gateway, query, NOW)
    old_text = next(h.text for h in old.hits if h.key == "81" and "7.1)" in h.text)
    new_text = next(h.text for h in new.hits if h.key == "81" and "7.1)" in h.text)
    assert "непредставления или представления неполных" in old_text
    assert "непредставления или представления неполных" not in new_text


async def test_explicit_article_comes_first(db, indexed, gateway):
    result = await search(db, gateway, "Что сказано в ст. 58 про отпуск?", NOW, k=4)
    assert result.hits[0].key == "58" and result.hits[0].explicit
    assert sum(h.key == "58" for h in result.hits) <= 2


async def test_search_without_vectors_falls_back_to_lexical(db, indexed, ai_registry):
    ai_registry.get("ollama").fail = "timeout"
    gateway = AIGateway(ai_registry, await load_config(db))
    result = await search(db, gateway, "прогул без уважительных причин", NOW)
    assert any("векторный поиск недоступен" in n for n in result.notes)
    assert result.hits and result.hits[0].key == "81"
    assert all(h.vector_rank is None for h in result.hits)


async def test_partial_embeddings_are_noted(db, synced, indexer, gateway, monkeypatch):
    monkeypatch.setattr(indexer, "embed_batch", 3)
    calls = {"n": 0}
    original = gateway.registry.get("ollama").embed

    async def first_batch_only(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] > 1:
            raise indexer_module.ProviderError("ollama", "stop")
        return await original(*args, **kwargs)

    monkeypatch.setattr(gateway.registry.get("ollama"), "embed", first_batch_only)
    await indexer.run()
    monkeypatch.setattr(gateway.registry.get("ollama"), "embed", original)
    result = await search(db, gateway, "трудовой договор", NOW)
    assert any(n.startswith("векторы готовы для 3 из") for n in result.notes)


async def test_no_redaction_on_date(db, indexed, gateway):
    result = await search(db, gateway, "трудовой договор", date(2001, 1, 1))
    assert result.hits == []
    assert result.notes == ["нет разобранных редакций, действовавших на 01.01.2001"]
