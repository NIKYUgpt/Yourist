from dataclasses import replace
from datetime import date

from sqlalchemy import func, select, text

from app.modules.corpus import queries
from app.modules.corpus.models import (
    ArticleVersion,
    Redaction,
    RedactionArticle,
    RedactionSource,
    RedactionStatus,
    SyncStatus,
)
from tests.corpus.conftest import DOC_HASH

OLD, NEW = "401591", "492286"


async def _synced(corpus):
    act = await corpus.track_act(DOC_HASH)
    run = await corpus.sync_act(act.id)
    return act, run


async def test_initial_sync_deduplicates_unchanged_articles(corpus, db):
    act, run = await _synced(corpus)

    assert run.status == SyncStatus.SUCCESS
    assert run.stats == {
        "new": 2, "parsed": 2, "updated": 0, "unchanged": 0, "withdrawn": 0, "failed": 0
    }  # fmt: skip
    # 1, 7, 56.1, 58 не менялись — по одной версии; у 81 две; 341.1-1 появилась одна
    versions = dict(
        (
            await db.execute(
                select(ArticleVersion.key, func.count())
                .where(ArticleVersion.act_id == act.id)
                .group_by(ArticleVersion.key)
            )
        ).all()
    )
    assert versions == {"1": 1, "7": 1, "56.1": 1, "58": 1, "81": 2, "341.1-1": 1}
    counts = dict(
        (
            await db.execute(
                select(Redaction.external_id, Redaction.articles_count).where(
                    Redaction.act_id == act.id
                )
            )
        ).all()
    )
    assert counts == {OLD: 5, NEW: 6}
    assert await db.scalar(select(func.count()).select_from(RedactionSource)) == 2


async def test_article_on_date_picks_redaction_in_force(corpus, db):
    act, _ = await _synced(corpus)
    marker = "в связи с назначением меры уголовно-правового характера в виде судебного штрафа"

    old_version, old_red = await queries.article_on(db, act.id, "81", date(2023, 1, 1))
    new_version, new_red = await queries.article_on(db, act.id, "81", date(2026, 10, 1))
    assert (old_red.external_id, new_red.external_id) == (OLD, NEW)
    assert marker not in old_version.text
    assert marker in new_version.text

    # до первой редакции в корпусе данных нет, статьи 341.1-1 в 2023 году не было
    assert await queries.article_on(db, act.id, "81", date(2022, 1, 1)) is None
    assert await queries.article_on(db, act.id, "341.1-1", date(2023, 1, 1)) is None


async def test_contents_keep_document_order(corpus, db):
    act, _ = await _synced(corpus)
    red, versions = await queries.contents_on(db, act.id, date(2026, 10, 1))
    assert red.external_id == NEW
    assert [v.key for v in versions] == ["1", "7", "56.1", "58", "81", "341.1-1"]


async def test_article_history_periods(corpus, db):
    act, _ = await _synced(corpus)

    periods = await queries.article_history(db, act.id, "81")
    assert [(p.valid_from, p.valid_to) for p in periods] == [
        (date(2022, 7, 14), date(2026, 9, 1)),
        (date(2026, 9, 1), None),
    ]
    assert periods[0].version.id != periods[1].version.id

    [single] = await queries.article_history(db, act.id, "1")
    assert (single.valid_from, single.valid_to) == (date(2022, 7, 14), None)

    [added] = await queries.article_history(db, act.id, "341.1-1")
    assert added.valid_from == date(2026, 9, 1)


async def test_changes_between_dates(corpus, db):
    act, _ = await _synced(corpus)
    changes = await queries.changes_between(db, act.id, date(2023, 1, 1), date(2026, 10, 1))

    assert {(i.key, i.kind) for i in changes.items} == {("81", "changed"), ("341.1-1", "added")}
    [art81] = [i for i in changes.items if i.key == "81"]
    assert any(line.startswith("+г) совершения по месту работы хищения") for line in art81.diff)
    assert any(line.startswith("-г) совершения по месту работы хищения") for line in art81.diff)
    assert all(line.startswith(("@@", "+", "-")) for line in art81.diff)

    same = await queries.changes_between(db, act.id, date(2026, 9, 1), date(2026, 10, 1))
    assert same.items == []


async def test_repeated_sync_only_rechecks_actual_redaction(corpus, source):
    act, _ = await _synced(corpus)
    source.calls.clear()

    run = await corpus.sync_act(act.id, trigger="schedule")

    assert run.stats["unchanged"] == 1
    assert run.stats["parsed"] == run.stats["updated"] == run.stats["new"] == 0
    # старая редакция не перекачивается, актуальная — проверяется по хешу
    assert source.calls == {"redactions": 1, f"html:{NEW}": 1, f"toc:{NEW}": 1}


async def test_regenerated_redaction_creates_new_version(corpus, source, db):
    act, _ = await _synced(corpus)
    source.edit(
        NEW, "Целями трудового законодательства", "Главными целями трудового законодательства"
    )

    run = await corpus.sync_act(act.id)

    assert run.stats["updated"] == 1
    periods = await queries.article_history(db, act.id, "1")
    assert [p.valid_from for p in periods] == [date(2022, 7, 14), date(2026, 9, 1)]
    assert periods[1].version.text.startswith("Главными целями")
    # сопоставление редакции пересобрано, а не дописано
    assert (
        await db.scalar(
            select(func.count())
            .select_from(RedactionArticle)
            .join(Redaction)
            .where(Redaction.external_id == NEW)
        )
        == 6
    )


async def test_redaction_removed_from_source_is_withdrawn(corpus, source, db):
    act, _ = await _synced(corpus)
    del source.items[NEW]

    run = await corpus.sync_act(act.id)

    assert run.stats["withdrawn"] == 1
    red = await queries.redaction_on(db, act.id, date(2026, 10, 1))
    assert red.external_id == OLD


async def test_failed_redaction_is_recorded_and_retried(corpus, source, db):
    source.failing.add(OLD)
    act, run = await _synced(corpus)

    assert run.status == SyncStatus.SUCCESS
    assert (run.stats["parsed"], run.stats["failed"]) == (1, 1)
    failed = await db.scalar(select(Redaction).where(Redaction.external_id == OLD))
    assert failed.status == RedactionStatus.FAILED
    assert "HTTP 500" in failed.error

    failed_id = failed.id
    source.failing.clear()
    retry = await corpus.sync_act(act.id)
    assert retry.stats["parsed"] == 1
    db.expire_all()
    assert (await db.get(Redaction, failed_id)).status == RedactionStatus.PARSED


async def test_new_timestamp_with_same_content_is_not_reparsed(corpus, source, db):
    act, _ = await _synced(corpus)
    red, html, toc = source.items[OLD]
    source.items[OLD] = (replace(red, source_timestamp="touched"), html, toc)

    run = await corpus.sync_act(act.id)

    # старую редакцию перекачали из-за новой метки, но хеш тот же — разбор не нужен
    assert run.stats["unchanged"] == 2
    assert run.stats["updated"] == run.stats["parsed"] == 0
    status = await db.scalar(select(Redaction.status).where(Redaction.external_id == OLD))
    assert status == RedactionStatus.PARSED


async def test_source_outage_marks_run_failed(corpus, source):
    act = await corpus.track_act(DOC_HASH)
    source.fail_list = True

    run = await corpus.sync_act(act.id)

    assert run.status == SyncStatus.FAILED
    assert "источник недоступен" in run.error


async def test_concurrent_sync_is_skipped(corpus, engine):
    act = await corpus.track_act(DOC_HASH)
    async with engine.connect() as conn:
        await conn.execute(text("SELECT pg_advisory_lock(7001, :id)"), {"id": act.id})
        assert await corpus.sync_act(act.id) is None
        await conn.execute(text("SELECT pg_advisory_unlock(7001, :id)"), {"id": act.id})
    assert await corpus.sync_act(act.id) is not None


async def test_track_act_is_idempotent(corpus, db):
    first = await corpus.track_act(DOC_HASH)
    second = await corpus.track_act(DOC_HASH)
    assert first.id == second.id
    assert (first.number, first.adopted_on) == ("197-ФЗ", date(2001, 12, 30))
