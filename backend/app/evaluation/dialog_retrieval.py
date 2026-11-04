"""Локальная оценка A/B/C на замороженных диалогах. Без генерации правовых ответов.

python -m app.evaluation.dialog_retrieval --dataset evaluation/dialogs_v1.json \
    --split evaluation --out evaluation/results/dialogs-run
"""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import random
import statistics
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import httpx
from sqlalchemy import select, text

from app.db.session import SessionLocal, engine
from app.evaluation.dialog_dataset import Dataset
from app.evaluation.retrieval_metrics import paired_bootstrap, score
from app.modules.ai.config import get_provider_settings
from app.modules.ai.profiles import Mode, Task
from app.modules.ai.registry import build_registry
from app.modules.ai.service import AIGateway, load_config
from app.modules.assistant.context import SYSTEM, contextualize, fallback_query
from app.modules.corpus.models import Act, ArticleVersion, RedactionArticle
from app.modules.corpus.queries import redaction_on
from app.modules.search.retrieval import search


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True, encoding="utf-8").strip()


async def corpus_snapshot(session, on):
    acts, snapshot = {}, []
    for act in await session.scalars(select(Act).where(Act.is_tracked).order_by(Act.id)):
        acts[act.id] = f"{act.source}:{act.external_id}"
        red = await redaction_on(session, act.id, on)
        if red is not None:
            snapshot.append(
                {
                    "act": acts[act.id],
                    "redaction": red.external_id,
                    "valid_from": red.valid_from.isoformat(),
                    "content_hash": red.content_hash,
                    "local_id": red.id,
                }
            )
    # Отпечаток текстов и векторов: одинаковый размер индекса ещё не означает тот же индекс.
    index_hash = await session.scalar(
        text("""
        SELECT md5(coalesce(string_agg(
            id::text || ':' || md5(context || text), ',' ORDER BY id), ''))
        FROM search.chunks
    """)
    )
    vector_hash = await session.scalar(
        text("""
        SELECT md5(coalesce(string_agg(chunk_id::text || ':' || space || ':' || md5(vector::text),
            ',' ORDER BY space, chunk_id), '')) FROM search.embeddings
    """)
    )
    return acts, {"redactions": snapshot, "chunks_md5": index_hash, "vectors_md5": vector_hash}


async def validate_gold(session, dataset, acts):
    reverse = {name: key for key, name in acts.items()}
    evidence = []
    for case in dataset.cases:
        for gold in case.articles:
            act_id = reverse.get(gold.act)
            red = await redaction_on(session, act_id, dataset.on) if act_id else None
            article = (
                None
                if red is None
                else await session.scalar(
                    select(ArticleVersion)
                    .join(RedactionArticle, RedactionArticle.version_id == ArticleVersion.id)
                    .where(
                        RedactionArticle.redaction_id == red.id, RedactionArticle.key == gold.key
                    )
                )
            )
            if article is None or article.is_repealed:
                raise ValueError(f"gold unavailable on date: {case.id}/{gold.act}/{gold.key}")
            evidence.append(
                {
                    "case": case.id,
                    "act": gold.act,
                    "key": gold.key,
                    "heading": article.heading,
                    "content_hash": article.content_hash,
                }
            )
    return evidence


def summarize(rows: list[dict], seed: int) -> dict:
    result = {}
    for method in ("last", "history", "rewrite"):
        selected = [r for r in rows if r["method"] == method]
        result[method] = {
            "n": len(selected),
            "recall_at_8": statistics.mean(r["score"]["recall"] for r in selected),
            "mrr_at_8": statistics.mean(r["score"]["mrr"] for r in selected),
            "median_search_ms": statistics.median(r["search_ms"] for r in selected),
            "fallbacks": sum(bool(r["context"] and r["context"]["fallback"]) for r in selected),
            "degraded_searches": sum(bool(r["notes"]) for r in selected),
        }
    for a, b in [("last", "history"), ("last", "rewrite"), ("history", "rewrite")]:
        result[f"{b}-{a}"] = {m: paired_bootstrap(rows, a, b, m, seed) for m in ("recall", "mrr")}
    return result


async def run(args) -> None:
    path = Path(args.dataset)
    raw = path.read_bytes()
    dataset = Dataset.model_validate_json(raw)
    cases = [c for c in dataset.cases if c.split == args.split]
    if not cases:
        raise ValueError("empty split")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)  # Не затирать результат предыдущего прогона.
    # Явный транспорт исключает системный HTTP_PROXY для локального стенда.
    registry = build_registry(get_provider_settings(), transport=httpx.AsyncHTTPTransport())
    report = {
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "dataset_sha256": digest(raw),
        "dataset": dataset.model_dump(mode="json"),
        "split": args.split,
        "seed": args.seed,
        "k": 8,
        "candidate_k": 24,
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "context_prompt_sha256": digest(SYSTEM.encode()),
        "retrieval_code_sha256": digest(Path("app/modules/search/retrieval.py").read_bytes()),
        "packages": {p: importlib.metadata.version(p) for p in ["sqlalchemy", "httpx", "pydantic"]},
        "rows": [],
    }
    try:
        async with SessionLocal() as session:
            # Один снимок данных; ни синхронизация, ни параллельный индексатор не меняют сравнение.
            await session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"))
            config = await load_config(session)
            for task in (Task.EMBED, Task.ANALYZE):
                mode, ref = config.resolve(task)
                if mode != Mode.LOCAL or ref.provider != "ollama":
                    raise ValueError("evaluation requires local Ollama embed/analyze profiles")
            gateway = AIGateway(registry, config)
            report["models"] = {
                task.value: config.resolve(task)[1].model_dump()
                for task in (Task.EMBED, Task.ANALYZE)
            }
            async with httpx.AsyncClient(trust_env=False) as client:
                response = await client.get(get_provider_settings().ollama_url + "/api/tags")
                response.raise_for_status()
                report["ollama_models"] = [
                    {k: item[k] for k in ("name", "digest", "size")}
                    for item in response.json()["models"]
                    if item["name"] in {m["model"] for m in report["models"].values()}
                ]
            acts, report["corpus"] = await corpus_snapshot(session, dataset.on)
            report["gold_evidence"] = await validate_gold(session, dataset, acts)
            rng = random.Random(args.seed)
            rng.shuffle(cases)
            save(out / "results.json", report)
            for case in cases:
                history = [t.model_dump() for t in case.history]
                methods = ["last", "history", "rewrite"]
                rng.shuffle(methods)
                for method in methods:
                    context = None
                    query = case.question
                    if method == "history":
                        query = fallback_query(case.question, history)
                    elif method == "rewrite":

                        async def chat(messages, **kwargs):
                            return await gateway.chat(Task.ANALYZE, messages, **kwargs)

                        context = await contextualize(chat, case.question, history)
                        query = context.query
                    # 24 кандидата сохраняются для следующего этапа. Все методы оцениваются
                    # по первым 8, без explicit-подъёма придуманных моделью номеров статей.
                    found = await search(
                        session, gateway, query, dataset.on, k=24, explicit=False, rerank=False
                    )
                    candidates = [
                        {
                            **asdict(h),
                            "act_ref": acts[h.act_id],
                            "valid_from": h.valid_from.isoformat(),
                        }
                        for h in found.hits
                    ]
                    expected = {(a.act, a.key) for a in case.articles}
                    row = {
                        "case": case.id,
                        "family": case.family,
                        "kind": case.kind,
                        "method": method,
                        "query": query,
                        "context": asdict(context) if context else None,
                        "score": score(
                            expected, [(h["act_ref"], h["key"]) for h in candidates[:8]]
                        ),
                        "search_ms": sum(found.timings_ms.values()),
                        "notes": found.notes,
                        "candidates": candidates,
                    }
                    report["rows"].append(row)
                    save(out / "results.json", report)
                    if found.notes:
                        raise RuntimeError("incomplete hybrid search; see saved notes")
                    print(case.id, method, row["score"], flush=True)
            report["summary"] = summarize(report["rows"], args.seed)
            report["status"] = "complete"
    except BaseException as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        raise
    finally:
        report["finished_at"] = datetime.now(UTC).isoformat()
        save(out / "results.json", report)
        await registry.aclose()
        await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", choices=["development", "evaluation"], required=True)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--out", required=True)
    asyncio.run(run(parser.parse_args()))
