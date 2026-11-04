"""Парное сравнение на сохранённых кандидатах history, без нового поиска или LLM."""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

from app.evaluation.dialog_retrieval import digest, git, json_digest, save
from app.evaluation.retrieval_metrics import paired_bootstrap, score
from app.modules.search.reranker import MODEL, REVISION, LocalReranker, RerankSettings, order


async def run(args):
    raw = Path(args.input).read_bytes()
    original = json.loads(raw)
    if original["status"] != "complete" or original["k"] != 8 or original["candidate_k"] != 24:
        raise ValueError("complete k8/pool24 experiment required")
    candidates = [r for r in original["rows"] if r["method"] == "history"]
    gold = {
        c["id"]: {(a["act"], a["key"]) for a in c["articles"]}
        for c in original["dataset"]["cases"]
        if c["split"] == original["split"]
    }
    if len(candidates) != len(gold) or {r["case"] for r in candidates} != set(gold):
        raise ValueError("incomplete or duplicated paired cases")
    settings = RerankSettings(
        rerank_model_path=args.model_path,
        rerank_max_length=512,
        rerank_batch_size=4,
        rerank_threads=4,
        rerank_timeout_seconds=300,
    )
    manifest = json.loads((Path(args.model_path) / "yourist-manifest.json").read_bytes())
    if manifest["model"] != MODEL or manifest["revision"] != REVISION:
        raise ValueError("model revision differs from preregistered variant")
    for name, expected in manifest["files_sha256"].items():
        file = Path(args.model_path) / name
        if file.parent.resolve() != Path(args.model_path).resolve():
            raise ValueError("invalid manifest path")
        with file.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != expected:
                raise ValueError(f"model file changed: {name}")
    ranker = LocalReranker(settings)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "input_sha256": digest(raw),
        "input_canonical_sha256": json_digest(original),
        "input": str(Path(args.input).as_posix()),
        "dataset_sha256": original["dataset_sha256"],
        "split": original["split"],
        "annotation": original["dataset"]["annotation"],
        "model": manifest,
        "settings": settings.model_dump(exclude={"rerank_model_path"}),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "reranker_code_sha256": digest(Path("app/modules/search/reranker.py").read_bytes()),
        "packages": {
            p: importlib.metadata.version(p) for p in ["torch", "transformers", "tokenizers"]
        },
        "rows": [],
    }
    try:
        started = time.perf_counter()
        await asyncio.to_thread(ranker._load)
        report["load_ms"] = round(1000 * (time.perf_counter() - started))
        for row in candidates:
            hits = row["candidates"]
            started = time.perf_counter()
            scores = await ranker.score(
                row["query"], [h["context"] + "\n" + h["text"] for h in hits]
            )
            elapsed = round(1000 * (time.perf_counter() - started))
            ranked = [hits[i] for i in order(scores, [h["explicit"] for h in hits])]
            result = score(gold[row["case"]], [(h["act_ref"], h["key"]) for h in ranked[:8]])
            baseline = score(gold[row["case"]], [(h["act_ref"], h["key"]) for h in hits[:8]])
            if baseline != row["score"]:
                raise ValueError("source scores do not match candidates")
            report["rows"].extend(
                [
                    {"case": row["case"], "method": "baseline", "score": baseline},
                    {
                        "case": row["case"],
                        "method": "reranker",
                        "score": result,
                        "latency_ms": elapsed,
                        "scores": scores,
                        "candidate_ids": [h["chunk_id"] for h in hits],
                        "ranked_ids": [h["chunk_id"] for h in ranked],
                    },
                ]
            )
            save(out / "results.json", report)
            print(row["case"], baseline, "->", result, f"{elapsed} ms", flush=True)
        summary = {}
        for method in ("baseline", "reranker"):
            rows = [r for r in report["rows"] if r["method"] == method]
            summary[method] = {
                m: statistics.mean(r["score"][m] for r in rows) for m in ("recall", "mrr")
            }
        summary["paired_difference"] = {
            m: paired_bootstrap(report["rows"], "baseline", "reranker", m, original["seed"])
            for m in ("recall", "mrr")
        }
        summary["median_rerank_ms"] = statistics.median(
            r["latency_ms"] for r in report["rows"] if r["method"] == "reranker"
        )
        report["summary"] = summary
        report["status"] = "complete"
    except BaseException as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        raise
    finally:
        report["finished_at"] = datetime.now(UTC).isoformat()
        save(out / "results.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out", required=True)
    asyncio.run(run(parser.parse_args()))
