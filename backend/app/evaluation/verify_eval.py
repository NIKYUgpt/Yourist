"""Оценка проверяющей модели на верных и испорченных ответах.

    python -m app.evaluation.verify_eval --results evaluation/results/<run>/results.json \\
        --base ollama/qwen3:8b --verifiers ollama/qwen3:8b yandex/yandexgpt-5.1 --out ...

Из результатов сравнения берутся ответы модели --base, в которых найдены все
эталонные факты. Для каждого такой ответ показывается проверяющей модели как есть
и с подменённым фактом (corrupt.py). Хорошая проверяющая модель принимает верный
ответ (supported) и отклоняет испорченный (partially или unsupported).
Дополнительно проверяются настоящие ошибки моделей из того же прогона — ответы,
в которых не найден ни один эталонный факт (непустые и не обрезанные по лимиту).
"""

import argparse
import asyncio
import json
import logging
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.evaluation.compare import SOURCES_K, _gateway, parse_model
from app.evaluation.corrupt import corrupt
from app.evaluation.metrics import load_questions
from app.modules.ai.config import get_provider_settings
from app.modules.ai.profiles import DEFAULT_CONFIG, Mode, ModelRef
from app.modules.ai.registry import Registry, build_registry
from app.modules.assistant.verifier import SUPPORTED, UNKNOWN, verify
from app.modules.search.retrieval import search

logger = logging.getLogger(__name__)


def _chat(registry: Registry, ref: ModelRef):
    async def chat(messages, **kwargs):
        return await registry.get(ref.provider).chat(
            ref.model, messages, options=ref.options, **kwargs
        )

    return chat


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    good = [r for r in rows if r["kind"] == "original"]
    bad = [r for r in rows if r["kind"] == "corrupted"]
    real = [r for r in rows if r["kind"] == "real_error"]

    def share(items, pred) -> float:
        return sum(1 for r in items if pred(r)) / len(items) if items else 0.0

    latencies = sorted(r["latency_ms"] for r in rows if r["latency_ms"] is not None)
    return {
        "pairs": len(bad),
        # испорченный ответ не принят как полностью подтверждённый
        "detected": share(bad, lambda r: r["verdict"] not in (SUPPORTED, UNKNOWN)),
        "detected_unsupported": share(bad, lambda r: r["verdict"] == "unsupported"),
        "real_errors": len(real),
        "real_detected": share(real, lambda r: r["verdict"] not in (SUPPORTED, UNKNOWN)),
        # верный ответ отклонён
        "false_alarm": share(good, lambda r: r["verdict"] not in (SUPPORTED, UNKNOWN)),
        "unknown": share(rows, lambda r: r["verdict"] == UNKNOWN),
        "latency_ms_median": latencies[len(latencies) // 2] if latencies else None,
    }


def render(data: dict[str, Any]) -> str:
    lines = [
        "# Проверяющая модель: верные и испорченные ответы",
        "",
        f"Ответы модели `{data['base']}` из прогона {data['source_run']}; пар (верный / "
        f"с подменённым фактом): {data['pairs']}. Подмены — правдоподобные: другое число "
        "или противоположное утверждение в месте эталонного факта.",
        "",
        "| Проверяющая модель | Поймала подмену | Из них «не подтверждается» "
        "| Поймала настоящие ошибки | Ложные тревоги на верных | Не разобран ответ "
        "| Медиана, с |",
        "|---|---|---|---|---|---|---|",
    ]
    for model, s in data["summary"].items():
        latency = f"{s['latency_ms_median'] / 1000:.1f}" if s["latency_ms_median"] else "—"
        real = (
            f"{round(s['real_detected'] * s['real_errors'])} из {s['real_errors']}"
            if s["real_errors"]
            else "—"
        )
        lines.append(
            f"| `{model}` | {s['detected'] * 100:.0f}% | {s['detected_unsupported'] * 100:.0f}% "
            f"| {real} | {s['false_alarm'] * 100:.0f}% | {s['unknown'] * 100:.0f}% | {latency} |"
        )
    lines += ["", "## Подмены", ""]
    for p in data["pair_list"]:
        lines.append(f"- {p['question']}: {p['change']}")
    if data["real_errors"]:
        lines += ["", "## Настоящие ошибки моделей", ""]
        verdicts = {
            (r["question"], r["source_model"], r["verifier"]): r["verdict"]
            for r in data["rows"]
            if r["kind"] == "real_error"
        }
        for e in data["real_errors"]:
            judged = ", ".join(
                f"`{v}`: {verdicts.get((e['question'], e['model'], v), '—')}"
                for v in data["summary"]
            )
            lines.append(f"- {e['question']} (`{e['model']}`): {judged}")
    lines.append("")
    return "\n".join(lines)


async def main(args: argparse.Namespace) -> None:
    results = json.loads(Path(args.results).read_text("utf-8"))
    questions = {q.id: q for q in load_questions({"questions": results["dataset_questions"]})}
    on = date.fromisoformat(results["on"])
    pairs = []
    for run in results["runs"]:
        q = questions[run["question"]]
        if run["model"] != args.base or q.off_topic or not run["score"].get("correct"):
            continue
        broken = corrupt(run["text"], q.facts)
        if broken:
            pairs.append(
                {
                    "question": q.id,
                    "original": run["text"],
                    "corrupted": broken[0],
                    "change": broken[1],
                }
            )
    if args.limit:
        pairs = pairs[: args.limit]
    real_errors = [
        {"question": run["question"], "model": run["model"], "text": run["text"]}
        for run in results["runs"]
        if not questions[run["question"]].off_topic
        and run["text"].strip()
        and run["score"].get("fact_share") == 0
        and run.get("finish_reason") not in ("length", "max_tokens")
    ]

    engine = create_async_engine(get_settings().database_url)
    registry = build_registry(get_provider_settings())
    rows: list[dict[str, Any]] = []
    try:
        async with async_sessionmaker(engine)() as session:
            gateway = _gateway(registry, DEFAULT_CONFIG, Mode.LOCAL)
            sources = {
                p["question"]: (
                    await search(session, gateway, questions[p["question"]].text, on, k=SOURCES_K)
                ).hits
                for p in [*pairs, *real_errors]
            }
            for spec in args.verifiers:
                ref = parse_model(spec)
                for p in pairs:
                    for kind in ("original", "corrupted"):
                        v = await verify(
                            _chat(registry, ref),
                            questions[p["question"]].text,
                            sources[p["question"]],
                            p[kind],
                            on,
                        )
                        rows.append(
                            {"verifier": spec, "question": p["question"], "kind": kind, **asdict(v)}
                        )
                        logger.info("%s %s %s: %s", spec, p["question"], kind, v.verdict)
                for e in real_errors:
                    q = questions[e["question"]]
                    v = await verify(_chat(registry, ref), q.text, sources[q.id], e["text"], on)
                    rows.append(
                        {
                            "verifier": spec,
                            "question": q.id,
                            "kind": "real_error",
                            "source_model": e["model"],
                            **asdict(v),
                        }
                    )
                    logger.info("%s %s real error (%s): %s", spec, q.id, e["model"], v.verdict)
    finally:
        await registry.aclose()
        await engine.dispose()

    data = {
        "source_run": results["started_at"][:16].replace("T", " "),
        "base": args.base,
        "pairs": len(pairs),
        "pair_list": pairs,
        "real_errors": real_errors,
        "summary": {s: summarize([r for r in rows if r["verifier"] == s]) for s in args.verifiers},
        "rows": rows,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "verify.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), "utf-8")
    (out / "verify.md").write_text(render(data), "utf-8")
    print(f"Готово: {out / 'verify.md'}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--verifiers", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    asyncio.run(main(parser.parse_args()))
