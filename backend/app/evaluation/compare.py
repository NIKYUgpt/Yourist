"""Сравнение моделей ответа и способов поиска на наборе вопросов.

    python -m app.evaluation.compare --dataset evaluation/labor_basic.json \\
        --models ollama/qwen3:8b yandex/yandexgpt-5.1 --out evaluation/results/run

Для честного сравнения моделей всё, кроме модели ответа, одинаково: источники
(поиск выполняется один раз на вопрос), промпт, температура, проверка ссылок.
Оценивается первая попытка ответа — без исправления по итогам проверки.
"""

import argparse
import asyncio
import json
import logging
import re
import statistics
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.evaluation.metrics import (
    Question,
    load_questions,
    retrieval_rank,
    score_answer,
)
from app.modules.ai.config import get_provider_settings
from app.modules.ai.profiles import DEFAULT_CONFIG, Mode, ModelRef, Task
from app.modules.ai.providers.base import ProviderError, Timer
from app.modules.ai.registry import Registry, build_registry
from app.modules.ai.service import AIGateway, load_config
from app.modules.assistant import prompts
from app.modules.assistant.citations import check
from app.modules.search.retrieval import Hit, search

logger = logging.getLogger(__name__)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
SOURCES_K = 8


@dataclass
class Run:
    question: str
    model: str
    text: str = ""
    error: str | None = None
    latency_ms: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    verdict: str | None = None
    issues: list[str] = field(default_factory=list)
    score: dict[str, Any] = field(default_factory=dict)


def parse_model(spec: str) -> ModelRef:
    """«ollama/qwen3:8b», «yandex/qwen3.6-35b-a3b?reasoning_effort=none».

    После «?» — параметры запроса (значения разбираются как JSON, иначе строка).
    У Ollama режим размышлений по умолчанию выключен, как в рабочем конвейере.
    """
    spec, _, query = spec.partition("?")
    provider, _, model = spec.partition("/")
    options: dict[str, Any] = {"think": False} if provider == "ollama" else {}
    for pair in filter(None, query.split("&")):
        key, _, value = pair.partition("=")
        try:
            options[key] = json.loads(value)
        except json.JSONDecodeError:
            options[key] = value
    return ModelRef(provider=provider, model=model, options=options)


def label(ref: ModelRef) -> str:
    extra = {k: v for k, v in ref.options.items() if not (k == "think" and v is False)}
    suffix = "?" + "&".join(f"{k}={v}" for k, v in extra.items()) if extra else ""
    return f"{ref.provider}/{ref.model}{suffix}"


def _gateway(registry: Registry, config, embed_mode: Mode | None) -> AIGateway | None:
    if embed_mode is None:
        return None  # только полнотекстовый поиск
    config = config.model_copy(deep=True)
    config.tasks[Task.EMBED].mode = embed_mode
    return AIGateway(registry, config)


async def compare_retrieval(
    session: AsyncSession, registry: Registry, questions: list[Question], on: date
) -> dict[str, Any]:
    config = await load_config(session)
    methods = {
        "полнотекстовый": None,
        "гибридный, локальные векторы": Mode.LOCAL,
        "гибридный, векторы API": Mode.API,
    }
    out: dict[str, Any] = {}
    for name, mode in methods.items():
        gateway = _gateway(registry, config, mode)
        ranks, notes = {}, set()
        for q in questions:
            if q.off_topic:
                continue
            found = await search(session, gateway, q.text, on, k=SOURCES_K)
            ranks[q.id] = retrieval_rank(q, found.hits)
            notes.update(found.notes)
        hit = [r for r in ranks.values() if r is not None]
        out[name] = {
            "space": gateway.embedding_key if gateway else None,
            "recall": len(hit) / len(ranks),
            "mrr": sum(1 / r for r in hit) / len(ranks),
            "ranks": ranks,
            "notes": sorted(notes),
        }
    return out


async def answer_once(
    session: AsyncSession,
    registry: Registry,
    ref: ModelRef,
    q: Question,
    sources: list[Hit],
    on: date,
) -> Run:
    run = Run(question=q.id, model=label(ref))
    messages = prompts.build_messages(q.text, sources, on)
    timer = Timer()
    try:
        result = await registry.get(ref.provider).chat(
            ref.model, messages, temperature=0.1, max_tokens=2000, options=ref.options
        )
    except ProviderError as exc:
        run.error, run.latency_ms = str(exc), timer.ms
        return run
    run.text = _THINK.sub("", result.text).strip()
    run.latency_ms = result.latency_ms
    run.prompt_tokens = result.usage.prompt_tokens
    run.completion_tokens = result.usage.completion_tokens
    run.finish_reason = result.finish_reason
    report = await check(session, run.text, sources, on)
    run.verdict = report.verdict
    run.issues = [f"{i.severity}: {i.message}" for i in report.issues]
    score = score_answer(q, run.text, report, sources)
    run.score = {**asdict(score), "fact_share": score.fact_share, "correct": score.correct}
    return run


def summarize(questions: list[Question], runs: list[Run]) -> dict[str, Any]:
    by_id = {q.id: q for q in questions}
    on_topic = [r for r in runs if not by_id[r.question].off_topic]
    off_topic = [r for r in runs if by_id[r.question].off_topic]
    ok = [r for r in on_topic if r.error is None]
    latencies = [r.latency_ms for r in runs if r.error is None and r.latency_ms is not None]

    def share(items: list[Run], pred) -> float:
        return sum(1 for r in items if pred(r)) / len(items) if items else 0.0

    def median(values: list[int | None]) -> float | None:
        clean = [v for v in values if v is not None]
        return statistics.median(clean) if clean else None

    return {
        "answered": len(ok),
        "errors": len(on_topic) - len(ok),
        "correct": share(on_topic, lambda r: r.score.get("correct", False)),
        "fact_share": statistics.mean(r.score.get("fact_share", 0.0) for r in on_topic)
        if on_topic
        else 0.0,
        "cites_expected": share(on_topic, lambda r: r.score.get("cites_expected", False)),
        "check_passed": share(on_topic, lambda r: r.verdict == "passed"),
        "check_failed": share(on_topic, lambda r: r.verdict == "failed"),
        "empty": share(on_topic, lambda r: r.score.get("empty", False)),
        "truncated": share(ok, lambda r: r.finish_reason in ("length", "max_tokens")),
        "off_topic_refused": share(off_topic, lambda r: r.score.get("refused", False)),
        "latency_ms_median": median(latencies),
        "latency_ms_p90": sorted(latencies)[int(len(latencies) * 0.9) - 1] if latencies else None,
        "prompt_tokens_median": median([r.prompt_tokens for r in ok]),
        "completion_tokens_median": median([r.completion_tokens for r in ok]),
    }


def _pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def render_report(data: dict[str, Any]) -> str:
    lines = [
        f"# Сравнение моделей: {data['dataset']}",
        "",
        f"Дата прогона: {data['started_at'][:16].replace('T', ' ')}; нормы на {data['on']}; "
        f"вопросов: {data['questions']} (из них не по теме: {data['off_topic']}).",
        "",
        "Условия: для всех моделей одинаковые источники (гибридный поиск, локальные векторы, "
        f"{SOURCES_K} фрагментов), промпт, температура 0,1; оценивается первая попытка "
        "без исправления. Факты проверяются нестрогими шаблонами — это оценка снизу.",
        "",
        "## Поиск норм",
        "",
        "| Способ | Нужная статья в выдаче | MRR |",
        "|---|---|---|",
    ]
    for name, r in data["retrieval"].items():
        lines.append(f"| {name} | {_pct(r['recall'])} | {r['mrr']:.2f} |")
    lines += [
        "",
        "## Модели ответа",
        "",
        "| Модель | Где | Верно (все факты) | Доля фактов | Сослалась на нужную статью "
        "| Проверка ссылок: без замечаний / с ошибками | Отказ не по теме | Медиана, с "
        "| Токены (вход / выход) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for model, s in data["summary"].items():
        where = "локально" if model.startswith("ollama/") else "API"
        tokens = (
            f"{s['prompt_tokens_median']:.0f} / {s['completion_tokens_median']:.0f}"
            if s["prompt_tokens_median"] is not None
            else "—"
        )
        latency = f"{s['latency_ms_median'] / 1000:.1f}" if s["latency_ms_median"] else "—"
        errors = f", ошибок вызова: {s['errors']}" if s["errors"] else ""
        lines.append(
            f"| `{model}` | {where} | {_pct(s['correct'])} | {_pct(s['fact_share'])} "
            f"| {_pct(s['cites_expected'])} "
            f"| {_pct(s['check_passed'])} / {_pct(s['check_failed'])} "
            f"| {_pct(s['off_topic_refused'])} | {latency} | {tokens}{errors} |"
        )
    models = list(data["summary"])
    lines += [
        "",
        "## По вопросам",
        "",
        "✓ — все факты, ◐ — часть, ✗ — ни одного, ⚠ — ошибка вызова или пустой ответ.",
        "",
        "| Вопрос | " + " | ".join(f"`{m.split('/', 1)[1]}`" for m in models) + " |",
        "|---|" + "---|" * len(models),
    ]
    by_key = {(r["question"], r["model"]): r for r in data["runs"]}
    for q in data["dataset_questions"]:
        cells = []
        for m in models:
            r = by_key.get((q["id"], m))
            if r is None or r["error"] or r["score"].get("empty"):
                cells.append("⚠")
            elif q.get("off_topic"):
                cells.append("✓" if r["score"].get("refused") else "✗")
            else:
                share = r["score"].get("fact_share", 0)
                cells.append("✓" if share == 1 else "◐" if share > 0 else "✗")
        lines.append(f"| {q['id']} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


async def main(args: argparse.Namespace) -> None:
    dataset = json.loads(Path(args.dataset).read_text("utf-8"))
    questions = load_questions(dataset)
    on = date.fromisoformat(dataset["on"])
    specs = [parse_model(m) for m in args.models]
    engine = create_async_engine(get_settings().database_url)
    registry = build_registry(get_provider_settings())
    started = datetime.now().isoformat(timespec="seconds")
    runs: list[Run] = []
    try:
        async with async_sessionmaker(engine, expire_on_commit=False)() as session:
            retrieval = await compare_retrieval(session, registry, questions, on)
            logger.info("Поиск: %s", {k: v["recall"] for k, v in retrieval.items()})
            gateway = _gateway(registry, DEFAULT_CONFIG, Mode.LOCAL)
            sources = {
                q.id: (await search(session, gateway, q.text, on, k=SOURCES_K)).hits
                for q in questions
            }
            for ref in specs:
                for q in questions:
                    run = await answer_once(session, registry, ref, q, sources[q.id], on)
                    runs.append(run)
                    logger.info(
                        "%s %s: %s %s",
                        run.model,
                        q.id,
                        run.score.get("fact_share"),
                        run.error or run.verdict,
                    )
    finally:
        await registry.aclose()
        await engine.dispose()

    data = {
        "dataset": dataset["name"],
        "on": dataset["on"],
        "started_at": started,
        "questions": len(questions),
        "off_topic": sum(q.off_topic for q in questions),
        "dataset_questions": dataset["questions"],
        "retrieval": retrieval,
        "summary": {
            label(ref): summarize(questions, [r for r in runs if r.model == label(ref)])
            for ref in specs
        },
        "runs": [asdict(r) for r in runs],
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    previous = out / "results.json"
    if args.append and previous.exists():
        # дописываем модели в готовый прогон: поиск и источники те же
        old = json.loads(previous.read_text("utf-8"))
        new_models = set(data["summary"])
        data["summary"] = {
            **{m: s for m, s in old["summary"].items() if m not in new_models},
            **data["summary"],
        }
        data["runs"] = [r for r in old["runs"] if r["model"] not in new_models] + data["runs"]
        data["started_at"], data["retrieval"] = old["started_at"], old["retrieval"]
    (out / "results.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), "utf-8")
    (out / "report.md").write_text(render_report(data), "utf-8")
    print(f"Готово: {out / 'report.md'}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--append", action="store_true", help="дописать модели в готовый прогон")
    asyncio.run(main(parser.parse_args()))
