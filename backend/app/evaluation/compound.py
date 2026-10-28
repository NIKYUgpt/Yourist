"""Сравнение способов поиска для составных вопросов (несколько проблем в одном вопросе).

    python -m app.evaluation.compound --dataset evaluation/labor_compound.json \\
        --out evaluation/results/run

Варианты:
- k8 — обычный поиск на 8 фрагментов (так отвечали на любой вопрос до разбора);
- k16 — обычный поиск на 16 фрагментов: «просто взять больше»;
- sentences — без модели: по 4 фрагмента на вопрос целиком и по 3 на каждое предложение;
- llm-search — разбор моделью и поиск по проблемам, промпт обычный;
- llm — разбор моделью, поиск по проблемам и перечень проблем в промпте;
- llm-sentences — как llm, но искать ещё и по каждому предложению вопроса.

Разбор и ответ — модели по умолчанию для локального режима; оценивается первая попытка
ответа без проверяющей модели, как в compare.py. Сначала разбираются все вопросы,
затем идут все ответы — чтобы модели не вытесняли друг друга из видеопамяти.
"""

import argparse
import asyncio
import json
import logging
import statistics
from dataclasses import asdict, dataclass, field, fields
from datetime import date, datetime
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.evaluation.compare import _THINK, _gateway, label, parse_model
from app.evaluation.metrics import CompoundQuestion, cited_keys, score_issues
from app.modules.ai.config import get_provider_settings
from app.modules.ai.profiles import DEFAULT_CONFIG, Mode, Task
from app.modules.ai.providers.base import ProviderError, Timer
from app.modules.ai.registry import build_registry
from app.modules.assistant import prompts
from app.modules.assistant.analysis import Analysis, analyze, is_compound, sentences
from app.modules.assistant.citations import check
from app.modules.search.retrieval import Hit, search, search_issues

logger = logging.getLogger(__name__)

VARIANTS = {
    "k8": "обычный поиск, 8 фрагментов",
    "k16": "обычный поиск, 16 фрагментов",
    "sentences": "по предложениям (без модели)",
    "llm-search": "разбор моделью, промпт обычный",
    "llm": "разбор моделью + перечень проблем в промпте",
    "llm-sentences": "разбор моделью + перечень, поиск ещё и по предложениям",
}
MAX_TOKENS = 2000


@dataclass
class Run:
    question: str
    variant: str
    sources: int = 0
    search_ms: int = 0
    text: str = ""
    error: str | None = None
    latency_ms: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    verdict: str | None = None
    issues: list[dict[str, Any]] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)


async def _sources(
    session, gateway, q: CompoundQuestion, variant: str, analysis: Analysis | None, on: date
) -> tuple[list[Hit], list[tuple[str, list[int]]] | None]:
    if variant in ("k8", "k16"):
        found = await search(session, gateway, q.text, on, k=8 if variant == "k8" else 16)
        return found.hits, None
    if variant == "sentences":
        found, _ = await search_issues(session, gateway, q.text, sentences(q.text), on)
        return found.hits, None
    if analysis is None or not analysis.compound:
        # как в конвейере: без разбора — обычный поиск
        return (await search(session, gateway, q.text, on, k=8)).hits, None
    queries = [i.query for i in analysis.issues]
    if variant == "llm-sentences":
        queries += sentences(q.text)
    found, linked = await search_issues(session, gateway, q.text, queries, on)
    issues = [(i.issue, n) for i, n in zip(analysis.issues, linked, strict=False)]
    return found.hits, None if variant == "llm-search" else issues


def summarize(questions: list[CompoundQuestion], runs: list[Run]) -> dict[str, Any]:
    issues = [i for r in runs for i in r.issues]
    ok = [r for r in runs if r.error is None]

    def share(items, pred) -> float:
        return sum(1 for x in items if pred(x)) / len(items) if items else 0.0

    def median(values) -> float | None:
        clean = [v for v in values if v is not None]
        return statistics.median(clean) if clean else None

    facts_total = sum(len(i["facts_found"]) + len(i["facts_missed"]) for i in issues)
    return {
        "questions": len(runs),
        "errors": len(runs) - len(ok),
        "issues": len(issues),
        "issue_in_sources": share(issues, lambda i: i["in_sources"]),
        "issue_cited": share(issues, lambda i: i["cited"]),
        "issue_answered": share(issues, lambda i: not i["facts_missed"]),
        "fact_share": sum(len(i["facts_found"]) for i in issues) / facts_total
        if facts_total
        else 0.0,
        "contradictions": sum(len(r.contradictions) for r in runs),
        "check_passed": share(runs, lambda r: r.verdict == "passed"),
        "check_failed": share(runs, lambda r: r.verdict == "failed"),
        "truncated": share(ok, lambda r: r.finish_reason in ("length", "max_tokens")),
        "sources_mean": statistics.mean(r.sources for r in runs) if runs else 0,
        "search_ms_median": median([r.search_ms for r in runs]),
        "latency_ms_median": median([r.latency_ms for r in ok]),
        "prompt_tokens_median": median([r.prompt_tokens for r in ok]),
        "prompt_tokens_max": max((r.prompt_tokens or 0 for r in ok), default=None),
        "completion_tokens_median": median([r.completion_tokens for r in ok]),
    }


def _pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def render_report(data: dict[str, Any]) -> str:
    lines = [
        f"# Составные вопросы: {data['dataset']}",
        "",
        f"Дата прогона: {data['started_at'][:16].replace('T', ' ')}; нормы на {data['on']}; "
        f"вопросов: {data['questions']}, проблем в них: {data['issues']}. "
        f"Ответ — `{data['answer_model']}`, разбор — `{data['analyze_model']}`.",
        "",
        "Условия: локальные векторы bge-m3, температура 0,1, первая попытка без проверяющей "
        "модели. Факты — нестрогие шаблоны (оценка снизу); «проблема раскрыта» — найдены все "
        "её факты.",
        "",
        "## Все вопросы",
        "",
    ]

    def table(summary: dict[str, Any]) -> list[str]:
        rows = [
            "| Вариант | Статья проблемы в источниках | Ответ сослался на неё "
            "| Проблема раскрыта | Доля фактов | Проверка ссылок без замечаний "
            "| Источников | Поиск, с | Ответ, с | Токены промпта (медиана / макс.) |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for variant, s in summary.items():
            search_s = f"{s['search_ms_median'] / 1000:.1f}" if s["search_ms_median"] else "—"
            answer_s = f"{s['latency_ms_median'] / 1000:.1f}" if s["latency_ms_median"] else "—"
            tokens = (
                f"{s['prompt_tokens_median']:.0f} / {s['prompt_tokens_max']}"
                if s["prompt_tokens_median"]
                else "—"
            )
            errors = f", ошибок вызова: {s['errors']}" if s["errors"] else ""
            rows.append(
                f"| {VARIANTS[variant]} | {_pct(s['issue_in_sources'])} "
                f"| {_pct(s['issue_cited'])} | {_pct(s['issue_answered'])} "
                f"| {_pct(s['fact_share'])} | {_pct(s['check_passed'])} "
                f"| {s['sources_mean']:.1f} | {search_s} | {answer_s} | {tokens}{errors} |"
            )
        return rows

    lines += table(data["summary"])
    lines += [
        "",
        "## Без вопросов, на которых разбиралась проблема (dev)",
        "",
        *table(data["summary_heldout"]),
        "",
        "## Разбор моделью",
        "",
        "| Вопрос | Разбирать (правило) | Проблем | Разбор, с | Проблемы |",
        "|---|---|---|---|---|",
    ]
    for qid, a in data["analysis"].items():
        found = "; ".join(i["issue"] for i in a["issues"]) or (a["error"] or "—")
        seconds = f"{a['latency_ms'] / 1000:.1f}" if a["latency_ms"] else "—"
        lines.append(
            f"| {qid} | {'да' if a['gate'] else 'нет'} | {len(a['issues'])} | {seconds} | {found} |"
        )
    lines += [
        "",
        "## По проблемам",
        "",
        "Источники / ответ: ✓ — статья в источниках и все факты в ответе, ◐ — статья "
        "в источниках, фактов нет или не все, · — статьи нет в источниках, но факты есть, "
        "✗ — ни статьи, ни фактов.",
        "",
        "| Вопрос | Проблема | " + " | ".join(data["summary"]) + " |",
        "|---|---|" + "---|" * len(data["summary"]),
    ]
    by_key = {(r["question"], r["variant"]): r for r in data["runs"]}
    for q in data["dataset_questions"]:
        for n, issue in enumerate(q["issues"]):
            cells = []
            for variant in data["summary"]:
                run = by_key.get((q["id"], variant))
                if run is None or run["error"]:
                    cells.append("⚠")
                    continue
                s = run["issues"][n]
                answered = not s["facts_missed"]
                cells.append(
                    "✓"
                    if s["in_sources"] and answered
                    else "◐"
                    if s["in_sources"]
                    else "·"
                    if answered
                    else "✗"
                )
            lines.append(f"| {q['id']} | {issue['name']} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def rescore(data: dict[str, Any], dataset: dict[str, Any]) -> dict[str, Any]:
    """Пересчитать факты по сохранённым ответам после уточнения эталона — без вызова моделей.

    Источники и ссылки ответа от эталона не зависят и остаются прежними.
    """
    questions = {q["id"]: CompoundQuestion.from_dict(q) for q in dataset["questions"]}
    for run in data["runs"]:
        if run["error"]:
            continue
        q = questions[run["question"]]
        scores, run["contradictions"] = score_issues(q, run["text"], [], set())
        run["issues"] = [
            {
                **asdict(s),
                "in_sources": old["in_sources"],
                "cited": old["cited"],
                "answered": s.answered,
            }
            for s, old in zip(scores, run["issues"], strict=True)
        ]
    names = {f.name for f in fields(Run)}
    runs = [Run(**{k: v for k, v in r.items() if k in names}) for r in data["runs"]]
    heldout = {qid for qid, q in questions.items() if not q.dev}
    data["dataset_questions"] = dataset["questions"]
    data["summary"] = {
        v: summarize([], [r for r in runs if r.variant == v]) for v in data["summary"]
    }
    data["summary_heldout"] = {
        v: summarize([], [r for r in runs if r.variant == v and r.question in heldout])
        for v in data["summary"]
    }
    return data


async def main(args: argparse.Namespace) -> None:
    dataset = json.loads(Path(args.dataset).read_text("utf-8"))
    if args.rescore:
        out = Path(args.out)
        data = rescore(json.loads((out / "results.json").read_text("utf-8")), dataset)
        (out / "results.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), "utf-8")
        (out / "report.md").write_text(render_report(data), "utf-8")
        print(f"Пересчитано: {out / 'report.md'}")
        return
    questions = [CompoundQuestion.from_dict(q) for q in dataset["questions"]]
    on = date.fromisoformat(dataset["on"])
    answer_ref = parse_model(args.answer_model) if args.answer_model else None
    answer_ref = answer_ref or DEFAULT_CONFIG.tasks[Task.ANSWER].local
    analyze_ref = parse_model(args.analyze_model) if args.analyze_model else None
    analyze_ref = analyze_ref or DEFAULT_CONFIG.tasks[Task.ANALYZE].local
    variants = args.variants or list(VARIANTS)
    engine = create_async_engine(get_settings().database_url)
    registry = build_registry(get_provider_settings())
    started = datetime.now().isoformat(timespec="seconds")
    runs: list[Run] = []
    analyses: dict[str, dict[str, Any]] = {}
    try:
        async with async_sessionmaker(engine, expire_on_commit=False)() as session:
            gateway = _gateway(registry, DEFAULT_CONFIG, Mode.LOCAL)

            async def chat(messages, **kwargs):
                return await registry.get(analyze_ref.provider).chat(
                    analyze_ref.model, messages, options=analyze_ref.options, **kwargs
                )

            parsed: dict[str, Analysis | None] = {}
            for q in questions:
                gate = is_compound(q.text)
                parsed[q.id] = await analyze(chat, q.text) if gate else None
                a = parsed[q.id] or Analysis()
                analyses[q.id] = {"gate": gate, **asdict(a)}
                logger.info("разбор %s: %d проблем %s", q.id, len(a.issues), a.error or "")

            prepared = {}
            for q in questions:
                for variant in variants:
                    timer = Timer()
                    hits, issues = await _sources(session, gateway, q, variant, parsed[q.id], on)
                    prepared[q.id, variant] = (hits, issues, timer.ms)

            provider = registry.get(answer_ref.provider)
            for variant in variants:
                for q in questions:
                    hits, issues, search_ms = prepared[q.id, variant]
                    run = Run(q.id, variant, sources=len(hits), search_ms=search_ms)
                    messages = prompts.build_messages(q.text, hits, on, issues)
                    try:
                        result = await provider.chat(
                            answer_ref.model,
                            messages,
                            temperature=0.1,
                            max_tokens=MAX_TOKENS,
                            options=answer_ref.options,
                        )
                    except ProviderError as exc:
                        run.error = str(exc)
                        runs.append(run)
                        continue
                    run.text = _THINK.sub("", result.text).strip()
                    run.latency_ms = result.latency_ms
                    run.prompt_tokens = result.usage.prompt_tokens
                    run.completion_tokens = result.usage.completion_tokens
                    run.finish_reason = result.finish_reason
                    report = await check(session, run.text, hits, on)
                    run.verdict = report.verdict
                    scores, run.contradictions = score_issues(
                        q, run.text, hits, cited_keys(report, hits)
                    )
                    run.issues = [{**asdict(s), "answered": s.answered} for s in scores]
                    runs.append(run)
                    logger.info(
                        "%s %s: раскрыто %d/%d, %s",
                        variant,
                        q.id,
                        sum(s.answered for s in scores),
                        len(scores),
                        run.verdict,
                    )
    finally:
        await registry.aclose()
        await engine.dispose()

    heldout = {q.id for q in questions if not q.dev}
    data = {
        "dataset": dataset["name"],
        "on": dataset["on"],
        "started_at": started,
        "answer_model": label(answer_ref),
        "analyze_model": label(analyze_ref),
        "questions": len(questions),
        "issues": sum(len(q.issues) for q in questions),
        "dataset_questions": dataset["questions"],
        "analysis": analyses,
        "summary": {v: summarize(questions, [r for r in runs if r.variant == v]) for v in variants},
        "summary_heldout": {
            v: summarize(questions, [r for r in runs if r.variant == v and r.question in heldout])
            for v in variants
        },
        "runs": [asdict(r) for r in runs],
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), "utf-8")
    (out / "report.md").write_text(render_report(data), "utf-8")
    print(f"Готово: {out / 'report.md'}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--answer-model", help="по умолчанию — локальная модель ответа")
    parser.add_argument("--analyze-model", help="по умолчанию — локальная модель разбора")
    parser.add_argument("--variants", nargs="*", choices=list(VARIANTS))
    parser.add_argument(
        "--rescore", action="store_true", help="пересчитать оценки готового прогона по эталону"
    )
    asyncio.run(main(parser.parse_args()))
