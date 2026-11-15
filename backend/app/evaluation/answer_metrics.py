"""Рубрика, незавершённые оценки, время и стоимость без подмены неизвестного нулём."""

import math
import random
import statistics
from collections import defaultdict

from pydantic import Field

from app.evaluation.answer_dataset import Strict


class IssueGrade(Strict):
    id: str
    conclusion_correct: bool
    conditions_complete: bool
    exceptions_complete: bool


class HumanGrade(Strict):
    response_id: str
    response_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    reviewer: str = Field(min_length=1)
    qualification: str = Field(min_length=1)
    completed_on: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    answerability_correct: bool
    issues: list[IssueGrade] = Field(min_length=1)
    material_contradiction: bool
    citations_correct: bool | None
    rationale: str = Field(min_length=1)

    def solution_pass(self, mandatory_ids: set[str]) -> bool:
        by_id = {i.id: i for i in self.issues}
        if len(by_id) != len(self.issues) or not mandatory_ids <= by_id.keys():
            raise ValueError("grade does not cover mandatory issues uniquely")
        return (
            self.answerability_correct
            and not self.material_contradiction
            and all(
                by_id[key].conclusion_correct
                and by_id[key].conditions_complete
                and by_id[key].exceptions_complete
                for key in mandatory_ids
            )
        )


def nearest_rank(values: list[float], p: float) -> float | None:
    if not 0 < p <= 1:
        raise ValueError("p must be in (0, 1]")
    return sorted(values)[math.ceil(p * len(values)) - 1] if values else None


def quality_bounds(values: list[bool | None]) -> dict:
    n = len(values)
    success = sum(v is True for v in values)
    unresolved = sum(v is None for v in values)
    return {
        "n": n,
        "success": success,
        "failure": n - success - unresolved,
        "unresolved": unresolved,
        "lower": success / n if n else None,
        "upper": (success + unresolved) / n if n else None,
        "coverage": (n - unresolved) / n if n else None,
    }


class BillableUsage(Strict):
    # Категории взаимоисключающие; reasoning входит в output и повторно не прибавляется.
    input_uncached: int | None = Field(None, ge=0)
    cache_read: int | None = Field(None, ge=0)
    cache_write: int | None = Field(None, ge=0)
    output: int | None = Field(None, ge=0)


class Tariff(Strict):
    provider: str
    model: str
    currency: str = Field(min_length=3, max_length=3)
    effective_on: str
    source_url: str
    # Цена за миллион токенов каждой категории, без неявных скидок.
    input_uncached: float = Field(ge=0)
    cache_read: float = Field(ge=0)
    cache_write: float = Field(ge=0)
    output: float = Field(ge=0)


def token_cost(usage: BillableUsage, tariff: Tariff | None) -> float | None:
    if tariff is None or any(v is None for v in usage.model_dump().values()):
        return None
    return sum(value * getattr(tariff, key) for key, value in usage.model_dump().items()) / 1e6


def paired_family_bootstrap(
    rows: list[dict], a: str, b: str, *, seed: int = 20261115, samples: int = 2000
) -> dict:
    """Повторы усредняются внутри случая, семьи ресемплируются целиком."""
    by_case = defaultdict(lambda: defaultdict(list))
    families = {}
    for row in rows:
        if row["variant"] not in (a, b):
            continue
        if row["solution_pass"] is None:
            return {"status": "unresolved", "difference": None, "ci95": None}
        by_case[row["case"]][row["variant"]].append(float(row["solution_pass"]))
        families[row["case"]] = row["family"]
    groups = defaultdict(list)
    for case, methods in by_case.items():
        if set(methods) != {a, b} or len(methods[a]) != len(methods[b]):
            raise ValueError("unpaired comparison")
        groups[families[case]].append(statistics.mean(methods[b]) - statistics.mean(methods[a]))
    if len(groups) < 2:
        return {"status": "insufficient_families", "difference": None, "ci95": None}
    rng = random.Random(seed)
    values = list(groups.values())
    estimates = []
    for _ in range(samples):
        sample = [v for group in rng.choices(values, k=len(values)) for v in group]
        estimates.append(statistics.mean(sample))
    return {
        "status": "estimated",
        "families": len(groups),
        "cases": len(by_case),
        "difference": statistics.mean(v for g in values for v in g),
        "ci95": [nearest_rank(estimates, 0.025), nearest_rank(estimates, 0.975)],
    }


def summarize(rows: list[dict]) -> dict:
    result = {}
    for variant in sorted({r["variant"] for r in rows}):
        selected = [r for r in rows if r["variant"] == variant]
        times = [r["elapsed_ms"] for r in selected]
        result[variant] = {
            "quality": quality_bounds([r["solution_pass"] for r in selected]),
            "operational_failures": sum(r["status"] != "ok" for r in selected),
            "median_attempt_ms": statistics.median(times),
            "p90_attempt_ms": nearest_rank(times, 0.9),
            "timed_out": sum(r.get("censored", False) for r in selected),
            "tool_errors": sum(
                sum(t["status"] == "error" for t in r.get("tool_trace", [])) for r in selected
            ),
            "navigation_degraded": sum(r.get("navigation_degraded", False) for r in selected),
            "navigation_no_sources": sum(r.get("navigation_no_sources", False) for r in selected),
            "model_calls": sum(len(r["attempts"]) for r in selected),
            "money": None,
            "money_status": "local_energy_and_amortization_not_measured",
        }
    return result
