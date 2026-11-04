"""Метрики по статьям; несколько фрагментов одной статьи не увеличивают recall."""

import random
import statistics
from collections import defaultdict


def score(expected: set[tuple[str, str]], found: list[tuple[str, str]]) -> dict[str, float]:
    if not expected:
        raise ValueError("recall is undefined without gold articles")
    relevant = expected.intersection(found)
    first = next((i for i, article in enumerate(found, 1) if article in expected), None)
    return {"recall": len(relevant) / len(expected), "mrr": 1 / first if first else 0.0}


def paired_bootstrap(rows: list[dict], a: str, b: str, metric: str, seed: int) -> dict:
    """Средняя разница B-A; одна оценочная реплика на диалог, парный ресемплинг."""
    grouped = defaultdict(dict)
    for row in rows:
        if row["method"] in (a, b):
            if row["method"] in grouped[row["case"]]:
                raise ValueError("duplicate paired observation")
            grouped[row["case"]][row["method"]] = row["score"][metric]
    if not grouped or any(set(pair) != {a, b} for pair in grouped.values()):
        raise ValueError("paired observations required")
    deltas = [pair[b] - pair[a] for pair in grouped.values()]
    rng = random.Random(seed)
    samples = sorted(statistics.mean(rng.choices(deltas, k=len(deltas))) for _ in range(2000))
    return {
        "difference": statistics.mean(deltas),
        "ci95": [samples[49], samples[1949]],
        "dialogs": len(deltas),
        "resamples": 2000,
    }
