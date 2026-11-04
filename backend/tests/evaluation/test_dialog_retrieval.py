import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.evaluation.dialog_dataset import Dataset
from app.evaluation.retrieval_metrics import paired_bootstrap, score

DATASET = Path(__file__).parents[2] / "evaluation/dialogs_v1.json"


def test_frozen_dataset_disjoint_families_and_articles():
    dataset = Dataset.model_validate_json(DATASET.read_bytes())
    splits = {s: [c for c in dataset.cases if c.split == s] for s in ("development", "evaluation")}
    assert len(splits["development"]) == 6 and len(splits["evaluation"]) == 12
    gold = {s: {(a.act, a.key) for c in cases for a in c.articles} for s, cases in splits.items()}
    assert not gold["development"] & gold["evaluation"]
    assert dataset.annotation == "synthetic_unreviewed"


def test_split_leakage_and_duplicate_id_rejected():
    raw = json.loads(DATASET.read_bytes())
    raw["cases"][-1]["family"] = raw["cases"][0]["family"]
    with pytest.raises(ValidationError, match="family crosses"):
        Dataset.model_validate(raw)
    raw = json.loads(DATASET.read_bytes())
    raw["cases"][-1]["id"] = raw["cases"][0]["id"]
    with pytest.raises(ValidationError, match="duplicate case"):
        Dataset.model_validate(raw)


def test_recall_counts_articles_not_chunks_and_distinguishes_acts():
    gold = {("tk", "70"), ("tk", "71")}
    result = score(gold, [("other", "70"), ("tk", "70"), ("tk", "70")])
    assert result == {"recall": 0.5, "mrr": 0.5}
    assert score(gold, []) == {"recall": 0, "mrr": 0}
    with pytest.raises(ValueError):
        score(set(), [])


def test_paired_bootstrap_requires_pairs_and_is_reproducible():
    rows = [
        {"case": case, "method": method, "score": {"recall": value}}
        for case in ("x", "y", "z")
        for method, value in [("a", 0), ("b", 1)]
    ]
    first = paired_bootstrap(rows, "a", "b", "recall", 42)
    assert first["difference"] == 1 and first["ci95"] == [1, 1]
    assert first == paired_bootstrap(rows, "a", "b", "recall", 42)
    with pytest.raises(ValueError, match="paired"):
        paired_bootstrap(rows[:-1], "a", "b", "recall", 42)
