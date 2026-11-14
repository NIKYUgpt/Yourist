import copy
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.evaluation.answer_dataset import Dataset, agreement, annotation_packet, fingerprint

DATA = Path(__file__).resolve().parents[2] / "evaluation/answers_v1.json"


def test_packets_do_not_anchor_independent_reviewers():
    data = Dataset.model_validate_json(DATA.read_bytes())
    packet = annotation_packet(data, "independent-a")
    assert packet["dataset_sha256"] == fingerprint(data.model_dump(mode="json"))
    assert len(packet["cases"]) == 4
    for case in packet["cases"]:
        assert case["label"] is None
        assert "candidate_articles" not in case and "machine_draft" not in case
    assert agreement(data)["decision"]["cohen_kappa"] is None
    assert not any(c.gold for c in data.cases)


def test_rejects_exposed_test_and_family_leaks():
    data = Dataset.model_validate_json(DATA.read_bytes()).model_dump(mode="json")
    bad = copy.deepcopy(data)
    bad["cases"][0]["split"] = "test"
    with pytest.raises(ValidationError, match="exposed"):
        Dataset.model_validate(bad)
    bad["cases"][0]["exposure"] = []
    with pytest.raises(ValidationError, match="adjudicated"):
        Dataset.model_validate(bad)
    bad = copy.deepcopy(data)
    bad["cases"].append(copy.deepcopy(bad["cases"][0]))
    with pytest.raises(ValidationError, match="duplicate case"):
        Dataset.model_validate(bad)


def test_machine_labels_cannot_be_used_as_gold():
    data = Dataset.model_validate_json(DATA.read_bytes())
    case = data.cases[0].model_dump(mode="json")
    norm = {
        "act": "tk",
        "article": "70",
        "redaction": "r1",
        "content_sha256": "a" * 64,
        "quote": "Текст",
        "source_url": "https://example.org/law",
    }
    label = {
        "answerability": "missing_facts",
        "issues": [
            {
                "id": "probation",
                "decision": "conditional",
                "conditions": ["Условие"],
                "exceptions": [],
                "rationale": "Основание",
                "evidence_alternatives": [[norm]],
            }
        ],
        "missing_facts": ["Соглашение"],
        "missing_authorities": [],
    }
    case["machine_draft"] = label
    data.cases[0] = type(data.cases[0]).model_validate(case)
    assert data.cases[0].gold is None
    review = {
        "reviewer": "a",
        "qualification": "юрист",
        "completed_on": "2026-11-15",
        "independent": True,
        "label": label,
    }
    case["reviews"] = [review, review]
    with pytest.raises(ValidationError, match="same person"):
        type(data.cases[0]).model_validate(case)
    case["reviews"] = [review, {**review, "reviewer": "b"}]
    case["adjudication"] = {
        "reviewer": "c",
        "completed_on": "2026-11-15",
        "resolution": "Совпало",
        "label": label,
    }
    assert type(data.cases[0]).model_validate(case).gold is not None
