import copy
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.evaluation.answer_dataset import (
    Dataset,
    IssueLabel,
    Label,
    agreement,
    annotation_packet,
    fingerprint,
    merge_reviews,
)

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


def test_review_import_rejects_wrong_dataset_and_blank_forms():
    data = Dataset.model_validate_json(DATA.read_bytes())
    packet = annotation_packet(data, "a")
    with pytest.raises(ValidationError):
        merge_reviews(data, [packet])
    packet["dataset_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="another dataset"):
        merge_reviews(data, [packet])


def test_families_and_source_groups_cannot_cross_splits_even_without_gold():
    # Модель проверяет разделения прежде допуска test, так что нарушение видно отдельно.
    raw = Dataset.model_validate_json(DATA.read_bytes()).model_dump(mode="json")
    other = copy.deepcopy(raw["cases"][0])
    other.update(id="new", split="test", exposure=[])
    raw["cases"].append(other)
    with pytest.raises(ValidationError, match="family leaks"):
        Dataset.model_validate(raw)


def unavailable_issue():
    return {
        "id": "probation",
        "decision": "undetermined",
        "conditions": [],
        "exceptions": [],
        "rationale": "Недостающий источник не позволяет сделать вывод",
        "evidence_alternatives": [],
        "evidence_status": "unavailable",
        "evidence_unavailable_reason": "Не найдена необходимая редакция нормы",
    }


def test_undetermined_can_explicitly_record_missing_authority_without_fabricating_norms():
    label = Label.model_validate(
        {
            "answerability": "missing_authority",
            "issues": [unavailable_issue()],
            "missing_facts": [],
            "missing_authorities": ["Необходимая редакция нормы"],
        }
    )
    assert label.issues[0].evidence_alternatives == []
    assert label.issues[0].evidence_status == "unavailable"


@pytest.mark.parametrize(
    "changes",
    [
        {"evidence_unavailable_reason": None},
        {"evidence_unavailable_reason": "   "},
        {"decision": "entailed"},
        {"decision": "contradicted"},
        {"decision": "conditional"},
        {"evidence_status": "supported"},
        {"evidence_alternatives": [[]]},
    ],
)
def test_unavailable_evidence_cannot_hide_a_determined_or_unexplained_conclusion(changes):
    with pytest.raises(ValidationError):
        IssueLabel.model_validate({**unavailable_issue(), **changes})


def completed_packet(data):
    packet = annotation_packet(data, "a")
    packet["qualification"] = "юрист"
    for entry in packet["cases"]:
        entry.update(
            completed_on="2026-11-15",
            independent=True,
            label={
                "answerability": "missing_authority",
                "issues": [{**unavailable_issue(), "id": i["id"]} for i in entry["issues"]],
                "missing_facts": [],
                "missing_authorities": ["Необходимая редакция нормы"],
            },
        )
    return packet


def test_review_import_accepts_completed_original_packet():
    data = Dataset.model_validate_json(DATA.read_bytes())
    merged = merge_reviews(data, [completed_packet(data)])
    assert all(len(c.reviews) == 1 and c.gold is None for c in merged.cases)
    assert all(not c.reviews for c in data.cases)


@pytest.mark.parametrize("field", ["legal_as_of", "question", "facts", "issues", "source"])
@pytest.mark.parametrize("remove", [False, True])
def test_review_import_rejects_changed_case_with_original_dataset_hash(field, remove):
    data = Dataset.model_validate_json(DATA.read_bytes())
    packet = completed_packet(data)
    if remove:
        del packet["cases"][0][field]
    else:
        packet["cases"][0][field] = "Изменённое задание"
    assert packet["dataset_sha256"] == fingerprint(data.model_dump(mode="json"))
    with pytest.raises(ValueError, match="immutable case fields"):
        merge_reviews(data, [packet])
