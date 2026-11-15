import asyncio
import json
from datetime import date
from pathlib import Path

import pytest

from app.evaluation.answer_benchmark import (
    Answer,
    MeteredGateway,
    answer_messages,
    blind_packet,
    check_citations,
    evaluate_case,
)
from app.evaluation.answer_dataset import Dataset, fingerprint
from app.evaluation.answer_metrics import (
    BillableUsage,
    HumanGrade,
    Tariff,
    nearest_rank,
    paired_family_bootstrap,
    quality_bounds,
    token_cost,
)
from app.evaluation.answer_review import apply_grades
from app.evaluation.frozen_corpus import Snapshot, oracle_sources
from app.modules.ai.profiles import DEFAULT_CONFIG
from tests.fakes import fake_registry

DATA = Path(__file__).resolve().parents[2] / "evaluation/answers_v1.json"


def dataset():
    return Dataset.model_validate_json(DATA.read_bytes())


def answer(case):
    return {
        "answerability": "missing_facts",
        "issues": [
            {
                "id": i.id,
                "decision": "conditional",
                "conditions": [],
                "exceptions": [],
                "explanation": "Не хватает фактов",
            }
            for i in case.issues
        ],
        "citations": [],
        "answer": "Нужно посмотреть соглашение",
    }


def test_unresolved_is_not_a_success_or_a_legal_failure():
    assert quality_bounds([True, False, None, None]) == {
        "n": 4,
        "success": 1,
        "failure": 1,
        "unresolved": 2,
        "lower": 0.25,
        "upper": 0.75,
        "coverage": 0.5,
    }
    assert nearest_rank(list(range(1, 13)), 0.9) == 11
    rows = [{"case": "x", "family": "f", "variant": v, "solution_pass": None} for v in ("B2", "B3")]
    assert paired_family_bootstrap(rows, "B2", "B3")["status"] == "unresolved"


def test_cost_uses_disjoint_categories_and_preserves_unknowns():
    tariff = Tariff(
        provider="p",
        model="m",
        currency="USD",
        effective_on="2026-11-15",
        source_url="https://example.org",
        input_uncached=10,
        cache_read=1,
        cache_write=12,
        output=20,
    )
    usage = BillableUsage(input_uncached=100, cache_read=200, cache_write=50, output=100)
    assert token_cost(usage, tariff) == pytest.approx(0.0038)
    assert token_cost(BillableUsage(input_uncached=1, output=1), tariff) is None
    assert token_cost(usage, None) is None


def test_repeats_not_counted_as_independent_cases():
    rows = [
        {"case": case, "family": case, "variant": v, "solution_pass": v == "B3"}
        for case in ("a", "b")
        for v in ("B2", "B3")
        for _ in range(4)
    ]
    result = paired_family_bootstrap(rows, "B2", "B3")
    assert result["difference"] == 1 and result["families"] == result["cases"] == 2
    assert result["ci95"] == [1, 1]


def test_prompts_and_blind_review_do_not_leak_labels_or_methods():
    data = dataset()
    case = data.cases[0]
    sent = json.loads(answer_messages(case, [])[-1]["content"])
    assert "candidate_articles" not in sent and "source" not in sent and "gold" not in sent
    report = {
        "seed": 1,
        "rows": [{"case": case.id, "status": "ok", "answer": answer(case), "variant": "B2"}],
    }
    packet, key = blind_packet(report, data)
    assert "variant" not in packet["responses"][0]
    assert key["report_sha256"] == fingerprint(report)
    assert packet["responses"][0]["gold"] is None


async def test_failed_generation_and_timeout_are_recorded():
    case = dataset().cases[0]
    reg = fake_registry()
    ai = MeteredGateway(reg, DEFAULT_CONFIG.model_copy(deep=True), timeout=0.01)
    reg.get("ollama").reply = lambda _: "not JSON"
    snap = Snapshot(legal_as_of=case.legal_as_of, articles=[])
    row = await evaluate_case(None, ai, case, "B0", snap, {})
    assert row["status"] == "failed" and row["solution_pass"] is False
    assert row["attempts"][0]["result"]["text"] == "not JSON"
    assert row["elapsed_ms"] > 0

    async def slow(*args, **kwargs):
        await asyncio.sleep(1)

    reg.get("ollama").chat = slow
    row = await evaluate_case(None, ai, case, "B0", snap, {})
    assert row["censored"] and row["attempts"][0]["error_type"] == "TimeoutError"


async def test_successful_generation_remains_ungraded():
    case = dataset().cases[0]
    reg = fake_registry()
    reg.get("ollama").reply = lambda _: json.dumps(answer(case))
    ai = MeteredGateway(reg, DEFAULT_CONFIG.model_copy(deep=True))
    row = await evaluate_case(
        None, ai, case, "B0", Snapshot(legal_as_of=case.legal_as_of, articles=[]), {}
    )
    assert row["status"] == "ok" and row["solution_pass"] is None
    assert len(row["attempts"]) == 1


def test_oracle_rejects_machine_or_absent_gold():
    case = dataset().cases[0]
    with pytest.raises(ValueError, match="adjudicated"):
        oracle_sources(case, Snapshot(legal_as_of=case.legal_as_of, articles=[]))
    with pytest.raises(ValueError, match="date differs"):
        oracle_sources(case, Snapshot(legal_as_of=date(2020, 1, 1), articles=[]))


async def test_context_limit_is_not_silently_accepted():
    case = dataset().cases[0]
    reg = fake_registry()
    reg.get("ollama").num_ctx = 11  # FakeProvider reports prompt_tokens=10.
    reg.get("ollama").reply = lambda _: json.dumps(answer(case))
    ai = MeteredGateway(reg, DEFAULT_CONFIG.model_copy(deep=True))
    row = await evaluate_case(
        None, ai, case, "B0", Snapshot(legal_as_of=case.legal_as_of, articles=[]), {}
    )
    assert row["status"] == "failed" and row["attempts"][0]["context_at_limit"]


def test_citation_existence_does_not_claim_applicability():
    case = dataset().cases[0]
    value = answer(case)
    value["citations"] = [{"act": "unknown", "article": "999"}]
    result = check_citations(
        Answer.model_validate(value), Snapshot(legal_as_of=date.today(), articles=[]), [], "B0"
    )
    assert result[0]["exists_in_snapshot"] is False
    assert result[0]["article_was_shown"] is None and result[0]["applicability"] == "unresolved"


def test_review_cannot_attach_to_changed_answer_or_promote_unreviewed_gold():
    data = dataset()
    case = data.cases[0]
    report = {
        "seed": 1,
        "dataset_sha256": fingerprint(data.model_dump(mode="json")),
        "rows": [{"case": case.id, "status": "ok", "answer": answer(case), "variant": "B2"}],
    }
    packet, key = blind_packet(report, data)
    response = packet["responses"][0]
    grade = HumanGrade(
        response_id=response["response_id"],
        response_sha256=response["response_sha256"],
        reviewer="person-a",
        qualification="юрист",
        completed_on="2026-11-15",
        answerability_correct=True,
        issues=[
            {
                "id": "probation",
                "conclusion_correct": True,
                "conditions_complete": True,
                "exceptions_complete": True,
            }
        ],
        material_contradiction=False,
        citations_correct=True,
        rationale="Проверено",
    )
    with pytest.raises(ValueError, match="adjudicated"):
        apply_grades(report, data, key, [grade.model_dump()])
    report["rows"][0]["answer"]["answer"] = "изменено"
    with pytest.raises(ValueError, match="another report"):
        apply_grades(report, data, key, [grade.model_dump()])
