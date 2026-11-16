import asyncio
import copy
import json
from datetime import date
from pathlib import Path

import httpx
import pytest

from app.evaluation.answer_benchmark import (
    VERIFY_SYSTEM,
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
    summarize,
    token_cost,
)
from app.evaluation.answer_review import apply_grades
from app.evaluation.frozen_corpus import Snapshot, oracle_sources
from app.modules.ai.profiles import DEFAULT_CONFIG, Task
from app.modules.ai.providers.ollama import OllamaProvider
from app.modules.ai.registry import Registry
from app.modules.search.retrieval import SearchResult
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


@pytest.mark.parametrize("task,budget", [(Task.ANSWER, 2048), (Task.VERIFY, 1024)])
async def test_profile_cannot_override_benchmark_controls(task, budget):
    captured = []

    def respond(request):
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"message": {"content": "{}"}, "done_reason": "stop"},
        )

    provider = OllamaProvider(
        "http://127.0.0.1:11434",
        num_ctx=8192,
        keep_alive="5m",
        timeout=1,
        retries=0,
        transport=httpx.MockTransport(respond),
    )
    config = DEFAULT_CONFIG.model_copy(deep=True)
    config.tasks[task].local.options.update(temperature=0.9, num_predict=-1)
    ai = MeteredGateway(Registry({"ollama": provider}), config)
    try:
        await ai.chat(
            task, [{"role": "user", "content": "fixture"}], temperature=0.0, max_tokens=budget
        )
    finally:
        await provider.aclose()
    assert captured[0]["options"]["temperature"] == 0.0
    assert captured[0]["options"]["num_predict"] == budget
    assert ai.attempts[0]["options"]["provider_options"]["num_predict"] == budget
    assert config.tasks[task].local.options["num_predict"] == -1


async def test_wrapped_http_timeout_is_censored_without_hidden_retries():
    calls = []

    def timeout(request):
        calls.append(request)
        raise httpx.ReadTimeout("fixture timeout", request=request)

    provider = OllamaProvider(
        "http://127.0.0.1:11434",
        num_ctx=8192,
        keep_alive="5m",
        timeout=1,
        retries=0,
        transport=httpx.MockTransport(timeout),
    )
    ai = MeteredGateway(Registry({"ollama": provider}), DEFAULT_CONFIG.model_copy(deep=True))
    case = dataset().cases[0]
    try:
        row = await evaluate_case(
            None, ai, case, "B0", Snapshot(legal_as_of=case.legal_as_of, articles=[]), {}
        )
    finally:
        await provider.aclose()
    assert len(calls) == len(row["attempts"]) == 1
    assert row["censored"] and row["attempts"][0]["censored"]
    assert row["error_type"] == "ProviderError"


async def test_cancellation_checkpoints_the_inflight_attempt_before_propagating():
    started = asyncio.Event()

    async def slow(*args, **kwargs):
        started.set()
        await asyncio.Event().wait()

    reg = fake_registry()
    reg.get("ollama").chat = slow
    ai = MeteredGateway(reg, DEFAULT_CONFIG.model_copy(deep=True))
    case = dataset().cases[0]
    saved = []
    task = asyncio.create_task(
        evaluate_case(
            None,
            ai,
            case,
            "B0",
            Snapshot(legal_as_of=case.legal_as_of, articles=[]),
            {},
            checkpoint=lambda row: saved.append(copy.deepcopy(row)),
        )
    )
    await started.wait()
    assert saved[-1]["attempts"][0]["status"] == "running"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    row = saved[-1]
    assert row["status"] == "interrupted" and row["censored"]
    assert row["solution_pass"] is False and row["elapsed_ms"] > 0
    assert row["attempts"][0]["censored"] and row["attempts"][0]["elapsed_ms"] > 0
    assert ai.checkpoint is None


async def test_b0_b3_inputs_and_single_revision_are_controlled(monkeypatch):
    searches = []

    async def fake_search(session, ai, query, on, **kwargs):
        searches.append((ai is None, query, on, kwargs))
        return SearchResult(query=query, on=on, hits=[], redactions=[])

    monkeypatch.setattr("app.evaluation.answer_benchmark.search", fake_search)
    case = dataset().cases[0]
    reg = fake_registry()
    reg.get("ollama").reply = lambda messages: json.dumps(
        {"needs_revision": True, "problems": ["fixture"], "missing_norms": []}
        if messages[0]["content"] == VERIFY_SYSTEM
        else answer(case)
    )
    ai = MeteredGateway(reg, DEFAULT_CONFIG.model_copy(deep=True))
    snapshot = Snapshot(legal_as_of=case.legal_as_of, articles=[])
    rows = {
        v: await evaluate_case(None, ai, case, v, snapshot, {}) for v in ("B0", "B1", "B2", "B3")
    }
    assert all(r["status"] == "ok" and r["solution_pass"] is None for r in rows.values())
    assert [item[0] for item in searches] == [True, False, False]
    assert all(item[1:3] == (case.question, case.legal_as_of) for item in searches)
    assert [item[3]["expand"] for item in searches] == [False, True, True]
    assert all(item[3]["rerank"] is False for item in searches)
    assert rows["B2"]["attempts"][0]["messages"] == rows["B3"]["attempts"][0]["messages"]
    assert [a["stage"] for a in rows["B3"]["attempts"]] == ["answer", "verify", "revision"]
    for variant, row in rows.items():
        payload = json.loads(row["attempts"][0]["messages"][-1]["content"])
        assert set(payload) == {
            "question",
            "facts",
            "date",
            "issues",
            "sources",
            "allow_prior_knowledge",
        }
        assert payload["allow_prior_knowledge"] is (variant == "B0")
        assert row["attempts"][0]["options"]["max_tokens"] == 2048
    assert rows["B3"]["attempts"][1]["options"]["max_tokens"] == 1024
    assert rows["B3"]["attempts"][2]["options"]["max_tokens"] == 2048


@pytest.mark.parametrize("repeat_ids,match", [([0, 0], "duplicate"), ([0, 2], "unpaired repeat")])
def test_bootstrap_rejects_equal_length_but_invalid_repeat_pairing(repeat_ids, match):
    rows = [
        {"case": case, "family": case, "variant": variant, "repeat": repeat, "solution_pass": True}
        for case in ("a", "b")
        for variant, repeats in (("B2", [0, 1]), ("B3", repeat_ids))
        for repeat in repeats
    ]
    with pytest.raises(ValueError, match=match):
        paired_family_bootstrap(rows, "B2", "B3")


def test_token_summary_preserves_unknown_and_counts_failed_returned_generations():
    row = {
        "variant": "B3",
        "status": "failed",
        "elapsed_ms": 100,
        "solution_pass": False,
        "attempts": [
            {
                "operation": "chat",
                "status": "ok",
                "result": {"usage": {"prompt_tokens": 10, "completion_tokens": 5}},
            },
            {"operation": "chat", "status": "error"},
            {"operation": "embed", "status": "ok", "usage": None},
        ],
    }
    summary = summarize([row])["B3"]
    assert summary["chat_tokens"]["prompt_tokens"] == {
        "total": None,
        "observed": 10,
        "missing_calls": 1,
    }
    assert summary["chat_tokens"]["completion_tokens"]["observed"] == 5
    assert summary["model_calls"] == 3 and summary["embedding_calls_without_usage"] == 1
    assert summary["money"] is None
