import hashlib
import json
from datetime import date
from pathlib import Path

import pytest

from app.evaluation.answer_benchmark import MissingNorm
from app.evaluation.answer_dataset import Dataset
from app.evaluation.frozen_corpus import Article, Snapshot
from app.evaluation.navigation import navigate, resolve_missing
from app.modules.ai.providers.base import ChatResult
from app.modules.corpus.navigation import CorpusNavigator, Limits, NavigationError


def snapshot():
    def article(key, text, repealed=False):
        return Article(
            act="tk",
            act_title="ТК РФ",
            key=key,
            heading=f"Статья {key}",
            path=["Часть I", "Глава 1"],
            text=text,
            fragments=[],
            redaction="r1",
            valid_from=date(2026, 1, 1),
            content_sha256=hashlib.sha256(text.encode()).hexdigest(),
            parser_content_hash="fixture",
            is_repealed=repealed,
            source_url="https://example.org/law",
        )

    return Snapshot(
        legal_as_of=date(2026, 11, 15),
        articles=[
            article("70", "См. статью 72. Текст нормы."),
            article("72", "Другой текст."),
            article("999", "Утратила силу", True),
        ],
    )


def test_structure_redaction_and_read_only_tools():
    nav = CorpusNavigator(snapshot())
    contents = nav.call("contents")
    section = nav.call("section", {"section_id": contents["sections"][0]["id"]})
    assert [a["key"] for a in section["articles"]] == ["70", "72", "999"]
    article = nav.call("article", {"act": "tk", "key": "70"})
    assert article["article"]["text"].startswith("См.")
    assert nav.call("redaction", {"act": "tk"})["redactions"] == [("r1", "2026-01-01")]
    with pytest.raises(NavigationError, match="call_budget"):
        nav.call("contents")
    assert len(nav.trace) == 5 and nav.trace[-1]["status"] == "error"


@pytest.mark.parametrize(
    "tool,args",
    [
        ("execute", {"command": "anything"}),
        ("article", {"act": "tk", "key": "70", "on": "2030-01-01"}),
        ("article", {"act": "tk", "key": "999"}),
        ("article", {"act": "other", "key": "70"}),
        ("article", {"act": "tk", "key": "70", "offset": -1}),
    ],
)
def test_rejects_unknown_tools_future_dates_repealed_and_wrong_acts(tool, args):
    nav = CorpusNavigator(snapshot())
    with pytest.raises(NavigationError):
        nav.call(tool, args)
    assert nav.trace[0]["status"] == "error"


def test_output_and_time_budgets_fail_without_delivering_text(monkeypatch):
    nav = CorpusNavigator(snapshot(), Limits(characters=100))
    with pytest.raises(NavigationError, match="output_budget"):
        nav.call("contents")
    assert "result" not in nav.trace[0]
    nav = CorpusNavigator(snapshot())
    monkeypatch.setattr("app.modules.corpus.navigation.time.monotonic", lambda: nav.started + 61)
    with pytest.raises(NavigationError, match="time_budget"):
        nav.call("contents")


def test_hash_and_future_redactions_are_checked():
    data = snapshot()
    data.articles[0].text = "Подменено"
    with pytest.raises(ValueError, match="hash"):
        CorpusNavigator(data)
    data = snapshot()
    data.articles[0].valid_from = date(2030, 1, 1)
    with pytest.raises(ValueError, match="future"):
        CorpusNavigator(data)


def test_missing_norm_is_a_proposal_until_resolved_in_correct_act():
    proposals = [
        MissingNorm(claim="Тезис", act=act, article=key, reason="Проверить")
        for act, key in (("tk", "72"), ("tk", "888"), ("other", "70"))
    ]
    sources, trace = resolve_missing(snapshot(), proposals)
    assert [s["key"] for s in sources] == ["72"]
    assert [t["status"] for t in trace] == ["ok", "error", "error"]


async def test_mcp_sdk_lists_only_read_tools_and_calls_same_navigator():
    pytest.importorskip("mcp")
    from app.evaluation.corpus_mcp import build_server

    server, nav = build_server(snapshot())
    tools = await server.list_tools()
    assert {t.name for t in tools} == {"contents", "section", "article", "crossrefs", "redaction"}
    assert all(t.annotations.readOnlyHint for t in tools)
    result = await server.call_tool("article", {"act": "tk", "key": "70"})
    assert result and nav.trace[0]["status"] == "ok"


async def test_navigation_chooses_only_real_section_and_articles():
    data = Dataset.model_validate_json(
        (Path(__file__).resolve().parents[2] / "evaluation/answers_v1.json").read_bytes()
    )
    snap = snapshot()
    section_id = next(iter(CorpusNavigator(snap).sections))

    class Selector:
        replies = [json.dumps({"section_id": section_id}), '{"keys":["70","72"]}']

        async def chat(self, *args, **kwargs):
            return ChatResult(self.replies.pop(0), "fake", "fake", 0)

    trace = []
    sources, actual_trace = await navigate(Selector(), data.cases[0], snap, [], trace)
    assert [s["key"] for s in sources] == ["70", "72"]
    assert actual_trace is trace and [t["tool"] for t in trace] == [
        "contents",
        "section",
        "article",
        "article",
    ]


async def test_navigation_keeps_partial_trace_if_model_fails():
    data = Dataset.model_validate_json(
        (Path(__file__).resolve().parents[2] / "evaluation/answers_v1.json").read_bytes()
    )

    class Failing:
        async def chat(self, *args, **kwargs):
            raise RuntimeError("model unavailable")

    trace = []
    with pytest.raises(RuntimeError):
        await navigate(Failing(), data.cases[0], snapshot(), [], trace)
    assert trace[0]["tool"] == "contents" and trace[0]["status"] == "ok"
