import json
import re
from pathlib import Path

from app.evaluation.compare import parse_model
from app.evaluation.metrics import (
    CompoundQuestion,
    IssueGold,
    Question,
    load_questions,
    retrieval_rank,
    score_answer,
    score_issues,
)
from app.modules.assistant.analysis import is_compound
from app.modules.assistant.citations import CheckReport, Citation

DATASET = Path(__file__).parents[2] / "evaluation" / "labor_basic.json"


class FakeHit:
    def __init__(self, key: str):
        self.key = key


def report(citations=(), markers=()) -> CheckReport:
    return CheckReport(
        verdict="passed", citations=list(citations), markers=list(markers), issues=[]
    )


def test_dataset_is_valid():
    data = json.loads(DATASET.read_text("utf-8"))
    questions = load_questions(data)
    assert len({q.id for q in questions}) == len(questions)
    for q in questions:
        assert q.off_topic or (q.articles and q.facts), q.id
        for fact in q.facts:
            re.compile(fact["pattern"])


def test_facts_and_citations():
    q = Question(
        id="probation",
        text="?",
        articles=["70"],
        facts=[{"name": "3 месяца", "pattern": "(3|тр[её]х|три)[\s-]*(х\s*)?месяц"}],
    )
    sources = [FakeHit("71"), FakeHit("70")]
    good = score_answer(q, "Не более трёх месяцев [2].", report(markers=[2]), sources)
    assert good.correct and good.cites_expected and good.fact_share == 1
    by_text = score_answer(
        q, "Срок — 3-х месяцев.", report([Citation(text="ст. 70", act="ТК РФ", key="70")]), sources
    )
    assert by_text.correct and by_text.cites_expected
    wrong = score_answer(q, "Не более одного месяца [1].", report(markers=[1]), sources)
    assert not wrong.correct and not wrong.cites_expected and wrong.facts_missed == ["3 месяца"]
    assert score_answer(q, "", report(), sources).empty


def test_refusal_and_rank():
    q = Question(id="x", text="?", articles=["80"], facts=[], off_topic=True)
    assert score_answer(q, "Вопрос не относится к трудовому праву.", report(), []).refused
    assert retrieval_rank(Question("y", "?", ["80"], []), [FakeHit("81"), FakeHit("80")]) == 2
    assert retrieval_rank(Question("y", "?", ["80"], []), [FakeHit("81")]) is None


def test_parse_model():
    ref = parse_model("ollama/hf.co/yandex/YandexGPT-5-Lite-8B-instruct-GGUF:Q4_K_M")
    assert (ref.provider, ref.model, ref.options) == (
        "ollama",
        "hf.co/yandex/YandexGPT-5-Lite-8B-instruct-GGUF:Q4_K_M",
        {"think": False},
    )
    assert parse_model("yandex/yandexgpt-5.1").options == {}


COMPOUND = Path(__file__).parents[2] / "evaluation" / "labor_compound.json"


def test_compound_dataset_is_valid():
    data = json.loads(COMPOUND.read_text("utf-8"))
    questions = [CompoundQuestion.from_dict(q) for q in data["questions"]]
    assert len({q.id for q in questions}) == len(questions)
    for q in questions:
        # разбор включается правилом на каждом вопросе набора
        assert is_compound(q.text), q.id
        assert len(q.issues) >= 3, q.id
        for issue in q.issues:
            assert issue.articles and issue.facts, (q.id, issue.name)
            for fact in issue.facts:
                re.compile(fact["pattern"])
        for bad in q.forbidden:
            re.compile(bad["pattern"])


def test_score_issues():
    q = CompoundQuestion(
        id="x",
        text="?",
        issues=[
            IssueGold("Срок", ["180"], [{"name": "два месяца", "pattern": r"дв(а|ух)\s+месяц"}]),
            IssueGold("Суд", ["392"], [{"name": "год", "pattern": r"одного\s+года"}]),
        ],
        forbidden=[{"name": "«нельзя оспорить»", "pattern": r"нельзя\s+оспорить"}],
    )
    scores, bad = score_issues(
        q, "Предупредить должны за два месяца [1].", [FakeHit("180")], {"180"}
    )
    assert [(s.in_sources, s.cited, s.answered) for s in scores] == [
        (True, True, True),
        (False, False, False),
    ]
    assert scores[1].facts_missed == ["год"] and bad == []
    _, bad = score_issues(q, "Это нельзя оспорить.", [], set())
    assert bad == ["«нельзя оспорить»"]
