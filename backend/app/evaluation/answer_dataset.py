"""Случаи, независимые аннотации и допуск к оценке правовых ответов.

Схема проверяет происхождение и полноту записей, но не удостоверяет компетенции
людей или юридическую истинность. Машинный черновик никогда не становится gold.
"""

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

Decision = Literal["entailed", "contradicted", "conditional", "undetermined"]
Answerability = Literal[
    "answerable", "missing_facts", "missing_authority", "out_of_scope", "contested"
]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Norm(Strict):
    act: str = Field(min_length=1)
    article: str = Field(pattern=r"^\d+(?:[.-]\d+)*$")
    part: str | None = None
    point: str | None = None
    paragraph: str | None = None
    redaction: str = Field(min_length=1)
    content_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    quote: str = Field(min_length=1)
    source_url: str = Field(pattern=r"^https?://")


class IssueLabel(Strict):
    id: str = Field(min_length=1)
    decision: Decision
    conditions: list[str]
    exceptions: list[str]
    rationale: str = Field(min_length=1)
    # Каждый внутренний список — обязательный набор (AND), внешние — варианты (OR).
    evidence_alternatives: list[list[Norm]]
    evidence_status: Literal["supported", "unavailable"] = "supported"
    evidence_unavailable_reason: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def nonempty_evidence(self) -> Self:
        if any(not group for group in self.evidence_alternatives):
            raise ValueError("empty evidence alternative")
        if self.evidence_status == "unavailable":
            if self.decision != "undetermined" or self.evidence_alternatives:
                raise ValueError("unavailable evidence requires undetermined decision and no norms")
            if not self.evidence_unavailable_reason or not self.evidence_unavailable_reason.strip():
                raise ValueError("unavailable evidence requires an explanation")
        elif not self.evidence_alternatives or self.evidence_unavailable_reason is not None:
            raise ValueError("supported evidence requires norms and no unavailability explanation")
        return self


class Label(Strict):
    answerability: Answerability
    issues: list[IssueLabel] = Field(min_length=1)
    missing_facts: list[str]
    missing_authorities: list[str]

    @model_validator(mode="after")
    def unique_issues(self) -> Self:
        if len({i.id for i in self.issues}) != len(self.issues):
            raise ValueError("duplicate issue labels")
        return self


class Review(Strict):
    reviewer: str = Field(min_length=1)
    qualification: str = Field(min_length=1)
    completed_on: date
    independent: Literal[True]
    label: Label


class Adjudication(Strict):
    reviewer: str = Field(min_length=1)
    completed_on: date
    resolution: str = Field(min_length=1)
    label: Label


class Issue(Strict):
    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    proposition: str = Field(min_length=1)
    mandatory: bool = True


class Source(Strict):
    url: str = Field(pattern=r"^https?://")
    published_on: date
    inspected_on: date
    kind: Literal["public_consultation", "synthetic", "court"]
    adaptation: str = Field(min_length=1)


class Case(Strict):
    id: str = Field(min_length=1)
    family: str = Field(min_length=1)
    source_group: str = Field(min_length=1)
    split: Literal["development", "test"]
    legal_as_of: date
    question: str = Field(min_length=1, max_length=8000)
    facts: list[str]
    issues: list[Issue] = Field(min_length=1)
    source: Source
    exposure: list[str]
    candidate_articles: list[str]
    machine_draft: Label | None = None
    reviews: list[Review] = Field(default_factory=list, max_length=2)
    adjudication: Adjudication | None = None

    @model_validator(mode="after")
    def consistency(self) -> Self:
        ids = {i.id for i in self.issues}
        if len(ids) != len(self.issues):
            raise ValueError("duplicate issues")
        labels = [r.label for r in self.reviews]
        if self.machine_draft:
            labels.append(self.machine_draft)
        if self.adjudication:
            if len(self.reviews) != 2:
                raise ValueError("adjudication needs two independent reviews")
            labels.append(self.adjudication.label)
        if len({r.reviewer for r in self.reviews}) != len(self.reviews):
            raise ValueError("two reviews by the same person")
        if any({i.id for i in label.issues} != ids for label in labels):
            raise ValueError("label issue IDs differ from the case")
        if self.split == "test" and self.exposure:
            raise ValueError("exposed cases cannot be blind test")
        return self

    @property
    def gold(self) -> Label | None:
        return self.adjudication.label if self.adjudication else None


class Dataset(Strict):
    version: str
    scope: str
    cases: list[Case] = Field(min_length=1)

    @model_validator(mode="after")
    def split_integrity(self) -> Self:
        if len({c.id for c in self.cases}) != len(self.cases):
            raise ValueError("duplicate case IDs")
        for field in ("family", "source_group"):
            seen: dict[str, str] = {}
            for case in self.cases:
                group = getattr(case, field)
                if group in seen and seen[group] != case.split:
                    raise ValueError(f"{field} leaks across splits: {group}")
                seen[group] = case.split
        if any(c.split == "test" and c.gold is None for c in self.cases):
            raise ValueError("test requires adjudicated labels")
        return self


def fingerprint(value: dict | list) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def write_json(path: Path, value: dict | list) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def annotation_packet(dataset: Dataset, reviewer: str) -> dict:
    """Без ответов источника, подсказок норм и машинного якоря; источник открыт по ссылке."""
    return {
        "dataset_sha256": fingerprint(dataset.model_dump(mode="json")),
        "reviewer": reviewer,
        "qualification": "",
        "instructions": "Работайте независимо. Сначала собственный вывод, затем сверка источника. "
        "Заполните Label по schema.json. Сохраните редакции, точные цитаты и хеши. "
        "Не обсуждайте решения до сдачи обоих пакетов.",
        "cases": [
            {
                "case_id": c.id,
                "legal_as_of": c.legal_as_of.isoformat(),
                "question": c.question,
                "facts": c.facts,
                "issues": [i.model_dump() for i in c.issues],
                "source": c.source.model_dump(mode="json"),
                "label": None,
                "completed_on": None,
                "independent": None,
            }
            for c in dataset.cases
        ],
    }


def agreement(dataset: Dataset) -> dict:
    pairs = []
    disagreements = []
    for case in dataset.cases:
        if len(case.reviews) != 2:
            continue
        a, b = (r.label for r in case.reviews)
        left = {"answerability": a.answerability, **{i.id: i.decision for i in a.issues}}
        right = {"answerability": b.answerability, **{i.id: i.decision for i in b.issues}}
        for key in left:
            pairs.append((key == "answerability", left[key], right[key]))
            if left[key] != right[key]:
                disagreements.append(
                    {"case": case.id, "field": key, "a": left[key], "b": right[key]}
                )
        # Условия и нормы также требуют разрешения, даже при совпадении класса.
        if a != b:
            disagreements.append(
                {"case": case.id, "field": "full_label", "needs_adjudication": True}
            )
    result = {
        "disagreements": disagreements,
        "reviewed_cases": sum(len(c.reviews) == 2 for c in dataset.cases),
    }
    for flag, name in ((True, "answerability"), (False, "decision")):
        selected = [(a, b) for kind, a, b in pairs if kind == flag]
        n = len(selected)
        observed = sum(a == b for a, b in selected) / n if n else None
        labels = {v for pair in selected for v in pair}
        chance = (
            sum(
                sum(a == v for a, _ in selected) * sum(b == v for _, b in selected) / n**2
                for v in labels
            )
            if n
            else None
        )
        result[name] = {
            "n": n,
            "agreement": observed,
            "cohen_kappa": (observed - chance) / (1 - chance) if n and chance < 1 else None,
        }
    return result


def merge_reviews(
    dataset: Dataset, packets: list[dict], adjudications: dict | None = None
) -> Dataset:
    """Импорт после независимой сдачи. Исходные пакеты сохраняет организатор."""
    expected = fingerprint(dataset.model_dump(mode="json"))
    original_entries = {
        entry["case_id"]: entry for entry in annotation_packet(dataset, "")["cases"]
    }
    cases = {c.id: c.model_dump(mode="json") for c in dataset.cases}
    if any(c["reviews"] or c["adjudication"] for c in cases.values()):
        raise ValueError("import into the original unreviewed dataset only")
    for packet in packets:
        if packet["dataset_sha256"] != expected:
            raise ValueError("review packet belongs to another dataset")
        entries = packet["cases"]
        if len(entries) != len(cases) or {e["case_id"] for e in entries} != set(cases):
            raise ValueError("review case coverage mismatch")
        for entry in entries:
            original = original_entries[entry["case_id"]]
            immutable = ("legal_as_of", "question", "facts", "issues", "source")
            if any(entry.get(field) != original[field] for field in immutable):
                raise ValueError("review packet changes immutable case fields")
            review = Review(
                reviewer=packet["reviewer"],
                qualification=packet["qualification"],
                completed_on=entry["completed_on"],
                independent=entry["independent"],
                label=entry["label"],
            )
            cases[entry["case_id"]]["reviews"].append(review.model_dump(mode="json"))
    if adjudications:
        if adjudications["dataset_sha256"] != expected:
            raise ValueError("adjudications belong to another dataset")
        for case_id, value in adjudications["cases"].items():
            cases[case_id]["adjudication"] = Adjudication.model_validate(value).model_dump(
                mode="json"
            )
    return Dataset.model_validate({**dataset.model_dump(), "cases": list(cases.values())})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--packet", type=Path)
    parser.add_argument("--reviews", type=Path, nargs=2)
    parser.add_argument("--adjudications", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    dataset = Dataset.model_validate_json(args.dataset.read_bytes())
    if args.reviews:
        if not args.out or args.out.exists():
            parser.error("--reviews requires a new --out path")
        dataset = merge_reviews(
            dataset,
            [json.loads(p.read_text("utf-8")) for p in args.reviews],
            json.loads(args.adjudications.read_text("utf-8")) if args.adjudications else None,
        )
        write_json(args.out, dataset.model_dump(mode="json"))
    if args.packet:
        args.packet.mkdir(parents=True, exist_ok=False)
        for reviewer in ("reviewer-a", "reviewer-b"):
            write_json(args.packet / f"{reviewer}.json", annotation_packet(dataset, reviewer))
        write_json(args.packet / "schema.json", Label.model_json_schema())
        write_json(args.packet / "agreement.json", agreement(dataset))
    print(
        json.dumps(
            {
                "cases": len(dataset.cases),
                "adjudicated": sum(c.gold is not None for c in dataset.cases),
                "sha256": fingerprint(dataset.model_dump(mode="json")),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
