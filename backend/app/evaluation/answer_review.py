"""Импорт независимых оценок ответов; ключи и хеши защищают от смешения прогонов."""

import argparse
import copy
import json
from pathlib import Path

from app.evaluation.answer_dataset import Dataset, fingerprint, write_json
from app.evaluation.answer_metrics import HumanGrade, paired_family_bootstrap, summarize


def apply_grades(report: dict, dataset: Dataset, key: dict, grades: list[dict]) -> dict:
    if key["report_sha256"] != fingerprint(report):
        raise ValueError("organizer key belongs to another report")
    if report["dataset_sha256"] != fingerprint(dataset.model_dump(mode="json")):
        raise ValueError("dataset differs from the run")
    result = copy.deepcopy(report)
    cases = {c.id: c for c in dataset.cases}
    seen = set()
    for raw in grades:
        grade = HumanGrade.model_validate(raw)
        if grade.response_id in seen:
            raise ValueError("duplicate response grade")
        seen.add(grade.response_id)
        binding = key["mapping"][grade.response_id]
        row = result["rows"][binding["row_index"]]
        if row["status"] != "ok":
            raise ValueError("cannot overwrite operational failures")
        if not (fingerprint(row["answer"]) == binding["response_sha256"] == grade.response_sha256):
            raise ValueError("answer hash mismatch")
        case = cases[row["case"]]
        # Эталон случая должен быть принят ДО прогона. Нельзя задним числом
        # узаконить просмотренный ответ модели как независимый эталон.
        if case.gold is None:
            raise ValueError("formal grading requires adjudicated gold frozen before the run")
        if {i.id for i in grade.issues} != {i.id for i in case.issues}:
            raise ValueError("grade issue coverage mismatch")
        if row["variant"] == "B0" and grade.citations_correct is not None:
            raise ValueError("external-source groundedness for B0 must be null")
        row["solution_pass"] = grade.solution_pass({i.id for i in case.issues if i.mandatory})
        row["grade_status"] = "human_reviewed"
        row["human_grade"] = grade.model_dump()
    result["summary"] = summarize(result["rows"])
    if {"B2", "B3"} <= set(result["variants"]):
        result["primary_contrast_B3_minus_B2"] = paired_family_bootstrap(result["rows"], "B2", "B3")
    result["parent_report_sha256"] = fingerprint(report)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("grades", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output already exists")
    report = json.loads((args.run / "results.json").read_text("utf-8"))
    dataset = Dataset.model_validate_json((args.run / "dataset.json").read_bytes())
    key = json.loads((args.run / "organizer-key.json").read_text("utf-8"))
    grades = json.loads(args.grades.read_text("utf-8"))
    write_json(args.out, apply_grades(report, dataset, key, grades))


if __name__ == "__main__":
    main()
