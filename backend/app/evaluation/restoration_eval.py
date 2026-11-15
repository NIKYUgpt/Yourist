"""Синтетический пилот D: детекция отдельно от идентичности и восстановления."""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import time
from dataclasses import asdict
from itertools import combinations
from pathlib import Path

from app.evaluation.answer_dataset import fingerprint, write_json
from app.evaluation.answer_metrics import nearest_rank
from app.modules.deid import names, rules
from app.modules.deid.restoration import restore_grammar
from app.modules.deid.service import _PLACEHOLDER, _select, deidentify, reidentify


def outside_unchanged(masked: str, restored: str, mapping: dict[str, str]) -> bool:
    """Проверка через явно разрешённые значения и формы, без wildcard для правового текста."""
    import re

    from app.modules.deid.restoration import CASES, inflect_person

    pieces, pos = [], 0
    for m in _PLACEHOLDER.finditer(masked):
        pieces.append(re.escape(masked[pos : m.start()]))
        values = {mapping.get(m.group(), m.group())}
        if m.group().startswith("[ФИО_") and m.group() in mapping:
            values.update(v for case in CASES if (v := inflect_person(mapping[m.group()], case)))
        pieces.append("(?:" + "|".join(re.escape(v) for v in sorted(values)) + ")")
        pos = m.end()
    pieces.append(re.escape(masked[pos:]))
    return re.fullmatch("".join(pieces), restored) is not None


async def evaluate(data: dict) -> dict:
    result = {
        "dataset_sha256": fingerprint(data),
        "status": data["status"],
        "packages": {p: importlib.metadata.version(p) for p in ("pymorphy3", "pymorphy3-dicts-ru")},
        "code_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                Path("app/modules/deid/restoration.py"),
                Path("app/modules/deid/service.py"),
                Path("app/modules/deid/names.py"),
            ]
        },
        "restoration": [],
        "masking": [],
    }
    for case in data["restoration"]:
        start = time.perf_counter()
        baseline = reidentify(case["masked"], case["mapping"])
        baseline_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        corrected = restore_grammar(case["masked"], case["mapping"])
        corrected_ms = (time.perf_counter() - start) * 1000
        result["restoration"].append(
            {
                "id": case["id"],
                "baseline": baseline,
                "corrected": asdict(corrected),
                "expected": case["expected"],
                "baseline_exact": baseline == case["expected"]
                if case["expected"] is not None
                else None,
                "corrected_exact": corrected.text == case["expected"]
                if case["expected"] is not None
                else None,
                "outside_unchanged": outside_unchanged(
                    case["masked"], corrected.text, case["mapping"]
                ),
                "baseline_ms": baseline_ms,
                "corrected_ms": corrected_ms,
            }
        )
    for case in data["masking"]:
        expected, last = {}, {}
        for item in case["entities"]:
            start = case["text"].index(item["text"], last.get(item["text"], 0))
            end = start + len(item["text"])
            last[item["text"]] = end
            expected[start, end, "ФИО"] = item["person"]
        found = _select([*rules.find(case["text"]), *names.find(case["text"])])
        masked = await deidentify(case["text"])
        labels = [m.group() for m in _PLACEHOLDER.finditer(masked.text)]
        if len(found) != len(labels):
            raise ValueError("detection/label alignment failed")
        predicted = {
            (e.start, e.end, e.type): label for e, label in zip(found, labels, strict=True)
        }
        shared = predicted.keys() & expected.keys()
        merges = splits = 0
        for a, b in combinations(sorted(shared), 2):
            merges += expected[a] != expected[b] and predicted[a] == predicted[b]
            splits += expected[a] == expected[b] and predicted[a] != predicted[b]
        result["masking"].append(
            {
                "id": case["id"],
                "masked": masked.text,
                "tp": len(shared),
                "fp": len(predicted.keys() - expected.keys()),
                "fn": len(expected.keys() - predicted.keys()),
                "identity_merges": merges,
                "identity_splits": splits,
                "identity_pairs_evaluated": len(shared) * (len(shared) - 1) // 2,
            }
        )
    restoration, masking = result["restoration"], result["masking"]
    tp, fp, fn = (sum(r[k] for r in masking) for k in ("tp", "fp", "fn"))
    result["summary"] = {
        "restoration_cases": len(restoration),
        "grammar_cases": sum(r["expected"] is not None for r in restoration),
        "baseline_exact": sum(r["baseline_exact"] is True for r in restoration),
        "corrected_exact": sum(r["corrected_exact"] is True for r in restoration),
        "outside_unchanged": sum(r["outside_unchanged"] for r in restoration),
        "corrected_p90_ms": nearest_rank([r["corrected_ms"] for r in restoration], 0.9),
        "detection_tp": tp,
        "detection_fp": fp,
        "detection_fn": fn,
        "detection_precision": tp / (tp + fp) if tp + fp else None,
        "detection_recall": tp / (tp + fn) if tp + fn else None,
        "identity_merges": sum(r["identity_merges"] for r in masking),
        "identity_splits": sum(r["identity_splits"] for r in masking),
        "limitation": (
            "authored synthetic development; no population inference; rules/dictionary only"
        ),
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    data = json.loads(args.dataset.read_text("utf-8"))
    result = asyncio.run(evaluate(data))
    write_json(args.out / "results.json", result)
    write_json(args.out / "dataset.json", data)
    print(json.dumps(result["summary"], ensure_ascii=False))
