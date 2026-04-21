from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..io import read_csv, read_jsonl

VALID_CORRECTNESS = {"true", "false", "partially_true"}
VALID_LOAD_BEARING = {"yes", "no"}
REQUIRED_PACKET_FIELDS = {
    "annotator_id",
    "annotation_version",
    "example_id",
    "source_corpus",
    "task_family",
    "contract_id",
    "question_text",
    "excerpt_text",
    "llama_answer_text",
    "claim_id",
    "claim_text",
    "correctness_label",
    "load_bearing_label",
    "flip_evidence_text",
    "notes",
}


def _require_packet_field(row: dict[str, Any], field_name: str) -> str:
    value = row.get(field_name)
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value


def validate_annotation_row(row: dict[str, Any]) -> dict[str, Any]:
    missing = REQUIRED_PACKET_FIELDS - set(row)
    if missing:
        raise ValueError(f"annotation row missing fields: {sorted(missing)}")
    validated = dict(row)
    for field_name in REQUIRED_PACKET_FIELDS:
        validated[field_name] = _require_packet_field(validated, field_name)
    correctness_label = validated["correctness_label"].strip()
    load_bearing_label = validated["load_bearing_label"].strip()
    if correctness_label and correctness_label not in VALID_CORRECTNESS:
        raise ValueError(f"invalid correctness_label: {correctness_label}")
    if load_bearing_label and load_bearing_label not in VALID_LOAD_BEARING:
        raise ValueError(f"invalid load_bearing_label: {load_bearing_label}")
    if load_bearing_label == "yes" and not validated["flip_evidence_text"].strip():
        raise ValueError("flip_evidence_text is required when load_bearing_label=yes")
    return validated


def load_annotation_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if path.suffix.lower() == ".csv":
            source_rows = read_csv(path)
        else:
            source_rows = read_jsonl(path)
        rows.extend(validate_annotation_row(row) for row in source_rows)
    return rows


def split_rows_by_annotator(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["annotator_id"]].append(row)
    return dict(grouped)


def is_completed_annotation_row(row: dict[str, Any]) -> bool:
    correctness_label = row["correctness_label"].strip()
    load_bearing_label = row["load_bearing_label"].strip()
    if correctness_label not in VALID_CORRECTNESS:
        return False
    if load_bearing_label not in VALID_LOAD_BEARING:
        return False
    if load_bearing_label == "yes" and not row["flip_evidence_text"].strip():
        return False
    return True


def summarize_annotation_completion(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_annotator = split_rows_by_annotator(rows)
    completed_by_annotator = {
        annotator_id: sum(is_completed_annotation_row(row) for row in annotator_rows)
        for annotator_id, annotator_rows in by_annotator.items()
    }
    total_rows = len(rows)
    total_completed = sum(is_completed_annotation_row(row) for row in rows)
    return {
        "n_rows": total_rows,
        "n_completed_rows": total_completed,
        "n_incomplete_rows": total_rows - total_completed,
        "annotators": sorted(by_annotator),
        "rows_by_annotator": {annotator_id: len(annotator_rows) for annotator_id, annotator_rows in by_annotator.items()},
        "completed_rows_by_annotator": completed_by_annotator,
    }


def _pair_completed_rows(rows: list[dict[str, Any]]) -> tuple[list[str], dict[str, dict[str, dict[str, Any]]]]:
    by_annotator = split_rows_by_annotator(rows)
    annotators = sorted(by_annotator)
    if len(annotators) != 2:
        raise ValueError(f"expected exactly 2 annotators, found {annotators}")
    paired: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if is_completed_annotation_row(row):
            paired[row["claim_id"]][row["annotator_id"]] = row
    complete_pairs = {
        claim_id: pair
        for claim_id, pair in paired.items()
        if set(pair) == set(annotators)
    }
    return annotators, complete_pairs


def correctness_binary_label(label: str) -> str:
    return "true" if label == "true" else "not_true"


def cohens_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    if len(labels_a) != len(labels_b):
        raise ValueError("Both annotator label lists must have the same length")
    if not labels_a:
        raise ValueError("Need at least one label")
    observed = sum(a == b for a, b in zip(labels_a, labels_b)) / len(labels_a)
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    categories = set(counts_a) | set(counts_b)
    expected = sum((counts_a[c] / len(labels_a)) * (counts_b[c] / len(labels_b)) for c in categories)
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def load_bearing_gate_passed(kappa: float, threshold: float = 0.6) -> bool:
    return kappa >= threshold


def extract_disagreement_examples(rows: list[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    annotators, complete_pairs = _pair_completed_rows(rows)
    disagreements: list[dict[str, Any]] = []
    for claim_id, pair in complete_pairs.items():
        left = pair[annotators[0]]
        right = pair[annotators[1]]
        correctness_disagree = left["correctness_label"] != right["correctness_label"]
        load_disagree = left["load_bearing_label"] != right["load_bearing_label"]
        if not correctness_disagree and not load_disagree:
            continue
        disagreements.append(
            {
                "claim_id": claim_id,
                "example_id": left["example_id"],
                "question_text": left["question_text"],
                "claim_text": left["claim_text"],
                "llama_answer_text": left["llama_answer_text"],
                "annotator_a": {
                    "annotator_id": annotators[0],
                    "correctness_label": left["correctness_label"],
                    "load_bearing_label": left["load_bearing_label"],
                    "flip_evidence_text": left["flip_evidence_text"],
                    "notes": left["notes"],
                },
                "annotator_b": {
                    "annotator_id": annotators[1],
                    "correctness_label": right["correctness_label"],
                    "load_bearing_label": right["load_bearing_label"],
                    "flip_evidence_text": right["flip_evidence_text"],
                    "notes": right["notes"],
                },
                "disagreement_kinds": [
                    kind
                    for kind, flag in (
                        ("correctness", correctness_disagree),
                        ("load_bearing", load_disagree),
                    )
                    if flag
                ],
            }
        )
        if len(disagreements) >= limit:
            break
    return disagreements


def compute_annotation_agreement(
    rows: list[dict[str, Any]],
    *,
    load_bearing_threshold: float = 0.6,
    attempt_index: int = 1,
) -> dict[str, Any]:
    completion = summarize_annotation_completion(rows)
    annotators, complete_pairs = _pair_completed_rows(rows)
    claim_ids = sorted(complete_pairs)
    if not claim_ids:
        return {
            "completion": completion,
            "n_complete_pairs": 0,
            "agreement": {},
            "gate": {
                "status": "incomplete",
                "next_action": "complete_annotations",
                "load_bearing_threshold": load_bearing_threshold,
                "attempt_index": attempt_index,
            },
            "disagreements": [],
        }

    correctness_a = [complete_pairs[claim_id][annotators[0]]["correctness_label"] for claim_id in claim_ids]
    correctness_b = [complete_pairs[claim_id][annotators[1]]["correctness_label"] for claim_id in claim_ids]
    binary_a = [correctness_binary_label(label) for label in correctness_a]
    binary_b = [correctness_binary_label(label) for label in correctness_b]
    load_a = [complete_pairs[claim_id][annotators[0]]["load_bearing_label"] for claim_id in claim_ids]
    load_b = [complete_pairs[claim_id][annotators[1]]["load_bearing_label"] for claim_id in claim_ids]

    correctness_kappa = cohens_kappa(correctness_a, correctness_b)
    correctness_binary_kappa = cohens_kappa(binary_a, binary_b)
    load_bearing_kappa = cohens_kappa(load_a, load_b)
    gate_pass = load_bearing_gate_passed(load_bearing_kappa, threshold=load_bearing_threshold)
    if gate_pass:
        next_action = "proceed_with_phase1"
        status = "pass"
    elif attempt_index <= 1:
        next_action = "revise_guidelines_and_repilot"
        status = "fail_revise_once"
    else:
        next_action = "drop_load_bearing_continue_correctness_only"
        status = "fail_drop_load_bearing"

    return {
        "completion": completion,
        "n_complete_pairs": len(claim_ids),
        "annotators": annotators,
        "agreement": {
            "correctness_kappa": correctness_kappa,
            "correctness_binary_kappa": correctness_binary_kappa,
            "load_bearing_kappa": load_bearing_kappa,
            "label_distribution": {
                "correctness": {
                    annotators[0]: dict(Counter(correctness_a)),
                    annotators[1]: dict(Counter(correctness_b)),
                },
                "load_bearing": {
                    annotators[0]: dict(Counter(load_a)),
                    annotators[1]: dict(Counter(load_b)),
                },
            },
        },
        "gate": {
            "status": status,
            "next_action": next_action,
            "load_bearing_threshold": load_bearing_threshold,
            "attempt_index": attempt_index,
        },
        "disagreements": extract_disagreement_examples(rows),
    }
