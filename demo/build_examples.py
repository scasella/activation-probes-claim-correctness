from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression


DISCLAIMER = (
    "This is a research demo. The probe was trained on Llama 3.1 8B activations during "
    "merger-agreement question answering, supervised on labels from an LLM judge. Flag colors "
    "reflect what the probe learned to associate with judge-assigned correctness on this domain. "
    "Do not use this for legal decisions."
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _sentence_spans(text: str) -> list[dict[str, Any]]:
    pattern = re.compile(r"[^.!?]+(?:[.!?]+|$)", re.MULTILINE)
    spans: list[dict[str, Any]] = []
    for match in pattern.finditer(text):
        start, end = match.span()
        raw = text[start:end]
        stripped = raw.strip()
        if not stripped:
            continue
        left_trim = len(raw) - len(raw.lstrip())
        right_trim = len(raw.rstrip())
        spans.append(
            {
                "text": stripped,
                "char_start": start + left_trim,
                "char_end": start + right_trim,
            }
        )
    return spans or [{"text": text.strip(), "char_start": 0, "char_end": len(text)}]


def _claim_char_span(claim: dict[str, Any], token_offsets: list[list[int] | tuple[int, int]]) -> tuple[int, int] | None:
    token_start = int(claim["token_start"])
    token_end = int(claim["token_end"])
    if token_start < 0 or token_end <= token_start or token_end > len(token_offsets):
        return None
    return int(token_offsets[token_start][0]), int(token_offsets[token_end - 1][1])


def _overlap(left: tuple[int, int], right: tuple[int, int]) -> int:
    return max(0, min(left[1], right[1]) - max(left[0], right[0]))


def _fit_calibrator(
    *,
    examples: dict[str, dict[str, Any]],
    claim_raw_scores: dict[str, float],
    claims: dict[str, dict[str, Any]],
    labels: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    train_claim_ids = [
        claim_id
        for claim_id, claim in claims.items()
        if claim_id in claim_raw_scores
        and claim_id in labels
        and examples[claim["example_id"]]["split"] == "train"
    ]
    x = np.asarray([[claim_raw_scores[claim_id]] for claim_id in train_claim_ids], dtype=float)
    y = np.asarray([1 if labels[claim_id]["correctness_label"] == "true" else 0 for claim_id in train_claim_ids], dtype=int)
    model = LogisticRegression(C=1.0, solver="liblinear", max_iter=5000)
    model.fit(x, y)
    coef = float(model.coef_[0][0])
    intercept = float(model.intercept_[0])
    return {
        "method": "Platt scaling on methods-paper train split raw residual-probe logits",
        "n_train_claims": len(train_claim_ids),
        "positive_train_claims": int(y.sum()),
        "negative_train_claims": int((1 - y).sum()),
        "coef": coef,
        "intercept": intercept,
    }


def _calibrated(raw_score: float, calibration: dict[str, Any]) -> float:
    return _sigmoid(float(calibration["coef"]) * raw_score + float(calibration["intercept"]))


def _pick_examples(candidates: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def add(candidate: dict[str, Any] | None) -> None:
        if candidate is None or candidate["example_id"] in selected_ids or len(selected) >= count:
            return
        selected.append(candidate)
        selected_ids.add(candidate["example_id"])

    low = sorted(candidates, key=lambda row: (row["mean"], row["example_id"]))
    high = sorted(candidates, key=lambda row: (-row["mean"], row["example_id"]))
    varied = sorted(candidates, key=lambda row: (-row["range"], row["example_id"]))
    mid = sorted(candidates, key=lambda row: (abs(row["mean"] - 0.5), -row["range"], row["example_id"]))

    for bucket in (varied[:3], low[:2], high[:2], mid):
        for candidate in bucket:
            add(candidate)
            if len(selected) >= count:
                return selected
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Build static examples for the claim-level probe uncertainty demo.")
    parser.add_argument("--examples-jsonl", type=Path, default=Path("artifacts/runs/maud_full_examples.jsonl"))
    parser.add_argument("--answer-runs-jsonl", type=Path, default=Path("artifacts/runs/maud_full_answer_runs.jsonl"))
    parser.add_argument("--claims-jsonl", type=Path, default=Path("artifacts/runs/maud_full_claims.jsonl"))
    parser.add_argument("--residual-features-jsonl", type=Path, default=Path("artifacts/runs/maud_full_probe_features_residual.jsonl"))
    parser.add_argument("--labels-jsonl", type=Path, default=Path("data/annotations/maud_full_judge_annotations.jsonl"))
    parser.add_argument("--probe-npz", type=Path, default=Path("artifacts/runs/residual_correctness_probe_direction.npz"))
    parser.add_argument("--output-json", type=Path, default=Path("demo/web/examples.json"))
    parser.add_argument("--calibration-json", type=Path, default=Path("demo/probe_calibration.json"))
    parser.add_argument("--n-examples", type=int, default=8)
    args = parser.parse_args()

    examples = {row["example_id"]: row for row in _read_jsonl(args.examples_jsonl)}
    answer_runs = {row["example_id"]: row for row in _read_jsonl(args.answer_runs_jsonl)}
    claims = {row["claim_id"]: row for row in _read_jsonl(args.claims_jsonl)}
    labels = {row["claim_id"]: row for row in _read_jsonl(args.labels_jsonl)}
    features = {row["claim_id"]: row for row in _read_jsonl(args.residual_features_jsonl)}

    probe = np.load(args.probe_npz, allow_pickle=False)
    direction = np.asarray(probe["residual_direction"], dtype=np.float64)
    intercept = float(probe["residual_intercept"][0])
    claim_raw_scores = {
        claim_id: float(np.asarray(row["vector"], dtype=np.float64) @ direction + intercept)
        for claim_id, row in features.items()
    }
    calibration = _fit_calibrator(
        examples=examples,
        claim_raw_scores=claim_raw_scores,
        claims=claims,
        labels=labels,
    )
    _write_json(args.calibration_json, calibration)

    claims_by_example: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for claim_id, claim in claims.items():
        if claim_id in claim_raw_scores:
            claims_by_example[claim["example_id"]].append(claim)

    candidates: list[dict[str, Any]] = []
    rendered_by_id: dict[str, dict[str, Any]] = {}
    for example_id, example in examples.items():
        if example.get("split") == "test" or example_id not in answer_runs:
            continue
        answer_run = answer_runs[example_id]
        token_offsets = answer_run["token_offsets"]
        claim_rows = []
        for claim in claims_by_example.get(example_id, []):
            span = _claim_char_span(claim, token_offsets)
            if span is None:
                continue
            raw = claim_raw_scores[claim["claim_id"]]
            claim_rows.append(
                {
                    "claim_id": claim["claim_id"],
                    "text": claim["claim_text"],
                    "char_start": span[0],
                    "char_end": span[1],
                    "raw_score": raw,
                    "calibrated_score": _calibrated(raw, calibration),
                    "judge_label": labels.get(claim["claim_id"], {}).get("correctness_label"),
                }
            )
        if not claim_rows:
            continue

        sentence_rows = []
        for sentence in _sentence_spans(answer_run["answer_text"]):
            sentence_span = (sentence["char_start"], sentence["char_end"])
            overlapping = [
                (claim, _overlap(sentence_span, (claim["char_start"], claim["char_end"])))
                for claim in claim_rows
            ]
            overlapping = [(claim, weight) for claim, weight in overlapping if weight > 0]
            if overlapping:
                weight_sum = sum(weight for _, weight in overlapping)
                raw_score = sum(claim["raw_score"] * weight for claim, weight in overlapping) / weight_sum
                calibrated_score = sum(claim["calibrated_score"] * weight for claim, weight in overlapping) / weight_sum
                score_source = "overlapping_claim_spans"
                claim_ids = [claim["claim_id"] for claim, _ in overlapping]
            else:
                raw_score = float(np.mean([claim["raw_score"] for claim in claim_rows]))
                calibrated_score = float(np.mean([claim["calibrated_score"] for claim in claim_rows]))
                score_source = "example_mean_no_claim_overlap"
                claim_ids = [claim["claim_id"] for claim in claim_rows]
            sentence_rows.append(
                {
                    **sentence,
                    "raw_score": raw_score,
                    "calibrated_score": calibrated_score,
                    "claim_ids": claim_ids,
                    "score_source": score_source,
                }
            )
        sentence_scores = [row["calibrated_score"] for row in sentence_rows]
        rendered = {
            "example_id": example_id,
            "maud_source_id": example_id,
            "source_corpus": example["source_corpus"],
            "contract_id": example["contract_id"],
            "contract_group": example["contract_group"],
            "split": example["split"],
            "question": example["question_text"],
            "answer": answer_run["answer_text"],
            "excerpt_preview": example["excerpt_text"][:900],
            "score_summary": {
                "mean": float(np.mean(sentence_scores)),
                "min": float(np.min(sentence_scores)),
                "max": float(np.max(sentence_scores)),
                "range": float(np.max(sentence_scores) - np.min(sentence_scores)),
            },
            "sentences": sentence_rows,
            "claims": claim_rows,
        }
        rendered_by_id[example_id] = rendered
        candidates.append({"example_id": example_id, **rendered["score_summary"]})

    chosen = _pick_examples(candidates, args.n_examples)
    output_examples = [rendered_by_id[row["example_id"]] for row in chosen]
    payload = {
        "metadata": {
            "generated_at": datetime.now(UTC).isoformat(),
            "demo_version": "v1",
            "disclaimer": DISCLAIMER,
            "score_explanation": (
                "Sentence colors are calibrated residual-probe scores aggregated from overlapping "
                "frozen claim spans. Lower scores are rendered redder; higher scores are rendered greener."
            ),
            "calibration": calibration,
            "n_examples": len(output_examples),
            "selection_rule": "Deterministic variety sample from non-test MAUD examples: high-range, low-score, high-score, and mid-score cases.",
            "inputs": {
                "examples_jsonl": str(args.examples_jsonl),
                "answer_runs_jsonl": str(args.answer_runs_jsonl),
                "claims_jsonl": str(args.claims_jsonl),
                "residual_features_jsonl": str(args.residual_features_jsonl),
                "labels_jsonl": str(args.labels_jsonl),
                "probe_npz": str(args.probe_npz),
            },
        },
        "examples": output_examples,
    }
    _write_json(args.output_json, payload)
    print(f"Wrote {len(output_examples)} demo examples to {args.output_json}")
    print(f"Wrote calibration to {args.calibration_json}")


if __name__ == "__main__":
    main()
