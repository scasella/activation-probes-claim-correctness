from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from interp_experiment.io import read_jsonl, write_json
from interp_experiment.schemas import ClaimFeatureRow


def _load_labels(path: Path) -> dict[str, dict[str, Any]]:
    return {row["claim_id"]: row for row in read_jsonl(path)}


def _load_features(path: Path) -> tuple[list[str], np.ndarray]:
    rows = [ClaimFeatureRow.from_dict(row) for row in read_jsonl(path)]
    return [row.claim_id for row in rows], np.asarray([row.vector for row in rows], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize a fixed residual correctness-probe direction from frozen MAUD features and v1 labels."
    )
    parser.add_argument("--labels-jsonl", type=Path, default=Path("data/annotations/maud_full_judge_annotations.jsonl"))
    parser.add_argument("--residual-features-jsonl", type=Path, default=Path("artifacts/runs/maud_full_probe_features_residual.jsonl"))
    parser.add_argument("--output-npz", type=Path, default=Path("artifacts/runs/residual_correctness_probe_direction.npz"))
    parser.add_argument("--summary-json", type=Path, default=Path("artifacts/runs/residual_correctness_probe_direction_summary.json"))
    parser.add_argument("--c-value", type=float, default=1.0)
    args = parser.parse_args()

    labels = _load_labels(args.labels_jsonl)
    claim_ids, x = _load_features(args.residual_features_jsonl)
    shared_idx = [idx for idx, claim_id in enumerate(claim_ids) if claim_id in labels]
    shared_claim_ids = [claim_ids[idx] for idx in shared_idx]
    x_shared = x[shared_idx]
    y = np.asarray([1 if labels[claim_id]["correctness_label"] == "true" else 0 for claim_id in shared_claim_ids], dtype=np.int64)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=args.c_value, solver="liblinear", max_iter=5000),
    )
    model.fit(x_shared, y)
    scaler: StandardScaler = model.named_steps["standardscaler"]
    logistic: LogisticRegression = model.named_steps["logisticregression"]

    scale = np.asarray(scaler.scale_, dtype=np.float64)
    scale[scale == 0.0] = 1.0
    standardized_coef = np.asarray(logistic.coef_[0], dtype=np.float64)
    residual_direction = standardized_coef / scale
    residual_intercept = float(logistic.intercept_[0] - np.dot(standardized_coef, scaler.mean_ / scale))
    decision_from_pipeline = model.decision_function(x_shared)
    decision_from_direction = x_shared @ residual_direction + residual_intercept
    max_abs_error = float(np.max(np.abs(decision_from_pipeline - decision_from_direction)))

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        claim_ids=np.asarray(shared_claim_ids),
        labels=y,
        residual_direction=residual_direction.astype(np.float32),
        residual_intercept=np.asarray([residual_intercept], dtype=np.float32),
        standardized_coef=standardized_coef.astype(np.float32),
        scaler_mean=np.asarray(scaler.mean_, dtype=np.float32),
        scaler_scale=np.asarray(scale, dtype=np.float32),
        c_value=np.asarray([args.c_value], dtype=np.float32),
    )
    summary = {
        "source": "deterministic_full_corpus_materialization_from_frozen_v1_labels",
        "note": (
            "No persisted residual-probe pickle existed in the repo. This artifact fixes a single "
            "interpretation direction by fitting the same C=1.0 standardized logistic probe used "
            "by the full-corpus residual signal check, on the frozen v1 label surface."
        ),
        "n_claims": len(shared_claim_ids),
        "n_features": int(x_shared.shape[1]),
        "positive_count": int(y.sum()),
        "negative_count": int((1 - y).sum()),
        "c_value": args.c_value,
        "direction_l2_norm": float(np.linalg.norm(residual_direction)),
        "standardized_coef_l2_norm": float(np.linalg.norm(standardized_coef)),
        "manual_decision_max_abs_error": max_abs_error,
    }
    write_json(args.summary_json, summary)
    print(f"Wrote residual probe direction to {args.output_npz}")
    print(f"Wrote residual probe summary to {args.summary_json}")


if __name__ == "__main__":
    main()
