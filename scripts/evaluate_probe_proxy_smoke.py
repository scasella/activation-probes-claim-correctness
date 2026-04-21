from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from interp_experiment.evaluation.metrics import auroc, brier_score
from interp_experiment.io import read_jsonl, write_json
from interp_experiment.schemas import ClaimFeatureRow
from interp_experiment.utils import ensure_parent


def _labels_from_rows(path: Path) -> dict[str, dict[str, str]]:
    return {
        row["claim_id"]: row
        for row in read_jsonl(path)
    }


def _load_feature_arrays(path: Path) -> tuple[list[str], np.ndarray]:
    if path.suffix.lower() == ".npz":
        payload = np.load(path, allow_pickle=False)
        claim_ids = [str(item) for item in payload["claim_ids"].tolist()]
        matrix = np.asarray(payload["matrix"], dtype=float)
        return claim_ids, matrix
    feature_rows = [ClaimFeatureRow.from_dict(row) for row in read_jsonl(path)]
    claim_ids = [row.claim_id for row in feature_rows]
    matrix = np.asarray([row.vector for row in feature_rows], dtype=float)
    return claim_ids, matrix


def _evaluate_feature_file(path: Path, labels: dict[str, dict[str, str]]) -> dict[str, object]:
    claim_ids, x_all = _load_feature_arrays(path)
    shared_idx = [idx for idx, claim_id in enumerate(claim_ids) if claim_id in labels]
    shared_claim_ids = [claim_ids[idx] for idx in shared_idx]
    y = np.asarray([1 if labels[claim_id]["correctness_label"] == "true" else 0 for claim_id in shared_claim_ids], dtype=int)
    x = x_all[shared_idx]
    if len(set(y.tolist())) < 2:
        return {
            "n_claims": len(shared_claim_ids),
            "correctness": {"auroc": None, "brier": None},
            "note": "single_class_labels",
        }
    loo = LeaveOneOut()
    probs = np.zeros(len(shared_claim_ids), dtype=float)
    for train_idx, test_idx in loo.split(x):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, solver="liblinear", max_iter=5000),
        )
        model.fit(x[train_idx], y[train_idx])
        probs[test_idx[0]] = model.predict_proba(x[test_idx])[0, 1]
    return {
        "n_claims": len(shared_claim_ids),
        "correctness": {
            "auroc": auroc(y.tolist(), probs.tolist()),
            "brier": brier_score(y.tolist(), probs.tolist()),
        },
        "note": "leave_one_out_binary_true_vs_not_true",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a proxy-only probe smoke evaluation on MAUD feature files.")
    parser.add_argument("--labels-jsonl", type=Path, required=True)
    parser.add_argument("--residual-features-jsonl", type=Path, required=True)
    parser.add_argument("--sae-features-path", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    labels = _labels_from_rows(args.labels_jsonl)
    payload = {
        "label_source": "judge_llm_proxy",
        "residual": _evaluate_feature_file(args.residual_features_jsonl, labels),
        "sae": _evaluate_feature_file(args.sae_features_path, labels),
        "proxy_only": True,
    }
    ensure_parent(args.output_json)
    write_json(args.output_json, payload)
    print(f"Wrote probe proxy smoke summary to {args.output_json}")


if __name__ == "__main__":
    main()
