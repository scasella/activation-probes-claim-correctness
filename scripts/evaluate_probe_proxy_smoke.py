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


def _evaluate_feature_file(
    path: Path,
    eval_labels: dict[str, dict[str, str]],
    fit_labels: dict[str, dict[str, str]] | None = None,
) -> dict[str, object]:
    fit_labels = fit_labels or eval_labels
    claim_ids, x_all = _load_feature_arrays(path)
    shared_idx = [
        idx
        for idx, claim_id in enumerate(claim_ids)
        if claim_id in eval_labels and claim_id in fit_labels
    ]
    shared_claim_ids = [claim_ids[idx] for idx in shared_idx]
    y_eval = np.asarray([1 if eval_labels[claim_id]["correctness_label"] == "true" else 0 for claim_id in shared_claim_ids], dtype=int)
    y_fit = np.asarray([1 if fit_labels[claim_id]["correctness_label"] == "true" else 0 for claim_id in shared_claim_ids], dtype=int)
    x = x_all[shared_idx]
    if len(set(y_eval.tolist())) < 2:
        return {
            "n_claims": len(shared_claim_ids),
            "correctness": {"auroc": None, "brier": None},
            "note": "single_class_eval_labels",
        }
    if len(set(y_fit.tolist())) < 2:
        return {
            "n_claims": len(shared_claim_ids),
            "correctness": {"auroc": None, "brier": None},
            "note": "single_class_fit_labels",
        }
    loo = LeaveOneOut()
    probs = np.zeros(len(shared_claim_ids), dtype=float)
    for train_idx, test_idx in loo.split(x):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, solver="liblinear", max_iter=5000),
        )
        model.fit(x[train_idx], y_fit[train_idx])
        probs[test_idx[0]] = model.predict_proba(x[test_idx])[0, 1]
    return {
        "n_claims": len(shared_claim_ids),
        "correctness": {
            "auroc": auroc(y_eval.tolist(), probs.tolist()),
            "brier": brier_score(y_eval.tolist(), probs.tolist()),
        },
        "note": "leave_one_out_binary_true_vs_not_true",
        "fit_label_source": "eval_labels" if fit_labels is eval_labels else "separate_fit_labels",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a proxy-only probe smoke evaluation on MAUD feature files.")
    parser.add_argument("--labels-jsonl", type=Path, required=True)
    parser.add_argument(
        "--fit-labels-jsonl",
        type=Path,
        default=None,
        help="Optional labels used only to fit probe classifiers. Use this for second-judge sensitivity so v2 labels are never used for probe fitting.",
    )
    parser.add_argument("--residual-features-jsonl", type=Path, required=True)
    parser.add_argument("--sae-features-path", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    labels = _labels_from_rows(args.labels_jsonl)
    fit_labels = _labels_from_rows(args.fit_labels_jsonl) if args.fit_labels_jsonl else labels
    payload = {
        "label_source": "judge_llm_proxy",
        "fit_label_source": "judge_llm_proxy_v1" if args.fit_labels_jsonl else "same_as_eval_labels",
        "residual": _evaluate_feature_file(args.residual_features_jsonl, labels, fit_labels),
        "sae": _evaluate_feature_file(args.sae_features_path, labels, fit_labels),
        "proxy_only": True,
    }
    ensure_parent(args.output_json)
    write_json(args.output_json, payload)
    print(f"Wrote probe proxy smoke summary to {args.output_json}")


if __name__ == "__main__":
    main()
