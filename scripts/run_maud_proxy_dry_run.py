from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from interp_experiment.evaluation.metrics import auroc, brier_score
from interp_experiment.io import read_jsonl, write_json
from interp_experiment.schemas import BaselinePrediction, ClaimFeatureRow, ExampleRow
from interp_experiment.utils import ensure_parent


def _load_examples(path: Path) -> dict[str, ExampleRow]:
    return {row.example_id: row for row in (ExampleRow.from_dict(item) for item in read_jsonl(path))}


def _load_labels(path: Path) -> dict[str, dict[str, str]]:
    return {row["claim_id"]: row for row in read_jsonl(path)}


def _load_targets(path: Path) -> dict[str, dict[str, object]]:
    return {row["claim_id"]: row for row in read_jsonl(path)}


def _load_residual_rows(path: Path) -> list[ClaimFeatureRow]:
    return [ClaimFeatureRow.from_dict(row) for row in read_jsonl(path)]


def _load_sae_rows(path: Path) -> list[ClaimFeatureRow]:
    payload = np.load(path, allow_pickle=False)
    claim_ids = [str(item) for item in payload["claim_ids"].tolist()]
    example_ids = [str(item) for item in payload["example_ids"].tolist()]
    matrix = np.asarray(payload["matrix"], dtype=float)
    return [
        ClaimFeatureRow(
            claim_id=claim_id,
            example_id=example_id,
            feature_source="sae",
            vector=matrix[idx].tolist(),
            correctness_target=None,
            load_bearing_target=None,
            stability_target=None,
        ).validate()
        for idx, (claim_id, example_id) in enumerate(zip(claim_ids, example_ids))
    ]


def _merge_rows(
    feature_rows: list[ClaimFeatureRow],
    examples: dict[str, ExampleRow],
    labels: dict[str, dict[str, str]],
    targets: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for feature_row in feature_rows:
        if feature_row.claim_id not in labels or feature_row.claim_id not in targets:
            continue
        label = labels[feature_row.claim_id]
        target = targets[feature_row.claim_id]
        example = examples[feature_row.example_id]
        rows.append(
            {
                "claim_id": feature_row.claim_id,
                "example_id": feature_row.example_id,
                "split": example.split,
                "vector": feature_row.vector,
                "correctness_entropy_target": float(target["correctness_target"]),
                "stability_target": int(target["stability_target"]),
                "correctness_label": label["correctness_label"],
                "load_bearing_label": label["load_bearing_label"],
            }
        )
    return rows


def _split_rows(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    return {
        split: [row for row in rows if row["split"] == split]
        for split in ("train", "validation", "test")
    }


def _ridge_prob(preds_train: np.ndarray, preds_eval: np.ndarray) -> np.ndarray:
    low = float(preds_train.min())
    high = float(preds_train.max())
    if high <= low:
        return np.full_like(preds_eval, 0.5, dtype=float)
    scaled = (preds_eval - low) / (high - low)
    return np.clip(scaled, 0.0, 1.0)


def _evaluate_correctness(rows_by_split: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    train_rows = rows_by_split["train"]
    val_rows = rows_by_split["validation"]
    test_rows = rows_by_split["test"]
    x_train = np.asarray([row["vector"] for row in train_rows], dtype=float)
    y_train_entropy = np.asarray([row["correctness_entropy_target"] for row in train_rows], dtype=float)
    x_val = np.asarray([row["vector"] for row in val_rows], dtype=float)
    y_val_entropy = np.asarray([row["correctness_entropy_target"] for row in val_rows], dtype=float)
    x_test = np.asarray([row["vector"] for row in test_rows], dtype=float)
    y_test_binary = np.asarray([1 if row["correctness_label"] == "true" else 0 for row in test_rows], dtype=int)
    x_trainval = np.asarray([row["vector"] for row in train_rows + val_rows], dtype=float)
    y_trainval_entropy = np.asarray([row["correctness_entropy_target"] for row in train_rows + val_rows], dtype=float)
    y_trainval_binary = np.asarray([1 if row["correctness_label"] == "true" else 0 for row in train_rows + val_rows], dtype=int)

    alpha_grid = [0.1, 1.0, 10.0]
    best_alpha = None
    best_mse = None
    for alpha in alpha_grid:
        model = Ridge(alpha=alpha)
        model.fit(x_train, y_train_entropy)
        preds = model.predict(x_val)
        mse = float(np.mean((preds - y_val_entropy) ** 2))
        if best_mse is None or mse < best_mse:
            best_mse = mse
            best_alpha = alpha
    ridge = Ridge(alpha=float(best_alpha))
    ridge.fit(x_trainval, y_trainval_entropy)
    trainval_preds = ridge.predict(x_trainval)
    test_preds = ridge.predict(x_test)
    ridge_probs = _ridge_prob(trainval_preds, test_preds)

    c_grid = [0.1, 1.0, 10.0]
    best_c = None
    best_brier = None
    for c_value in c_grid:
        model = make_pipeline(StandardScaler(), LogisticRegression(C=c_value, solver="liblinear", max_iter=5000))
        model.fit(x_train, np.asarray([1 if row["correctness_label"] == "true" else 0 for row in train_rows], dtype=int))
        probs = model.predict_proba(x_val)[:, 1]
        brier = brier_score(
            [1 if row["correctness_label"] == "true" else 0 for row in val_rows],
            probs.tolist(),
        )
        if best_brier is None or brier < best_brier:
            best_brier = brier
            best_c = c_value
    logistic = make_pipeline(StandardScaler(), LogisticRegression(C=float(best_c), solver="liblinear", max_iter=5000))
    logistic.fit(x_trainval, y_trainval_binary)
    logistic_probs = logistic.predict_proba(x_test)[:, 1]

    return {
        "ridge_entropy": {
            "selected_alpha": best_alpha,
            "test": {
                "auroc": auroc(y_test_binary.tolist(), ridge_probs.tolist()),
                "brier": brier_score(y_test_binary.tolist(), ridge_probs.tolist()),
            },
        },
        "logistic_ablation": {
            "selected_c": best_c,
            "test": {
                "auroc": auroc(y_test_binary.tolist(), logistic_probs.tolist()),
                "brier": brier_score(y_test_binary.tolist(), logistic_probs.tolist()),
            },
        },
        "n_train": len(train_rows),
        "n_validation": len(val_rows),
        "n_test": len(test_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MAUD proxy dry-run training/evaluation pipeline.")
    parser.add_argument("--examples-jsonl", type=Path, required=True)
    parser.add_argument("--labels-jsonl", type=Path, required=True)
    parser.add_argument("--targets-jsonl", type=Path, required=True)
    parser.add_argument("--residual-features-jsonl", type=Path, required=True)
    parser.add_argument("--sae-features-path", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    examples = _load_examples(args.examples_jsonl)
    labels = _load_labels(args.labels_jsonl)
    targets = _load_targets(args.targets_jsonl)
    residual_rows = _merge_rows(_load_residual_rows(args.residual_features_jsonl), examples, labels, targets)
    sae_rows = _merge_rows(_load_sae_rows(args.sae_features_path), examples, labels, targets)

    payload = {
        "label_source": "judge_llm_proxy",
        "proxy_only": True,
        "residual": _evaluate_correctness(_split_rows(residual_rows)),
        "sae": _evaluate_correctness(_split_rows(sae_rows)),
    }
    ensure_parent(args.output_json)
    write_json(args.output_json, payload)
    print(f"Wrote MAUD proxy dry-run summary to {args.output_json}")


if __name__ == "__main__":
    main()
