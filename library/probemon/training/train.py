from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from probemon.core.activations import HuggingFaceActivationExtractor, mean_pool_char_span
from probemon.core.probe import Probe
from probemon.training.dataset import CanonicalDataset


@dataclass(slots=True)
class FitResult:
    probe: Probe
    selected_c: float
    validation_auroc: float
    validation_brier: float
    dataset_stats: dict[str, Any]


def _label_to_binary(label: str) -> int:
    return 1 if label == "correct" else 0


def _example_split(metadata: dict[str, Any]) -> str:
    return str(metadata.get("split", "train"))


def _extract_features(dataset: CanonicalDataset, extractor: Any) -> tuple[np.ndarray, list[int], list[str], list[str]]:
    vectors: list[np.ndarray] = []
    labels: list[int] = []
    splits: list[str] = []
    claim_ids: list[str] = []
    for example_index, example in enumerate(dataset.examples):
        encoded = extractor.encode_answer_with_activations(example.prompt, example.generation)
        for claim_index, claim in enumerate(example.claims):
            vectors.append(mean_pool_char_span(encoded.residual_stream, encoded.token_offsets, claim.char_start, claim.char_end))
            labels.append(_label_to_binary(claim.label))
            splits.append(_example_split(example.metadata))
            claim_ids.append(str(claim.metadata.get("claim_id", f"claim-{example_index}-{claim_index}")))
    return np.vstack(vectors), labels, splits, claim_ids


def _raw_space_probe(model: LogisticRegression, scaler: StandardScaler) -> tuple[np.ndarray, float]:
    coef = model.coef_.reshape(-1)
    scale = np.where(scaler.scale_ == 0.0, 1.0, scaler.scale_)
    direction = coef / scale
    bias = float(model.intercept_[0] - np.sum(coef * scaler.mean_ / scale))
    return direction.astype(np.float32), bias


def _fit_platt(logits: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    if len(set(y_true.tolist())) < 2 or np.allclose(logits, logits[0]):
        return 1.0, 0.0
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    calibrator.fit(logits.reshape(-1, 1), y_true)
    return float(calibrator.coef_[0, 0]), float(calibrator.intercept_[0])


def fit_probe(
    *,
    model: Any | None,
    dataset: CanonicalDataset,
    layer: int = 19,
    output_path: str | Path | None = None,
    features: np.ndarray | None = None,
    c_values: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    probe_id: str = "custom-probe",
    metadata: dict[str, Any] | None = None,
) -> FitResult:
    if features is None:
        extractor = model if hasattr(model, "encode_answer_with_activations") else HuggingFaceActivationExtractor(model_name, layer=layer)
        x, labels, splits, claim_ids = _extract_features(dataset, extractor)
    else:
        x = np.asarray(features, dtype=float)
        labels = [_label_to_binary(claim.label) for _, claim in dataset.iter_claims()]
        splits = [_example_split(example.metadata) for example, claim in dataset.iter_claims()]
        claim_ids = [str(claim.metadata.get("claim_id", idx)) for idx, (_, claim) in enumerate(dataset.iter_claims())]
        if x.shape[0] != len(labels):
            raise ValueError(f"features rows ({x.shape[0]}) must match dataset claims ({len(labels)})")

    y = np.asarray(labels, dtype=int)
    train_idx = np.asarray([i for i, split in enumerate(splits) if split == "train"], dtype=int)
    val_idx = np.asarray([i for i, split in enumerate(splits) if split in {"validation", "val"}], dtype=int)
    if train_idx.size == 0 or val_idx.size == 0:
        raise ValueError("fit_probe requires train and validation splits in example metadata")
    rows = []
    best: tuple[float, float, LogisticRegression, StandardScaler] | None = None
    for c_value in c_values:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x[train_idx])
        clf = LogisticRegression(C=float(c_value), solver="liblinear", max_iter=5000)
        clf.fit(x_train, y[train_idx])
        val_probs = clf.predict_proba(scaler.transform(x[val_idx]))[:, 1]
        roc = float(roc_auc_score(y[val_idx], val_probs))
        brier = float(brier_score_loss(y[val_idx], val_probs))
        rows.append({"c": float(c_value), "validation_auroc": roc, "validation_brier": brier})
        if best is None or (roc, -brier) > (best[0], -best[1]):
            best = (roc, brier, clf, scaler)
    assert best is not None
    validation_auroc, validation_brier, clf, scaler = best
    direction, bias = _raw_space_probe(clf, scaler)
    val_logits = x[val_idx] @ direction + bias
    platt_a, platt_b = _fit_platt(val_logits, y[val_idx])
    meta = {
        "probe_id": probe_id,
        "model_name": model_name,
        "model_aliases": ["meta-llama/Meta-Llama-3.1-8B-Instruct"],
        "layer": layer,
        "pooling": "mean_pool_claim_char_span",
        "training_dataset": dataset.name,
        "training_dataset_size": dataset.n_claims,
        "training_label_source": "user_supplied_canonical_labels",
        "validation_auroc": validation_auroc,
        "validation_brier": validation_brier,
        "version": "0.1.0",
        "domain_description": "User-provided canonical claim dataset.",
        "c_sweep": rows,
        "claim_ids": claim_ids,
    }
    if metadata:
        meta.update(metadata)
    probe = Probe(probe_id=probe_id, direction=direction, bias=bias, platt_a=platt_a, platt_b=platt_b, metadata=meta)
    if output_path is not None:
        probe.save(output_path)
    return FitResult(
        probe=probe,
        selected_c=float(clf.C),
        validation_auroc=validation_auroc,
        validation_brier=validation_brier,
        dataset_stats={"n_claims": len(labels), "splits": {split: splits.count(split) for split in sorted(set(splits))}},
    )
