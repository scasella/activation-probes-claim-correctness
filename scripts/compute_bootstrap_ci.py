from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from interp_experiment.evaluation.metrics import auroc, brier_score
from interp_experiment.io import read_jsonl, write_json
from interp_experiment.schemas import BaselinePrediction, ClaimFeatureRow


METHOD_LABELS = {
    "llama_self_report": "Llama self-report",
    "gpt54_cached": "GPT-5.4 external scorer",
    "residual_probe": "Residual activation probe",
    "sae_probe": "SAE feature probe",
}
METHOD_ORDER = ["llama_self_report", "gpt54_cached", "residual_probe", "sae_probe"]
LABELS = ["true", "partially_true", "false"]


def _load_labels(path: Path) -> dict[str, dict[str, Any]]:
    return {row["claim_id"]: row for row in read_jsonl(path)}


def _load_claim_order(path: Path) -> list[str]:
    return [row["claim_id"] for row in read_jsonl(path)]


def _load_predictions(path: Path) -> dict[str, BaselinePrediction]:
    rows: list[BaselinePrediction] = []
    if path.is_dir():
        for file_path in sorted(path.glob("*.jsonl")):
            rows.extend(BaselinePrediction.from_dict(row) for row in read_jsonl(file_path))
    else:
        rows.extend(BaselinePrediction.from_dict(row) for row in read_jsonl(path))
    return {row.claim_id: row for row in rows}


def _load_feature_arrays(path: Path) -> tuple[list[str], np.ndarray]:
    if path.suffix.lower() == ".npz":
        payload = np.load(path, allow_pickle=False)
        return [str(item) for item in payload["claim_ids"].tolist()], np.asarray(payload["matrix"], dtype=float)
    rows = [ClaimFeatureRow.from_dict(row) for row in read_jsonl(path)]
    return [row.claim_id for row in rows], np.asarray([row.vector for row in rows], dtype=float)


def _loo_probe_scores(
    feature_path: Path,
    fit_labels: dict[str, dict[str, Any]],
    candidate_claim_ids: set[str],
) -> dict[str, float]:
    claim_ids, x_all = _load_feature_arrays(feature_path)
    shared_idx = [idx for idx, claim_id in enumerate(claim_ids) if claim_id in fit_labels and claim_id in candidate_claim_ids]
    shared_claim_ids = [claim_ids[idx] for idx in shared_idx]
    x = x_all[shared_idx]
    y_fit = np.asarray([1 if fit_labels[claim_id]["correctness_label"] == "true" else 0 for claim_id in shared_claim_ids], dtype=int)
    if len(set(y_fit.tolist())) < 2:
        raise ValueError("Probe fit labels contain a single class")
    probs = np.zeros(len(shared_claim_ids), dtype=float)
    for train_idx, test_idx in LeaveOneOut().split(x):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, solver="liblinear", max_iter=5000))
        model.fit(x[train_idx], y_fit[train_idx])
        probs[test_idx[0]] = model.predict_proba(x[test_idx])[0, 1]
    return {claim_id: float(prob) for claim_id, prob in zip(shared_claim_ids, probs)}


def _score_maps(
    *,
    llama_predictions: Path,
    gpt54_predictions: Path,
    residual_features: Path,
    sae_features: Path,
    fit_labels: dict[str, dict[str, Any]],
    candidate_claim_ids: set[str],
) -> dict[str, dict[str, float]]:
    llama = _load_predictions(llama_predictions)
    gpt54 = _load_predictions(gpt54_predictions)
    return {
        "llama_self_report": {claim_id: row.correctness_confidence for claim_id, row in llama.items()},
        "gpt54_cached": {claim_id: row.correctness_confidence for claim_id, row in gpt54.items()},
        "residual_probe": _loo_probe_scores(residual_features, fit_labels, candidate_claim_ids),
        "sae_probe": _loo_probe_scores(sae_features, fit_labels, candidate_claim_ids),
    }


def _binary_labels(claim_ids: list[str], labels: dict[str, dict[str, Any]]) -> np.ndarray:
    return np.asarray([1 if labels[claim_id]["correctness_label"] == "true" else 0 for claim_id in claim_ids], dtype=int)


def _scores(claim_ids: list[str], scores: dict[str, float]) -> np.ndarray:
    return np.asarray([scores[claim_id] for claim_id in claim_ids], dtype=float)


def _percentile(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    arr = np.asarray(values, dtype=float)
    return float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


def _safe_metric(metric: Callable[[Any, Any], float], y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    try:
        return float(metric(y_true, y_score))
    except ValueError:
        return None


def _metric_ci(
    *,
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: Callable[[Any, Any], float],
    resample_indices: list[np.ndarray],
) -> dict[str, Any]:
    point = _safe_metric(metric, y_true, y_score)
    samples: list[float] = []
    dropped = 0
    for idx in resample_indices:
        value = _safe_metric(metric, y_true[idx], y_score[idx])
        if value is None:
            dropped += 1
            continue
        samples.append(value)
    low, high = _percentile(samples)
    return {
        "point": point,
        "ci_low": low,
        "ci_high": high,
        "n_valid_resamples": len(samples),
        "n_resamples_dropped": dropped,
    }


def _delta_ci(
    *,
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    metric: Callable[[Any, Any], float],
    resample_indices: list[np.ndarray],
) -> dict[str, Any]:
    point_a = _safe_metric(metric, y_true, score_a)
    point_b = _safe_metric(metric, y_true, score_b)
    if point_a is None or point_b is None:
        point = None
    else:
        point = float(point_a - point_b)
    samples: list[float] = []
    dropped = 0
    for idx in resample_indices:
        value_a = _safe_metric(metric, y_true[idx], score_a[idx])
        value_b = _safe_metric(metric, y_true[idx], score_b[idx])
        if value_a is None or value_b is None:
            dropped += 1
            continue
        samples.append(float(value_a - value_b))
    low, high = _percentile(samples)
    excludes_zero = None
    if low is not None and high is not None:
        excludes_zero = bool(low > 0.0 or high < 0.0)
    return {
        "point": point,
        "ci_low": low,
        "ci_high": high,
        "n_valid_resamples": len(samples),
        "n_resamples_dropped": dropped,
        "ci_excludes_zero": excludes_zero,
    }


def _cohen_kappa_from_arrays(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) == 0:
        return float("nan")
    observed = float(np.mean(left == right))
    n = len(left)
    left_counts = Counter(left.tolist())
    right_counts = Counter(right.tolist())
    expected = sum((left_counts[label] / n) * (right_counts[label] / n) for label in LABELS)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else float("nan")
    return float((observed - expected) / (1.0 - expected))


def _kappa_ci(
    claim_ids: list[str],
    judge1: dict[str, dict[str, Any]],
    judge2: dict[str, dict[str, Any]],
    resample_indices: list[np.ndarray],
) -> dict[str, Any]:
    left = np.asarray([judge1[claim_id]["correctness_label"] for claim_id in claim_ids], dtype=object)
    right = np.asarray([judge2[claim_id]["correctness_label"] for claim_id in claim_ids], dtype=object)
    point = _cohen_kappa_from_arrays(left, right)
    samples = [_cohen_kappa_from_arrays(left[idx], right[idx]) for idx in resample_indices]
    samples = [value for value in samples if not np.isnan(value)]
    low, high = _percentile(samples)
    return {
        "point": point,
        "ci_low": low,
        "ci_high": high,
        "n_valid_resamples": len(samples),
        "n_resamples_dropped": len(resample_indices) - len(samples),
    }


def _make_resamples(n: int, n_resamples: int, rng: np.random.Generator) -> list[np.ndarray]:
    return [rng.integers(0, n, size=n, endpoint=False) for _ in range(n_resamples)]


def _context_metrics(
    *,
    claim_ids: list[str],
    labels: dict[str, dict[str, Any]],
    score_maps: dict[str, dict[str, float]],
    resample_indices: list[np.ndarray],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    shared = [claim_id for claim_id in claim_ids if claim_id in labels and all(claim_id in score_maps[m] for m in METHOD_ORDER)]
    y = _binary_labels(shared, labels)
    method_scores = {method: _scores(shared, score_maps[method]) for method in METHOD_ORDER}
    auroc_payload = {
        method: _metric_ci(y_true=y, y_score=method_scores[method], metric=auroc, resample_indices=resample_indices)
        for method in METHOD_ORDER
    }
    brier_payload = {
        method: _metric_ci(y_true=y, y_score=method_scores[method], metric=brier_score, resample_indices=resample_indices)
        for method in METHOD_ORDER
    }
    delta_payload = {
        "residual_minus_self_report": _delta_ci(
            y_true=y,
            score_a=method_scores["residual_probe"],
            score_b=method_scores["llama_self_report"],
            metric=auroc,
            resample_indices=resample_indices,
        ),
        "gpt54_minus_residual": _delta_ci(
            y_true=y,
            score_a=method_scores["gpt54_cached"],
            score_b=method_scores["residual_probe"],
            metric=auroc,
            resample_indices=resample_indices,
        ),
        "residual_minus_sae": _delta_ci(
            y_true=y,
            score_a=method_scores["residual_probe"],
            score_b=method_scores["sae_probe"],
            metric=auroc,
            resample_indices=resample_indices,
        ),
    }
    return auroc_payload, brier_payload, delta_payload


def _validate_points(payload: dict[str, Any], paths: dict[str, Path], tolerance: float = 1e-12) -> dict[str, Any]:
    expected = {
        "judge1": {
            "llama_self_report": json.loads(paths["judge1_baseline"].read_text())["llama_self_report"]["correctness"],
            "gpt54_cached": json.loads(paths["judge1_baseline"].read_text())["gpt54_cached"]["correctness"],
            "residual_probe": json.loads(paths["judge1_probe"].read_text())["residual"]["correctness"],
            "sae_probe": json.loads(paths["judge1_probe"].read_text())["sae"]["correctness"],
        },
        "judge2": {
            "llama_self_report": json.loads(paths["judge2_baseline"].read_text())["llama_self_report"]["correctness"],
            "gpt54_cached": json.loads(paths["judge2_baseline"].read_text())["gpt54_cached"]["correctness"],
            "residual_probe": json.loads(paths["judge2_probe"].read_text())["residual"]["correctness"],
            "sae_probe": json.loads(paths["judge2_probe"].read_text())["sae"]["correctness"],
        },
        "agreement_set": {
            method: json.loads(paths["agreement_eval"].read_text())["metrics"][method]["correctness"]
            for method in METHOD_ORDER
        },
    }
    diffs: list[dict[str, Any]] = []
    max_abs_diff = 0.0
    for context_name, methods in expected.items():
        for method, metrics in methods.items():
            for metric_name in ("auroc", "brier"):
                actual = payload[metric_name][context_name][method]["point"]
                expected_value = metrics[metric_name]
                diff = abs(float(actual) - float(expected_value))
                max_abs_diff = max(max_abs_diff, diff)
                diffs.append(
                    {
                        "context": context_name,
                        "method": method,
                        "metric": metric_name,
                        "actual": actual,
                        "expected": expected_value,
                        "abs_diff": diff,
                    }
                )
                if diff > tolerance:
                    raise RuntimeError(
                        f"Point-estimate mismatch for {context_name}/{method}/{metric_name}: "
                        f"actual={actual} expected={expected_value} diff={diff}"
                    )
    return {"status": "passed", "tolerance": tolerance, "max_abs_diff": max_abs_diff, "checked": diffs}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute paired bootstrap CIs for MAUD judge-proxy headline metrics.")
    parser.add_argument("--judge1-labels-jsonl", type=Path, default=Path("data/annotations/maud_full_judge_annotations.jsonl"))
    parser.add_argument("--judge2-labels-jsonl", type=Path, default=Path("data/annotations/maud_full_judge_annotations_v2.jsonl"))
    parser.add_argument("--llama-predictions", type=Path, default=Path("data/cached_baselines/llama_self_report/parsed/maud_full/_all_predictions.jsonl"))
    parser.add_argument("--gpt54-predictions", type=Path, default=Path("data/cached_baselines/gpt54/parsed/maud_full"))
    parser.add_argument("--residual-features-jsonl", type=Path, default=Path("artifacts/runs/maud_full_probe_features_residual.jsonl"))
    parser.add_argument("--sae-features-path", type=Path, default=Path("artifacts/runs/maud_full_probe_features_sae.npz"))
    parser.add_argument("--judge1-baseline-json", type=Path, default=Path("artifacts/runs/maud_full_proxy_baseline_eval.json"))
    parser.add_argument("--judge1-probe-json", type=Path, default=Path("artifacts/runs/maud_full_probe_proxy_smoke.json"))
    parser.add_argument("--judge2-baseline-json", type=Path, default=Path("artifacts/runs/maud_full_proxy_baseline_eval_v2.json"))
    parser.add_argument("--judge2-probe-json", type=Path, default=Path("artifacts/runs/maud_full_probe_proxy_smoke_v2.json"))
    parser.add_argument("--agreement-eval-json", type=Path, default=Path("artifacts/runs/maud_agreement_set_eval.json"))
    parser.add_argument("--output-json", type=Path, default=Path("artifacts/runs/maud_bootstrap_ci.json"))
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260424)
    args = parser.parse_args()

    judge1 = _load_labels(args.judge1_labels_jsonl)
    judge2 = _load_labels(args.judge2_labels_jsonl)
    judge1_order = _load_claim_order(args.judge1_labels_jsonl)
    judge1_ids = [claim_id for claim_id in judge1_order if claim_id in judge2]
    shared_ids = sorted(set(judge1) & set(judge2))
    agreement_ids = [
        claim_id
        for claim_id in shared_ids
        if judge1[claim_id]["correctness_label"] == judge2[claim_id]["correctness_label"]
    ]
    score_maps = _score_maps(
        llama_predictions=args.llama_predictions,
        gpt54_predictions=args.gpt54_predictions,
        residual_features=args.residual_features_jsonl,
        sae_features=args.sae_features_path,
        fit_labels=judge1,
        candidate_claim_ids=set(shared_ids),
    )

    rng = np.random.default_rng(args.seed)
    full_resamples = _make_resamples(len(shared_ids), args.n_resamples, rng)
    judge1_resamples = _make_resamples(len(judge1_ids), args.n_resamples, rng)
    agreement_resamples = _make_resamples(len(agreement_ids), args.n_resamples, rng)

    j1_auroc, j1_brier, j1_deltas = _context_metrics(
        claim_ids=judge1_ids,
        labels=judge1,
        score_maps=score_maps,
        resample_indices=judge1_resamples,
    )
    j2_auroc, j2_brier, j2_deltas = _context_metrics(
        claim_ids=shared_ids,
        labels=judge2,
        score_maps=score_maps,
        resample_indices=full_resamples,
    )
    agreement_auroc, agreement_brier, agreement_deltas = _context_metrics(
        claim_ids=agreement_ids,
        labels=judge1,
        score_maps=score_maps,
        resample_indices=agreement_resamples,
    )

    # The generic delta helper assumes one label vector. Recompute this judge-coupling
    # delta explicitly because the same GPT-5.4 scores are evaluated against two label vectors.
    y1 = _binary_labels(shared_ids, judge1)
    y2 = _binary_labels(shared_ids, judge2)
    gpt_scores = _scores(shared_ids, score_maps["gpt54_cached"])
    gpt_samples: list[float] = []
    dropped = 0
    for idx in full_resamples:
        value1 = _safe_metric(auroc, y1[idx], gpt_scores[idx])
        value2 = _safe_metric(auroc, y2[idx], gpt_scores[idx])
        if value1 is None or value2 is None:
            dropped += 1
            continue
        gpt_samples.append(float(value1 - value2))
    low, high = _percentile(gpt_samples)
    gpt54_judge_delta = {
        "point": float(auroc(y1, gpt_scores) - auroc(y2, gpt_scores)),
        "ci_low": low,
        "ci_high": high,
        "n_valid_resamples": len(gpt_samples),
        "n_resamples_dropped": dropped,
        "ci_excludes_zero": bool(low is not None and high is not None and (low > 0.0 or high < 0.0)),
    }

    payload: dict[str, Any] = {
        "n_resamples": args.n_resamples,
        "seed": args.seed,
        "confidence_level": 0.95,
        "bootstrap_method": "paired claim-level bootstrap with percentile 95% intervals",
        "n_resamples_dropped": 0,
        "auroc_undefined_resamples_dropped": {
            "total": int(
                sum(
                    block["n_resamples_dropped"]
                    for metric_by_context in (j1_auroc, j2_auroc, agreement_auroc)
                    for block in metric_by_context.values()
                )
            ),
            "note": "Counts across AUROC cell CIs only; paired-delta and kappa dropped counts are reported per entry.",
        },
        "agreement_set_size": len(agreement_ids),
        "method_labels": METHOD_LABELS,
        "auroc": {
            "judge1": j1_auroc,
            "judge2": j2_auroc,
            "agreement_set": agreement_auroc,
        },
        "brier": {
            "judge1": j1_brier,
            "judge2": j2_brier,
            "agreement_set": agreement_brier,
        },
        "deltas": {
            "residual_minus_self_report": {
                "judge1": j1_deltas["residual_minus_self_report"],
                "judge2": j2_deltas["residual_minus_self_report"],
                "agreement_set": agreement_deltas["residual_minus_self_report"],
                "metric": "auroc",
            },
            "gpt54_minus_residual": {
                "judge1": j1_deltas["gpt54_minus_residual"],
                "judge2": j2_deltas["gpt54_minus_residual"],
                "agreement_set": agreement_deltas["gpt54_minus_residual"],
                "metric": "auroc",
            },
            "residual_minus_sae": {
                "judge1": j1_deltas["residual_minus_sae"],
                "judge2": j2_deltas["residual_minus_sae"],
                "agreement_set": agreement_deltas["residual_minus_sae"],
                "metric": "auroc",
            },
            "gpt54_judge1_minus_judge2": {
                **gpt54_judge_delta,
                "metric": "auroc",
            },
        },
        "kappa": _kappa_ci(shared_ids, judge1, judge2, full_resamples),
        "inputs": {
            "judge1_labels_jsonl": str(args.judge1_labels_jsonl),
            "judge2_labels_jsonl": str(args.judge2_labels_jsonl),
            "llama_predictions": str(args.llama_predictions),
            "gpt54_predictions": str(args.gpt54_predictions),
            "residual_features_jsonl": str(args.residual_features_jsonl),
            "sae_features_path": str(args.sae_features_path),
        },
    }
    payload["n_resamples_dropped"] = payload["auroc_undefined_resamples_dropped"]["total"]
    payload["point_estimate_validation"] = _validate_points(
        payload,
        {
            "judge1_baseline": args.judge1_baseline_json,
            "judge1_probe": args.judge1_probe_json,
            "judge2_baseline": args.judge2_baseline_json,
            "judge2_probe": args.judge2_probe_json,
            "agreement_eval": args.agreement_eval_json,
        },
    )
    write_json(args.output_json, payload)
    print(f"Wrote bootstrap CIs to {args.output_json}")
    print(f"Point validation max abs diff: {payload['point_estimate_validation']['max_abs_diff']:.3g}")


if __name__ == "__main__":
    main()
