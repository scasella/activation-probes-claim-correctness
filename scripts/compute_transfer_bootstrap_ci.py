from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np

from interp_experiment.evaluation.metrics import auroc, brier_score
from interp_experiment.io import write_json


def _metric_block(
    *,
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: Callable[[Any, Any], float],
    resample_indices: list[np.ndarray],
) -> dict[str, Any]:
    point = float(metric(y_true, y_score))
    samples: list[float] = []
    dropped = 0
    for idx in resample_indices:
        try:
            samples.append(float(metric(y_true[idx], y_score[idx])))
        except ValueError:
            dropped += 1
    low, high = np.quantile(np.asarray(samples, dtype=float), [0.025, 0.975]).tolist()
    return {
        "point": point,
        "ci_low": float(low),
        "ci_high": float(high),
        "n_valid_resamples": len(samples),
        "n_resamples_dropped": dropped,
    }


def _delta_block(
    *,
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    metric: Callable[[Any, Any], float],
    resample_indices: list[np.ndarray],
) -> dict[str, Any]:
    point = float(metric(y_true, score_a) - metric(y_true, score_b))
    samples: list[float] = []
    dropped = 0
    for idx in resample_indices:
        try:
            samples.append(float(metric(y_true[idx], score_a[idx]) - metric(y_true[idx], score_b[idx])))
        except ValueError:
            dropped += 1
    low, high = np.quantile(np.asarray(samples, dtype=float), [0.025, 0.975]).tolist()
    return {
        "point": point,
        "ci_low": float(low),
        "ci_high": float(high),
        "n_valid_resamples": len(samples),
        "n_resamples_dropped": dropped,
        "ci_excludes_zero": bool(low > 0.0 or high < 0.0),
    }


def _load_eval(path: Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text())
    claim_ids = [str(item) for item in payload["residual_probe"]["test"]["claim_ids"]]
    y_true = np.asarray(payload["residual_probe"]["test"]["y_true"], dtype=int)
    residual_scores = np.asarray(payload["residual_probe"]["test"]["scores"], dtype=float)
    self_score_map = payload["llama_self_report"]["test"]["scores"]
    self_scores = np.asarray([float(self_score_map[claim_id]) for claim_id in claim_ids], dtype=float)
    return claim_ids, y_true, residual_scores, self_scores


def _resamples(n: int, n_resamples: int, seed: int) -> list[np.ndarray]:
    rng = random.Random(seed)
    indices = list(range(n))
    return [np.asarray([indices[rng.randrange(n)] for _ in indices], dtype=int) for _ in range(n_resamples)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute paired bootstrap CIs for two-method transfer eval artifacts.")
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260424)
    args = parser.parse_args()

    claim_ids, y_true, residual_scores, self_scores = _load_eval(args.eval_json)
    resample_indices = _resamples(len(y_true), args.n_resamples, args.seed)
    payload = {
        "dataset": args.dataset,
        "n_claims": len(claim_ids),
        "n_positive": int(y_true.sum()),
        "n_negative": int(len(y_true) - y_true.sum()),
        "n_resamples": args.n_resamples,
        "seed": args.seed,
        "confidence_level": 0.95,
        "bootstrap_method": "paired claim-level bootstrap with percentile 95% intervals",
        "inputs": {"eval_json": str(args.eval_json)},
        "auroc": {
            "llama_self_report": _metric_block(
                y_true=y_true,
                y_score=self_scores,
                metric=auroc,
                resample_indices=resample_indices,
            ),
            "residual_probe": _metric_block(
                y_true=y_true,
                y_score=residual_scores,
                metric=auroc,
                resample_indices=resample_indices,
            ),
        },
        "brier": {
            "llama_self_report": _metric_block(
                y_true=y_true,
                y_score=self_scores,
                metric=brier_score,
                resample_indices=resample_indices,
            ),
            "residual_probe": _metric_block(
                y_true=y_true,
                y_score=residual_scores,
                metric=brier_score,
                resample_indices=resample_indices,
            ),
        },
        "deltas": {
            "residual_minus_self_report": {
                **_delta_block(
                    y_true=y_true,
                    score_a=residual_scores,
                    score_b=self_scores,
                    metric=auroc,
                    resample_indices=resample_indices,
                ),
                "metric": "auroc",
            }
        },
    }
    write_json(args.output_json, payload)
    print(f"Wrote transfer bootstrap CIs to {args.output_json}")


if __name__ == "__main__":
    main()
