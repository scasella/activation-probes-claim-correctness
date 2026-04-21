from __future__ import annotations

import math
import random
from collections.abc import Callable

import numpy as np


def brier_score(y_true: list[int] | np.ndarray, y_prob: list[float] | np.ndarray) -> float:
    truth = np.asarray(y_true, dtype=float)
    prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((prob - truth) ** 2))


def auroc(y_true: list[int] | np.ndarray, y_score: list[float] | np.ndarray) -> float:
    truth = np.asarray(y_true, dtype=int)
    score = np.asarray(y_score, dtype=float)
    positives = truth == 1
    negatives = truth == 0
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC requires at least one positive and one negative example")
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    pos_ranks = ranks[positives]
    return float((pos_ranks.sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def bootstrap_ci(
    values: list[float],
    n_resamples: int = 1000,
    seed: int = 13,
    alpha: float = 0.05,
) -> tuple[float, float]:
    if not values:
        raise ValueError("bootstrap_ci requires at least one value")
    rng = random.Random(seed)
    samples = []
    for _ in range(n_resamples):
        draw = [values[rng.randrange(len(values))] for _ in range(len(values))]
        samples.append(sum(draw) / len(draw))
    samples.sort()
    lower_idx = int((alpha / 2) * n_resamples)
    upper_idx = int((1 - alpha / 2) * n_resamples) - 1
    return float(samples[lower_idx]), float(samples[upper_idx])


def paired_bootstrap_metric_delta(
    y_true: list[int],
    y_score_a: list[float],
    y_score_b: list[float],
    metric: Callable[[list[int], list[float]], float],
    n_resamples: int = 1000,
    seed: int = 17,
) -> dict[str, float]:
    if not (len(y_true) == len(y_score_a) == len(y_score_b)):
        raise ValueError("paired bootstrap inputs must have equal length")
    base_delta = metric(y_true, y_score_a) - metric(y_true, y_score_b)
    rng = random.Random(seed)
    deltas = []
    indices = list(range(len(y_true)))
    for _ in range(n_resamples):
        sample = [indices[rng.randrange(len(indices))] for _ in indices]
        truth = [y_true[i] for i in sample]
        pred_a = [y_score_a[i] for i in sample]
        pred_b = [y_score_b[i] for i in sample]
        try:
            delta = metric(truth, pred_a) - metric(truth, pred_b)
        except ValueError:
            continue
        deltas.append(delta)
    if not deltas:
        raise ValueError("Unable to compute paired bootstrap interval: all resamples were degenerate")
    ci_low, ci_high = bootstrap_ci(deltas, n_resamples=n_resamples, seed=seed + 1)
    return {
        "delta": float(base_delta),
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def calibration_bin_stats(y_true: list[int], y_prob: list[float], n_bins: int = 10) -> list[dict[str, float]]:
    truth = np.asarray(y_true, dtype=float)
    prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for left, right in zip(bins[:-1], bins[1:]):
        if math.isclose(right, 1.0):
            mask = (prob >= left) & (prob <= right)
        else:
            mask = (prob >= left) & (prob < right)
        if not mask.any():
            continue
        rows.append(
            {
                "bin_left": float(left),
                "bin_right": float(right),
                "mean_confidence": float(prob[mask].mean()),
                "empirical_accuracy": float(truth[mask].mean()),
                "count": float(mask.sum()),
            }
        )
    return rows
