from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    values = np.asarray(x, dtype=float)
    clipped = np.clip(values, -60.0, 60.0)
    result = 1.0 / (1.0 + np.exp(-clipped))
    if np.isscalar(x):
        return float(result)
    return result


def apply_platt(logits: np.ndarray, platt_a: float, platt_b: float) -> np.ndarray:
    return np.asarray(sigmoid(platt_a * np.asarray(logits, dtype=float) + platt_b), dtype=float)
