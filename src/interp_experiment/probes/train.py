from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .feature_views import matrix_from_rows
from ..schemas import ClaimFeatureRow


@dataclass(slots=True)
class ProbeBundle:
    task_name: str
    model_type: str
    hyperparameter: float
    model: Any


def _load_sklearn():
    try:
        from sklearn.linear_model import LogisticRegression, Ridge
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn is required for probe training. Install with `uv sync`."
        ) from exc
    return LogisticRegression, Ridge


def train_correctness_ridge(rows: list[ClaimFeatureRow], alpha: float = 1.0) -> ProbeBundle:
    _, Ridge = _load_sklearn()
    x, y = matrix_from_rows(rows, "correctness_target")
    model = Ridge(alpha=alpha)
    model.fit(x, y)
    return ProbeBundle(task_name="correctness", model_type="ridge", hyperparameter=alpha, model=model)


def train_binary_probe(
    rows: list[ClaimFeatureRow],
    target_name: str,
    c_value: float = 1.0,
) -> ProbeBundle:
    LogisticRegression, _ = _load_sklearn()
    x, y = matrix_from_rows(rows, target_name)  # type: ignore[arg-type]
    model = LogisticRegression(C=c_value, penalty="l2", max_iter=2000)
    model.fit(x, y)
    return ProbeBundle(task_name=target_name, model_type="logistic_regression_l2", hyperparameter=c_value, model=model)
