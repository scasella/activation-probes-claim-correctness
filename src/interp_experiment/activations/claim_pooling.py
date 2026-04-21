from __future__ import annotations

from typing import Any

import numpy as np

from ..schemas import ClaimRow


def mean_pool_claim_features(feature_tensor: Any, claim: ClaimRow) -> list[float]:
    array = np.asarray(feature_tensor)
    if array.ndim < 2:
        raise ValueError("feature_tensor must be at least 2D")
    if claim.token_end >= array.shape[0]:
        raise ValueError(
            f"Claim token span [{claim.token_start}, {claim.token_end}] exceeds feature length {array.shape[0]}"
        )
    if claim.token_start < 0 or claim.token_start > claim.token_end:
        raise ValueError("Claim token span is invalid")
    pooled = array[claim.token_start : claim.token_end + 1].mean(axis=0)
    if pooled.size == 0 or not np.isfinite(pooled).all():
        raise ValueError("Pooled feature vector is empty or non-finite")
    return pooled.astype(float).tolist()
