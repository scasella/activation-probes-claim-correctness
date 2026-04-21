from __future__ import annotations

from typing import Any

import numpy as np

from ..schemas import ClaimRow


def mean_pool_claim_features(feature_tensor: Any, claim: ClaimRow) -> list[float]:
    array = np.asarray(feature_tensor)
    pooled = array[claim.token_start : claim.token_end + 1].mean(axis=0)
    return pooled.astype(float).tolist()
