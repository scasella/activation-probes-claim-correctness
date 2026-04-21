from __future__ import annotations

from typing import Literal

import numpy as np

from ..schemas import ClaimFeatureRow


def matrix_from_rows(
    rows: list[ClaimFeatureRow],
    target_name: Literal["correctness_target", "load_bearing_target", "stability_target"],
) -> tuple[np.ndarray, np.ndarray]:
    filtered = [row for row in rows if getattr(row, target_name) is not None]
    if not filtered:
        raise ValueError(f"No rows contain target {target_name}")
    x = np.asarray([row.vector for row in filtered], dtype=float)
    y = np.asarray([getattr(row, target_name) for row in filtered], dtype=float)
    return x, y
