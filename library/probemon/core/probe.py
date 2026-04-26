from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .calibration import apply_platt


class ModelMismatchError(ValueError):
    pass


@dataclass(slots=True)
class Probe:
    probe_id: str
    direction: np.ndarray
    bias: float
    platt_a: float
    platt_b: float
    metadata: dict[str, Any]

    @property
    def model_name(self) -> str:
        return str(self.metadata["model_name"])

    @property
    def layer(self) -> int:
        return int(self.metadata["layer"])

    def check_model_name(self, model_name: str | None) -> None:
        if not model_name:
            return
        allowed = {self.model_name, *self.metadata.get("model_aliases", [])}
        if model_name not in allowed:
            expected = ", ".join(sorted(allowed))
            raise ModelMismatchError(
                f"Probe {self.probe_id!r} was trained for {expected}; got model {model_name!r}."
            )

    def raw_logits(self, vectors: np.ndarray) -> np.ndarray:
        matrix = np.asarray(vectors, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.shape[1] != self.direction.shape[0]:
            raise ValueError(
                f"Feature dimension mismatch: probe expects {self.direction.shape[0]}, got {matrix.shape[1]}"
            )
        return matrix @ self.direction + self.bias

    def score_vectors(self, vectors: np.ndarray) -> np.ndarray:
        return apply_platt(self.raw_logits(vectors), self.platt_a, self.platt_b)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            direction=np.asarray(self.direction, dtype=np.float32),
            bias=np.asarray([self.bias], dtype=np.float32),
            platt_a=np.asarray([self.platt_a], dtype=np.float32),
            platt_b=np.asarray([self.platt_b], dtype=np.float32),
            metadata=np.asarray(json.dumps(self.metadata, sort_keys=True)),
        )


def _metadata_from_npz(value: Any) -> dict[str, Any]:
    if isinstance(value, np.ndarray):
        text = str(value.tolist()) if value.shape else str(value.item())
    else:
        text = str(value)
    return json.loads(text)


def load_probe(probe_id_or_path: str | Path) -> Probe:
    from probemon.pretrained.registry import get_probe_path, normalize_probe_id

    probe_id = normalize_probe_id(str(probe_id_or_path))
    path = get_probe_path(probe_id) if probe_id else Path(probe_id_or_path)
    payload = np.load(path, allow_pickle=False)
    metadata = _metadata_from_npz(payload["metadata"])
    return Probe(
        probe_id=metadata.get("probe_id", path.stem),
        direction=np.asarray(payload["direction"], dtype=float),
        bias=float(np.asarray(payload["bias"]).reshape(-1)[0]),
        platt_a=float(np.asarray(payload["platt_a"]).reshape(-1)[0]),
        platt_b=float(np.asarray(payload["platt_b"]).reshape(-1)[0]),
        metadata=metadata,
    )
