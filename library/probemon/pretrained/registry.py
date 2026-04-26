from __future__ import annotations

from importlib.resources import files
from pathlib import Path


PROBES: dict[str, str] = {
    "legal-qa-llama-3.1-8b-v1": "legal-qa-llama-3.1-8b-v1.npz",
    "biography-llama-3.1-8b-v1": "biography-llama-3.1-8b-v1.npz",
}


def normalize_probe_id(value: str) -> str | None:
    if value in PROBES:
        return value
    return None


def get_probe_path(probe_id: str) -> Path:
    if probe_id not in PROBES:
        known = ", ".join(sorted(PROBES))
        raise KeyError(f"Unknown probe {probe_id!r}. Known probes: {known}")
    return Path(files("probemon.pretrained").joinpath("artifacts", PROBES[probe_id]))


def list_probes() -> list[str]:
    return sorted(PROBES)
