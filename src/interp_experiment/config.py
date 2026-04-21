from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .paths import CONFIGS_DIR


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping in {path}, found {type(payload)!r}")
    return payload


def load_config(name: str) -> dict[str, Any]:
    return load_yaml(CONFIGS_DIR / name)


def load_all_configs() -> dict[str, dict[str, Any]]:
    return {path.name: load_yaml(path) for path in sorted(CONFIGS_DIR.glob("*.yaml"))}
