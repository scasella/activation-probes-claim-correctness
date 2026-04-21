from __future__ import annotations

import os
from pathlib import Path

from .paths import ROOT


def repo_env_path() -> Path:
    return ROOT / ".env"


def load_repo_env(override: bool = False) -> dict[str, str]:
    path = repo_env_path()
    loaded: dict[str, str] = {}
    if not path.exists():
        return loaded
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if override or key not in os.environ:
            os.environ[key] = value
        loaded[key] = value
    return loaded
