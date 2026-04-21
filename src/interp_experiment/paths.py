from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = project_root()
CONFIGS_DIR = ROOT / "configs"
DOCS_DIR = ROOT / "docs"
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
RUNS_DIR = ARTIFACTS_DIR / "runs"
