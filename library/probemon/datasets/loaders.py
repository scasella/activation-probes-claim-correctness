from __future__ import annotations

from pathlib import Path

from probemon.training.adapters import convert_factscore_to_canonical, convert_maud_to_canonical
from probemon.training.dataset import CanonicalDataset


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_dataset(name: str, *, repo_root: str | Path | None = None) -> CanonicalDataset:
    """Load a scaffolding dataset from local project artifacts.

    Datasets are intentionally not bundled in the wheel. MAUD redistribution
    should be checked against the Atticus Project terms before bundling.
    FActScore is publicly released by its authors in an MIT-licensed upstream
    repository, but data-specific redistribution terms should still be checked
    before bundling the human annotations. This loader uses local artifacts
    when run from the methods-paper repository.
    """

    root = Path(repo_root) if repo_root is not None else _repo_root()
    if name == "maud-legal-qa":
        return convert_maud_to_canonical(
            examples_jsonl=root / "artifacts/runs/maud_full_examples.jsonl",
            claims_jsonl=root / "artifacts/runs/maud_full_claims.jsonl",
            labels_jsonl=root / "data/annotations/maud_full_judge_annotations.jsonl",
        )
    if name == "factscore-biographies":
        return convert_factscore_to_canonical(
            examples_jsonl=root / "data/factscore/factscore_chatgpt_examples.jsonl",
            claims_jsonl=root / "data/factscore/factscore_chatgpt_claims.jsonl",
        )
    raise KeyError("Unknown dataset. Expected 'maud-legal-qa' or 'factscore-biographies'.")
