from __future__ import annotations

import argparse
from pathlib import Path

from interp_experiment.config import load_config
from interp_experiment.data.contracts import write_source_pool
from interp_experiment.data.seed_corpora import build_hybrid_source_pool
from interp_experiment.data.split_freeze import freeze_contract_splits
from interp_experiment.env import load_repo_env


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and freeze the MAUD/CUAD hybrid source pool.")
    parser.add_argument("--output-jsonl", type=Path, default=Path("data/source_pool/examples.jsonl"))
    parser.add_argument("--manifest-json", type=Path, default=Path("data/source_pool/manifest.json"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    load_repo_env()
    cfg = load_config("dataset.yaml")
    examples = build_hybrid_source_pool()
    frozen = freeze_contract_splits(
        examples,
        train_ratio=float(cfg["split"]["train"]),
        validation_ratio=float(cfg["split"]["validation"]),
        test_ratio=float(cfg["split"]["test"]),
        seed=args.seed,
    )
    write_source_pool(frozen, args.output_jsonl, args.manifest_json)
    print(f"Wrote {len(frozen)} examples to {args.output_jsonl}")


if __name__ == "__main__":
    main()
