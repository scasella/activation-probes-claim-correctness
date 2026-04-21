from __future__ import annotations

import argparse
from pathlib import Path

from interp_experiment.config import load_config
from interp_experiment.data.contracts import write_source_pool
from interp_experiment.data.pilot import sample_pilot_examples
from interp_experiment.data.seed_corpora import build_hybrid_source_pool
from interp_experiment.data.split_freeze import freeze_contract_splits
from interp_experiment.env import load_repo_env
from interp_experiment.schemas import ExampleRow


def _allocate_evenly(total: int, keys: list[str]) -> dict[str, int]:
    base = total // len(keys)
    remainder = total % len(keys)
    return {
        key: base + (1 if index < remainder else 0)
        for index, key in enumerate(keys)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and freeze the MAUD/CUAD hybrid source pool.")
    parser.add_argument("--output-jsonl", type=Path, default=Path("data/source_pool/examples.jsonl"))
    parser.add_argument("--manifest-json", type=Path, default=Path("data/source_pool/manifest.json"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--pilot-size", type=int, default=0)
    args = parser.parse_args()

    load_repo_env()
    cfg = load_config("dataset.yaml")
    pilot_size = int(args.pilot_size or cfg.get("pilot_size", 0))
    source_names = list(cfg["sources"].keys())
    pilot_allocation = _allocate_evenly(pilot_size, source_names)
    target_overrides = {
        source_name: int(source_cfg["target_examples"]) + pilot_allocation[source_name] * int(source_cfg.get("max_rows_per_contract", 1))
        for source_name, source_cfg in cfg["sources"].items()
    }
    candidates = build_hybrid_source_pool(seed=args.seed, target_overrides=target_overrides)
    pilot_examples = sample_pilot_examples(
        candidates,
        pilot_size=pilot_size,
        stratify_field="source_corpus",
        seed=args.seed,
        excluded_splits=(),
        max_rows_per_contract=1,
    )
    pilot_contract_ids = {row.contract_id for row in pilot_examples}
    nonpilot_candidates = [row for row in candidates if row.contract_id not in pilot_contract_ids]

    trimmed_nonpilot: list[ExampleRow] = []
    for source_name, source_cfg in cfg["sources"].items():
        remaining_target = int(source_cfg["target_examples"]) - pilot_allocation[source_name]
        source_rows = [row for row in nonpilot_candidates if row.source_corpus == source_name]
        if len(source_rows) < remaining_target:
            raise SystemExit(
                f"Not enough non-pilot rows for source {source_name}: need {remaining_target}, found {len(source_rows)}"
            )
        trimmed_nonpilot.extend(source_rows[:remaining_target])

    frozen_nonpilot = freeze_contract_splits(
        trimmed_nonpilot,
        train_ratio=float(cfg["split"]["train"]),
        validation_ratio=float(cfg["split"]["validation"]),
        test_ratio=float(cfg["split"]["test"]),
        seed=args.seed,
    )
    pilot_rows = [
        ExampleRow(
            example_id=row.example_id,
            source_corpus=row.source_corpus,
            task_family=row.task_family,
            contract_id=row.contract_id,
            contract_group=row.contract_group,
            excerpt_text=row.excerpt_text,
            question_text=row.question_text,
            public_seed_answer=row.public_seed_answer,
            llama_answer_text=row.llama_answer_text,
            split="pilot",
            cross_dist_group=row.cross_dist_group,
        ).validate()
        for row in pilot_examples
    ]
    final_rows = pilot_rows + frozen_nonpilot
    write_source_pool(final_rows, args.output_jsonl, args.manifest_json)
    print(f"Wrote {len(final_rows)} examples to {args.output_jsonl}")


if __name__ == "__main__":
    main()
