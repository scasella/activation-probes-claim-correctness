from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from interp_experiment.data.pilot import sample_pilot_examples
from interp_experiment.io import read_jsonl, write_json, write_jsonl
from interp_experiment.schemas import ExampleRow


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample a balanced pilot set for the annotation reliability study.")
    parser.add_argument("--examples-jsonl", type=Path, default=Path("data/source_pool/examples.jsonl"))
    parser.add_argument("--output-jsonl", type=Path, default=Path("data/annotations/pilot_examples.jsonl"))
    parser.add_argument("--manifest-json", type=Path, default=Path("data/annotations/pilot_manifest.json"))
    parser.add_argument("--pilot-size", type=int, default=30)
    parser.add_argument("--stratify-field", default="source_corpus")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    examples = [ExampleRow.from_dict(row) for row in read_jsonl(args.examples_jsonl)]
    existing_pilot = [row for row in examples if row.split == "pilot"]
    if existing_pilot:
        if len(existing_pilot) != args.pilot_size:
            raise SystemExit(
                f"Expected exactly {args.pilot_size} pilot rows in {args.examples_jsonl}, found {len(existing_pilot)}"
            )
        sampled = existing_pilot
    else:
        sampled = sample_pilot_examples(
            examples,
            pilot_size=args.pilot_size,
            stratify_field=args.stratify_field,
            seed=args.seed,
            excluded_splits=("test",),
            max_rows_per_contract=1,
        )
    if any(row.split == "test" for row in sampled):
        raise SystemExit("Pilot sample includes test rows, which is not allowed")
    write_jsonl(args.output_jsonl, [row.as_dict() for row in sampled])
    contract_counts = Counter(row.contract_id for row in sampled)
    manifest = {
        "pilot_size": len(sampled),
        "stratify_field": args.stratify_field,
        "seed": args.seed,
        "by_source": dict(Counter(row.source_corpus for row in sampled)),
        "by_task_family": dict(Counter(row.task_family for row in sampled)),
        "by_cross_dist_group": dict(Counter(row.cross_dist_group for row in sampled)),
        "by_split": dict(Counter(row.split for row in sampled)),
        "unique_contract_count": len(contract_counts),
        "max_rows_per_contract": max(contract_counts.values(), default=0),
        "top_repeated_contracts": contract_counts.most_common(5),
    }
    write_json(args.manifest_json, manifest)
    print(f"Wrote {len(sampled)} pilot examples to {args.output_jsonl}")


if __name__ == "__main__":
    main()
