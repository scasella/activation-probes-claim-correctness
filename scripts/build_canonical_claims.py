from __future__ import annotations

import argparse
from pathlib import Path

from interp_experiment.data.claims import build_canonical_claims
from interp_experiment.io import read_jsonl, write_jsonl
from interp_experiment.schemas import ExampleRow


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the frozen canonical claim list from deterministic Llama answers.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--annotation-version", default="v1")
    args = parser.parse_args()

    examples = [ExampleRow.from_dict(row) for row in read_jsonl(args.input_jsonl)]
    claim_rows = []
    for example in examples:
        claim_rows.extend(build_canonical_claims(example, annotation_version=args.annotation_version))
    write_jsonl(args.output_jsonl, [row.as_dict() for row in claim_rows])
    print(f"Wrote {len(claim_rows)} canonical claims to {args.output_jsonl}")


if __name__ == "__main__":
    main()
