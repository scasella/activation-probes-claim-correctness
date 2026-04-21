from __future__ import annotations

import argparse
from pathlib import Path

from interp_experiment.io import read_jsonl, write_jsonl
from interp_experiment.schemas import ExampleRow


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a filtered subset of examples.jsonl.")
    parser.add_argument("--input-jsonl", type=Path, default=Path("data/source_pool/examples.jsonl"))
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--source-corpus", default=None)
    parser.add_argument("--include-splits", nargs="*", default=None)
    args = parser.parse_args()

    rows = [ExampleRow.from_dict(row) for row in read_jsonl(args.input_jsonl)]
    if args.source_corpus is not None:
        rows = [row for row in rows if row.source_corpus == args.source_corpus]
    if args.include_splits:
        allowed = set(args.include_splits)
        rows = [row for row in rows if row.split in allowed]
    write_jsonl(args.output_jsonl, [row.as_dict() for row in rows])
    print(f"Wrote {len(rows)} examples to {args.output_jsonl}")


if __name__ == "__main__":
    main()
