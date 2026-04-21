from __future__ import annotations

import argparse
from pathlib import Path

from interp_experiment.data.pilot import build_claim_annotation_packet
from interp_experiment.io import read_jsonl, write_jsonl
from interp_experiment.schemas import ClaimRow, ExampleRow


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-annotator claim annotation packet rows.")
    parser.add_argument("--examples-jsonl", type=Path, required=True)
    parser.add_argument("--claims-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--annotators", required=True, help="Comma-separated annotator ids, e.g. a1,a2")
    parser.add_argument("--annotation-version", default="v1")
    args = parser.parse_args()

    annotator_ids = [item.strip() for item in args.annotators.split(",") if item.strip()]
    if not annotator_ids:
        raise SystemExit("At least one annotator id is required")

    examples = [ExampleRow.from_dict(row) for row in read_jsonl(args.examples_jsonl)]
    claims = [ClaimRow.from_dict(row) for row in read_jsonl(args.claims_jsonl)]
    packet = build_claim_annotation_packet(
        examples=examples,
        claims=claims,
        annotator_ids=annotator_ids,
        annotation_version=args.annotation_version,
    )
    write_jsonl(args.output_jsonl, packet)
    print(f"Wrote {len(packet)} annotation packet rows to {args.output_jsonl}")


if __name__ == "__main__":
    main()
