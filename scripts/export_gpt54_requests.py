from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from interp_experiment.baselines.gpt54_cached import export_gpt54_request_packet
from interp_experiment.io import read_jsonl
from interp_experiment.schemas import ClaimRow, ExampleRow


def main() -> None:
    parser = argparse.ArgumentParser(description="Export cached GPT-5.4 request packets for Codex App Server runs.")
    parser.add_argument("--examples-jsonl", type=Path, required=True)
    parser.add_argument("--claims-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/gpt54_requests"))
    args = parser.parse_args()

    examples = {row.example_id: row for row in (ExampleRow.from_dict(item) for item in read_jsonl(args.examples_jsonl))}
    claims_by_example: dict[str, list[ClaimRow]] = defaultdict(list)
    for claim in (ClaimRow.from_dict(item) for item in read_jsonl(args.claims_jsonl)):
        claims_by_example[claim.example_id].append(claim)

    count = 0
    for example_id, claims in claims_by_example.items():
        packet_path = args.output_dir / f"{example_id}.json"
        export_gpt54_request_packet(examples[example_id], claims, packet_path)
        count += 1
    print(f"Wrote {count} GPT-5.4 request packets to {args.output_dir}")


if __name__ == "__main__":
    main()
