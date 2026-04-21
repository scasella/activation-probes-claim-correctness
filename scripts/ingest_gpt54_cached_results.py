from __future__ import annotations

import argparse
from pathlib import Path

from interp_experiment.baselines.gpt54_cached import ingest_gpt54_cached_output


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse cached GPT-5.4 raw outputs into validated JSONL.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/cached_baselines/gpt54/raw"))
    parser.add_argument("--parsed-dir", type=Path, default=Path("data/cached_baselines/gpt54/parsed"))
    parser.add_argument("--prompt-version", default="v1")
    args = parser.parse_args()

    count = 0
    for raw_path in sorted(args.raw_dir.glob("*.json")):
        parsed_path = args.parsed_dir / f"{raw_path.stem}.jsonl"
        ingest_gpt54_cached_output(raw_path, parsed_path, prompt_version=args.prompt_version)
        count += 1
    print(f"Parsed {count} cached GPT-5.4 outputs")


if __name__ == "__main__":
    main()
