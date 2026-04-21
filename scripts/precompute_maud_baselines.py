from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> None:
    result = subprocess.run(cmd, cwd=str(cwd), check=False, text=True)
    if result.returncode != 0:
        raise SystemExit(f"Command failed ({result.returncode}): {' '.join(cmd)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute MAUD baseline artifacts short of ground-truth evaluation.")
    parser.add_argument("--examples-jsonl", type=Path, default=Path("data/annotations/maud_pilot_examples.jsonl"))
    parser.add_argument("--claims-jsonl", type=Path, default=Path("data/annotations/maud_pilot_claims.jsonl"))
    args = parser.parse_args()

    cwd = Path.cwd()
    _run(
        [
            "uv",
            "run",
            "python",
            "scripts/materialize_llama_self_report.py",
            "--examples-jsonl",
            str(args.examples_jsonl),
            "--claims-jsonl",
            str(args.claims_jsonl),
        ],
        cwd,
    )
    _run(
        [
            "uv",
            "run",
            "python",
            "scripts/materialize_gpt54_baseline.py",
            "--examples-jsonl",
            str(args.examples_jsonl),
            "--claims-jsonl",
            str(args.claims_jsonl),
        ],
        cwd,
    )


if __name__ == "__main__":
    main()
