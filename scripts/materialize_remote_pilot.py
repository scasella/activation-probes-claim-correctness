from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from interp_experiment.env import load_repo_env
from interp_experiment.io import write_json
from interp_experiment.utils import ensure_parent


def _extract_section(text: str, label: str) -> str:
    begin = f"===BEGIN_{label}==="
    end = f"===END_{label}==="
    start_idx = text.find(begin)
    end_idx = text.find(end)
    if start_idx < 0 or end_idx < 0 or end_idx <= start_idx:
        raise RuntimeError(f"Could not locate section {label} in remote output")
    return text[start_idx + len(begin) : end_idx].strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run remote pilot answer generation and write local artifacts.")
    parser.add_argument("--input-jsonl", type=Path, default=Path("data/annotations/pilot_examples.jsonl"))
    parser.add_argument("--output-examples-jsonl", type=Path, default=Path("data/annotations/pilot_examples_with_answers.jsonl"))
    parser.add_argument("--output-claims-jsonl", type=Path, default=Path("data/annotations/pilot_claims.jsonl"))
    parser.add_argument("--summary-json", type=Path, default=Path("data/annotations/pilot_generation_summary.json"))
    parser.add_argument("--log-path", type=Path, default=Path("artifacts/runs/pilot_generation_remote.log"))
    parser.add_argument("--gpu", default="A100-80GB")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()

    load_repo_env()
    ensure_parent(args.log_path)
    cmd = [
        "uv",
        "run",
        "--extra",
        "modal",
        "python",
        "scripts/modal_gpu.py",
        f"--gpu={args.gpu}",
        f"--timeout={args.timeout}",
        "--sync-extra",
        "inference",
        "--sync-extra",
        "interp",
        "--",
        "python",
        "scripts/remote_pilot_generation.py",
        "--input-jsonl",
        str(args.input_jsonl),
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    args.log_path.write_text(
        result.stdout + ("\n--- STDERR ---\n" + result.stderr if result.stderr else ""),
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise SystemExit(
            "Remote pilot generation failed. "
            f"See {args.log_path} for details. Tail: {result.stderr[-400:] or result.stdout[-400:]}"
        )

    summary = json.loads(_extract_section(result.stdout, "SUMMARY_JSON"))
    examples_jsonl = _extract_section(result.stdout, "EXAMPLES_JSONL")
    claims_jsonl = _extract_section(result.stdout, "CLAIMS_JSONL")

    ensure_parent(args.output_examples_jsonl)
    args.output_examples_jsonl.write_text(examples_jsonl + "\n", encoding="utf-8")
    ensure_parent(args.output_claims_jsonl)
    args.output_claims_jsonl.write_text(claims_jsonl + "\n", encoding="utf-8")
    write_json(args.summary_json, summary)
    print(f"Wrote pilot answers to {args.output_examples_jsonl}")
    print(f"Wrote pilot claims to {args.output_claims_jsonl}")
    print(f"Wrote pilot summary to {args.summary_json}")


if __name__ == "__main__":
    main()
