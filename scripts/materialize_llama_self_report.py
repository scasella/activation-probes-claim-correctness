from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from interp_experiment.env import load_repo_env
from interp_experiment.io import write_json, write_jsonl
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
    parser = argparse.ArgumentParser(description="Materialize Llama self-report baseline artifacts locally.")
    parser.add_argument("--examples-jsonl", type=Path, default=Path("data/annotations/maud_pilot_examples.jsonl"))
    parser.add_argument("--claims-jsonl", type=Path, default=Path("data/annotations/maud_pilot_claims.jsonl"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/cached_baselines/llama_self_report/raw/maud_pilot"))
    parser.add_argument("--parsed-dir", type=Path, default=Path("data/cached_baselines/llama_self_report/parsed/maud_pilot"))
    parser.add_argument("--summary-json", type=Path, default=Path("artifacts/runs/maud_llama_self_report_summary.json"))
    parser.add_argument("--log-path", type=Path, default=Path("artifacts/runs/maud_llama_self_report_remote.log"))
    parser.add_argument("--gpu", default="A100-80GB")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--max-new-tokens", type=int, default=384)
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
        "--",
        "python",
        "scripts/remote_llama_self_report.py",
        "--examples-jsonl",
        str(args.examples_jsonl),
        "--claims-jsonl",
        str(args.claims_jsonl),
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
            "Remote Llama self-report generation failed. "
            f"See {args.log_path} for details. Tail: {result.stderr[-400:] or result.stdout[-400:]}"
        )

    summary = json.loads(_extract_section(result.stdout, "SUMMARY_JSON"))
    raw_rows = [json.loads(line) for line in _extract_section(result.stdout, "RAW_JSONL").splitlines() if line.strip()]
    parsed_by_example = [
        json.loads(line)
        for line in _extract_section(result.stdout, "PARSED_BY_EXAMPLE_JSONL").splitlines()
        if line.strip()
    ]

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    for row in raw_rows:
        raw_path = args.raw_dir / f"{row['example_id']}.txt"
        raw_path.write_text(row["raw_text"], encoding="utf-8")
    args.parsed_dir.mkdir(parents=True, exist_ok=True)
    flat_predictions: list[dict[str, object]] = []
    for row in parsed_by_example:
        example_id = row["example_id"]
        predictions = row["predictions"]
        per_example_path = args.parsed_dir / f"{example_id}.jsonl"
        write_jsonl(per_example_path, predictions)
        flat_predictions.extend(predictions)
    write_jsonl(args.parsed_dir / "_all_predictions.jsonl", flat_predictions)
    write_json(args.summary_json, summary)
    print(f"Wrote {len(raw_rows)} raw self-report files to {args.raw_dir}")
    print(f"Wrote parsed self-report files to {args.parsed_dir}")
    print(f"Wrote self-report summary to {args.summary_json}")


if __name__ == "__main__":
    main()
