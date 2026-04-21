from __future__ import annotations

import argparse
import base64
import json
import subprocess
from pathlib import Path

from interp_experiment.io import read_jsonl, write_json, write_jsonl
from interp_experiment.schemas import AnswerRunRow, ClaimRow
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
    parser = argparse.ArgumentParser(description="Materialize probe features locally via a remote Modal run.")
    parser.add_argument("--answer-runs-jsonl", type=Path, required=True)
    parser.add_argument("--claims-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--feature-source", choices=["residual", "sae"], required=True)
    parser.add_argument("--gpu", default="A100-80GB")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--log-path", type=Path, required=True)
    args = parser.parse_args()

    answer_runs = [AnswerRunRow.from_dict(row) for row in read_jsonl(args.answer_runs_jsonl)]
    claim_ids = {row["example_id"] for row in read_jsonl(args.claims_jsonl)}
    filtered_runs = [row.as_dict() for row in answer_runs if row.example_id in claim_ids]
    ensure_parent(args.log_path)
    staged_answer_runs = args.log_path.with_suffix(f".{args.feature_source}.answer_runs.jsonl")
    ensure_parent(staged_answer_runs)
    staged_answer_runs.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in filtered_runs), encoding="utf-8")
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
        "scripts/remote_probe_features.py",
        "--answer-runs-jsonl",
        str(staged_answer_runs),
        "--claims-jsonl",
        str(args.claims_jsonl),
        "--feature-source",
        args.feature_source,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    args.log_path.write_text(
        result.stdout + ("\n--- STDERR ---\n" + result.stderr if result.stderr else ""),
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise SystemExit(
            "Remote probe feature extraction failed. "
            f"See {args.log_path} for details. Tail: {result.stderr[-400:] or result.stdout[-400:]}"
        )

    summary = json.loads(_extract_section(result.stdout, "SUMMARY_JSON"))
    if args.feature_source == "sae":
        payload = _extract_section(result.stdout, "FEATURES_NPZ_B64")
        ensure_parent(args.output_jsonl)
        args.output_jsonl.write_bytes(base64.b64decode(payload))
        rows_written = summary["n_rows"]
    else:
        rows = [json.loads(line) for line in _extract_section(result.stdout, "FEATURES_JSONL").splitlines() if line.strip()]
        write_jsonl(args.output_jsonl, rows)
        rows_written = len(rows)
    write_json(args.summary_json, summary)
    print(f"Wrote {rows_written} {args.feature_source} feature rows to {args.output_jsonl}")
    print(f"Wrote feature summary to {args.summary_json}")


if __name__ == "__main__":
    main()
