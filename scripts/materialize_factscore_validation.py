from __future__ import annotations

import argparse
import base64
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

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


def _jsonl_section(text: str, label: str) -> list[dict[str, Any]]:
    section = _extract_section(text, label)
    return [json.loads(line) for line in section.splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FActScore materialization in a Modal GPU sandbox.")
    parser.add_argument("--examples-jsonl", type=Path, default=Path("data/factscore/factscore_chatgpt_examples.jsonl"))
    parser.add_argument("--claims-jsonl", type=Path, default=Path("data/factscore/factscore_chatgpt_claims.jsonl"))
    parser.add_argument("--answer-runs-jsonl", type=Path, default=Path("artifacts/runs/factscore_chatgpt_answer_runs.jsonl"))
    parser.add_argument("--tokenized-claims-jsonl", type=Path, default=Path("artifacts/runs/factscore_chatgpt_claims_tokenized.jsonl"))
    parser.add_argument("--features-npz", type=Path, default=Path("artifacts/runs/factscore_chatgpt_probe_features_residual.npz"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/cached_baselines/llama_self_report/raw/factscore_chatgpt"))
    parser.add_argument(
        "--parsed-jsonl",
        type=Path,
        default=Path("data/cached_baselines/llama_self_report/parsed/factscore_chatgpt/_all_predictions.jsonl"),
    )
    parser.add_argument("--summary-json", type=Path, default=Path("artifacts/runs/factscore_chatgpt_materialization_summary.json"))
    parser.add_argument("--log-path", type=Path, default=Path("artifacts/runs/factscore_chatgpt_materialization_remote.log"))
    parser.add_argument("--gpu", default="A100-80GB")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--self-report-chunk-size", type=int, default=20)
    args = parser.parse_args()

    load_repo_env()
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/modal_gpu.py",
        "--gpu",
        args.gpu,
        "--timeout",
        str(args.timeout),
        "--sync-extra",
        "inference",
        "--",
        "python",
        "scripts/remote_factscore_materialize.py",
        "--examples-jsonl",
        str(args.examples_jsonl),
        "--claims-jsonl",
        str(args.claims_jsonl),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--self-report-chunk-size",
        str(args.self_report_chunk_size),
    ]
    if args.start_index or args.end_index:
        cmd.extend(["--start-index", str(args.start_index), "--end-index", str(args.end_index)])
    if args.max_examples:
        cmd.extend(["--max-examples", str(args.max_examples)])

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    ensure_parent(args.log_path)
    args.log_path.write_text(result.stdout + ("\n--- STDERR ---\n" + result.stderr if result.stderr else ""), encoding="utf-8")
    if result.returncode != 0:
        raise SystemExit(
            "Remote FActScore materialization failed. "
            f"See {args.log_path}. Tail: {result.stderr[-500:] or result.stdout[-500:]}"
        )

    summary = json.loads(_extract_section(result.stdout, "SUMMARY_JSON"))
    answer_runs = _jsonl_section(result.stdout, "ANSWER_RUNS_JSONL")
    tokenized_claims = _jsonl_section(result.stdout, "TOKENIZED_CLAIMS_JSONL")
    raw_self_reports = _jsonl_section(result.stdout, "RAW_SELF_REPORT_JSONL")
    parsed_self_reports = _jsonl_section(result.stdout, "PARSED_SELF_REPORT_JSONL")
    features_payload = base64.b64decode(_extract_section(result.stdout, "FEATURES_NPZ_B64"))

    write_json(args.summary_json, summary)
    write_jsonl(args.answer_runs_jsonl, answer_runs)
    write_jsonl(args.tokenized_claims_jsonl, tokenized_claims)
    ensure_parent(args.features_npz)
    args.features_npz.write_bytes(features_payload)
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    for row in raw_self_reports:
        name = f"{row['example_id']}_chunk{int(row.get('chunk_index', 0)):03d}.txt"
        (args.raw_dir / name).write_text(row["raw_text"], encoding="utf-8")
    write_jsonl(args.parsed_jsonl, parsed_self_reports)

    npz = np.load(args.features_npz, allow_pickle=False)
    print(f"Wrote FActScore materialization summary to {args.summary_json}")
    print(f"Wrote {len(answer_runs)} answer runs to {args.answer_runs_jsonl}")
    print(f"Wrote {len(tokenized_claims)} tokenized claims to {args.tokenized_claims_jsonl}")
    print(f"Wrote residual feature matrix {npz['matrix'].shape} to {args.features_npz}")
    print(f"Wrote {len(parsed_self_reports)} parsed self-report predictions to {args.parsed_jsonl}")


if __name__ == "__main__":
    main()
