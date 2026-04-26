from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

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
    parser = argparse.ArgumentParser(description="Run matched FELM generation in a Modal GPU sandbox.")
    parser.add_argument("--examples-jsonl", type=Path, default=Path("data/felm/felm_wk_matched_seed_examples.jsonl"))
    parser.add_argument("--output-examples-jsonl", type=Path, default=Path("data/felm/felm_wk_matched_examples.jsonl"))
    parser.add_argument("--answer-runs-jsonl", type=Path, default=Path("artifacts/runs/felm_wk_matched_answer_runs.jsonl"))
    parser.add_argument("--summary-json", type=Path, default=Path("artifacts/runs/felm_wk_matched_generation_summary.json"))
    parser.add_argument("--log-path", type=Path, default=Path("artifacts/runs/felm_wk_matched_generation_remote.log"))
    parser.add_argument("--gpu", default="A100-80GB")
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    args = parser.parse_args()

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
        "scripts/remote_felm_matched_generate.py",
        "--examples-jsonl",
        str(args.examples_jsonl),
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    if args.start_index or args.end_index:
        cmd.extend(["--start-index", str(args.start_index), "--end-index", str(args.end_index)])
    if args.max_examples:
        cmd.extend(["--max-examples", str(args.max_examples)])

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    ensure_parent(args.log_path)
    args.log_path.write_text(
        result.stdout + ("\n--- STDERR ---\n" + result.stderr if result.stderr else ""),
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise SystemExit(
            "Remote matched FELM generation failed. "
            f"See {args.log_path}. Tail: {result.stderr[-500:] or result.stdout[-500:]}"
        )

    summary = json.loads(_extract_section(result.stdout, "SUMMARY_JSON"))
    examples = _jsonl_section(result.stdout, "EXAMPLES_JSONL")
    answer_runs = _jsonl_section(result.stdout, "ANSWER_RUNS_JSONL")
    write_json(args.summary_json, summary)
    write_jsonl(args.output_examples_jsonl, examples)
    write_jsonl(args.answer_runs_jsonl, answer_runs)
    print(f"Wrote matched generated examples to {args.output_examples_jsonl}")
    print(f"Wrote matched answer runs to {args.answer_runs_jsonl}")
    print(f"Wrote matched generation summary to {args.summary_json}")


if __name__ == "__main__":
    main()
