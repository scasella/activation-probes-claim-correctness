from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="Materialize claim targets locally via a remote Modal run.")
    parser.add_argument("--examples-jsonl", type=Path, required=True)
    parser.add_argument("--claims-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--gpu", default="A100-80GB")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--n-paraphrases", type=int, default=3)
    parser.add_argument("--extractor-backend", choices=["auto", "huggingface"], default="auto")
    args = parser.parse_args()

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
        "scripts/remote_claim_targets.py",
        "--examples-jsonl",
        str(args.examples_jsonl),
        "--claims-jsonl",
        str(args.claims_jsonl),
        "--n-samples",
        str(args.n_samples),
        "--n-paraphrases",
        str(args.n_paraphrases),
        "--extractor-backend",
        args.extractor_backend,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    args.log_path.write_text(
        result.stdout + ("\n--- STDERR ---\n" + result.stderr if result.stderr else ""),
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise SystemExit(
            "Remote claim target materialization failed. "
            f"See {args.log_path} for details. Tail: {result.stderr[-400:] or result.stdout[-400:]}"
        )

    summary = json.loads(_extract_section(result.stdout, "SUMMARY_JSON"))
    rows = [json.loads(line) for line in _extract_section(result.stdout, "TARGETS_JSONL").splitlines() if line.strip()]
    ensure_parent(args.output_jsonl)
    args.output_jsonl.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8")
    ensure_parent(args.summary_json)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} target rows to {args.output_jsonl}")
    print(f"Wrote target summary to {args.summary_json}")


if __name__ == "__main__":
    main()
