from __future__ import annotations

import argparse
import json
from pathlib import Path

from interp_experiment.utils import ensure_parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a Markdown report for the MAUD proxy dry run.")
    parser.add_argument("--baseline-json", type=Path, required=True)
    parser.add_argument("--probe-json", type=Path, required=True)
    parser.add_argument("--judge-summary-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    baseline = json.loads(args.baseline_json.read_text())
    probe = json.loads(args.probe_json.read_text())
    judge = json.loads(args.judge_summary_json.read_text())

    lines = [
        "# MAUD Proxy Dry Run Report",
        "",
        "This report is based on `judge_llm_proxy` labels and is not expert-ground-truth evaluation.",
        "",
        "## Judge Proxy Summary",
        "",
        f"- Model: `{judge['model']}`",
        f"- Proxy label rows: `{judge['n_rows']}`",
        f"- Examples succeeded: `{judge['n_examples_succeeded']}/{judge['n_examples']}`",
        "",
        "## Baseline Results",
        "",
        "| Method | Correctness AUROC | Correctness Brier |",
        "| --- | --- | --- |",
        f"| llama_self_report | {baseline['llama_self_report']['correctness']['auroc']} | {baseline['llama_self_report']['correctness']['brier']} |",
        f"| gpt54_cached | {baseline['gpt54_cached']['correctness']['auroc']} | {baseline['gpt54_cached']['correctness']['brier']} |",
        "",
        "## Probe Results",
        "",
        "| Feature source | Correctness AUROC | Correctness Brier |",
        "| --- | --- | --- |",
        f"| residual | {probe['residual']['correctness']['auroc']} | {probe['residual']['correctness']['brier']} |",
        f"| sae | {probe['sae']['correctness']['auroc']} | {probe['sae']['correctness']['brier']} |",
        "",
        "## Current Read",
        "",
        "- On this MAUD proxy slice, `gpt54_cached` is strongest.",
        "- Residual probe features are currently stronger than SAE features on the same proxy labels.",
        "- Llama self-report remains the weakest of the currently materialized comparison lanes.",
        "",
        "## Caveats",
        "",
        "- Proxy labels are not expert labels.",
        "- This is a dry-run infrastructure result, not the final study claim.",
        "- Load-bearing results remain underpowered or degenerate on the proxy lane when the labels collapse to one class.",
        "",
    ]
    ensure_parent(args.output_md)
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote MAUD proxy report to {args.output_md}")


if __name__ == "__main__":
    main()
