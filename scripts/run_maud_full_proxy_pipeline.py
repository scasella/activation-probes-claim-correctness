from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> None:
    result = subprocess.run(cmd, cwd=str(cwd), check=False, text=True)
    if result.returncode != 0:
        raise SystemExit(f"Command failed ({result.returncode}): {' '.join(cmd)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full MAUD proxy dry-run pipeline.")
    parser.add_argument("--source-jsonl", type=Path, default=Path("data/source_pool/examples.jsonl"))
    parser.add_argument("--workdir-root", type=Path, default=Path("artifacts/runs"))
    args = parser.parse_args()

    cwd = Path.cwd()
    root = args.workdir_root
    maud_examples = root / "maud_full_examples.jsonl"
    maud_examples_with_answers = root / "maud_full_examples_with_answers.jsonl"
    maud_answer_runs = root / "maud_full_answer_runs.jsonl"
    maud_claims = root / "maud_full_claims.jsonl"
    maud_targets = root / "maud_full_claim_targets.jsonl"
    maud_residual = root / "maud_full_probe_features_residual.jsonl"
    maud_sae = root / "maud_full_probe_features_sae.npz"

    _run(
        [
            "uv",
            "run",
            "python",
            "scripts/export_example_subset.py",
            "--input-jsonl",
            str(args.source_jsonl),
            "--output-jsonl",
            str(maud_examples),
            "--source-corpus",
            "maud",
            "--include-splits",
            "train",
            "validation",
            "test",
        ],
        cwd,
    )
    _run(
        [
            "uv",
            "run",
            "python",
            "scripts/materialize_remote_pilot.py",
            "--input-jsonl",
            str(maud_examples),
            "--output-examples-jsonl",
            str(maud_examples_with_answers),
            "--output-answer-runs-jsonl",
            str(maud_answer_runs),
            "--output-claims-jsonl",
            str(maud_claims),
            "--summary-json",
            str(root / "maud_full_generation_summary.json"),
            "--log-path",
            str(root / "maud_full_generation_remote.log"),
        ],
        cwd,
    )
    _run(
        [
            "uv",
            "run",
            "python",
            "scripts/materialize_judge_annotations.py",
            "--examples-jsonl",
            str(maud_examples_with_answers),
            "--claims-jsonl",
            str(maud_claims),
            "--output-jsonl",
            "data/annotations/maud_full_judge_annotations.jsonl",
            "--summary-json",
            str(root / "maud_full_judge_annotation_summary.json"),
            "--raw-dir",
            "data/annotations/judge_llm_raw/maud_full",
            "--log-dir",
            str(root / "maud_full_judge_logs"),
        ],
        cwd,
    )
    _run(
        [
            "uv",
            "run",
            "python",
            "scripts/materialize_llama_self_report.py",
            "--examples-jsonl",
            str(maud_examples_with_answers),
            "--claims-jsonl",
            str(maud_claims),
            "--raw-dir",
            "data/cached_baselines/llama_self_report/raw/maud_full",
            "--parsed-dir",
            "data/cached_baselines/llama_self_report/parsed/maud_full",
            "--summary-json",
            str(root / "maud_full_llama_self_report_summary.json"),
            "--log-path",
            str(root / "maud_full_llama_self_report_remote.log"),
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
            str(maud_examples_with_answers),
            "--claims-jsonl",
            str(maud_claims),
            "--requests-dir",
            "data/cached_baselines/gpt54/requests/maud_full",
            "--raw-dir",
            "data/cached_baselines/gpt54/raw/maud_full",
            "--parsed-dir",
            "data/cached_baselines/gpt54/parsed/maud_full",
            "--summary-json",
            str(root / "maud_full_gpt54_summary.json"),
            "--log-dir",
            str(root / "maud_full_gpt54_logs"),
        ],
        cwd,
    )
    _run(
        [
            "uv",
            "run",
            "python",
            "scripts/materialize_claim_targets.py",
            "--examples-jsonl",
            str(maud_examples_with_answers),
            "--claims-jsonl",
            str(maud_claims),
            "--output-jsonl",
            str(maud_targets),
            "--summary-json",
            str(root / "maud_full_claim_targets_summary.json"),
            "--log-path",
            str(root / "maud_full_claim_targets_remote.log"),
        ],
        cwd,
    )
    _run(
        [
            "uv",
            "run",
            "python",
            "scripts/materialize_probe_features.py",
            "--answer-runs-jsonl",
            str(maud_answer_runs),
            "--claims-jsonl",
            str(maud_claims),
            "--output-jsonl",
            str(maud_residual),
            "--summary-json",
            str(root / "maud_full_probe_features_residual_summary.json"),
            "--feature-source",
            "residual",
            "--log-path",
            str(root / "maud_full_probe_features_residual_remote.log"),
        ],
        cwd,
    )
    _run(
        [
            "uv",
            "run",
            "python",
            "scripts/materialize_probe_features.py",
            "--answer-runs-jsonl",
            str(maud_answer_runs),
            "--claims-jsonl",
            str(maud_claims),
            "--output-jsonl",
            str(maud_sae),
            "--summary-json",
            str(root / "maud_full_probe_features_sae_summary.json"),
            "--feature-source",
            "sae",
            "--log-path",
            str(root / "maud_full_probe_features_sae_remote.log"),
        ],
        cwd,
    )
    _run(
        [
            "uv",
            "run",
            "python",
            "scripts/evaluate_baselines_against_labels.py",
            "--labels-jsonl",
            "data/annotations/maud_full_judge_annotations.jsonl",
            "--llama-predictions",
            "data/cached_baselines/llama_self_report/parsed/maud_full/_all_predictions.jsonl",
            "--gpt54-predictions",
            "data/cached_baselines/gpt54/parsed/maud_full",
            "--output-json",
            str(root / "maud_full_proxy_baseline_eval.json"),
            "--label-source",
            "judge_llm_proxy",
        ],
        cwd,
    )
    _run(
        [
            "uv",
            "run",
            "python",
            "scripts/evaluate_probe_proxy_smoke.py",
            "--labels-jsonl",
            "data/annotations/maud_full_judge_annotations.jsonl",
            "--residual-features-jsonl",
            str(maud_residual),
            "--sae-features-path",
            str(maud_sae),
            "--output-json",
            str(root / "maud_full_probe_proxy_smoke.json"),
        ],
        cwd,
    )
    _run(
        [
            "uv",
            "run",
            "python",
            "scripts/run_maud_proxy_dry_run.py",
            "--examples-jsonl",
            str(maud_examples_with_answers),
            "--labels-jsonl",
            "data/annotations/maud_full_judge_annotations.jsonl",
            "--targets-jsonl",
            str(maud_targets),
            "--residual-features-jsonl",
            str(maud_residual),
            "--sae-features-path",
            str(maud_sae),
            "--output-json",
            str(root / "maud_full_proxy_dry_run.json"),
        ],
        cwd,
    )
    _run(
        [
            "uv",
            "run",
            "python",
            "scripts/render_maud_proxy_report.py",
            "--baseline-json",
            str(root / "maud_full_proxy_baseline_eval.json"),
            "--probe-json",
            str(root / "maud_full_probe_proxy_smoke.json"),
            "--judge-summary-json",
            str(root / "maud_full_judge_annotation_summary.json"),
            "--output-md",
            str(root / "maud_full_proxy_report.md"),
        ],
        cwd,
    )


if __name__ == "__main__":
    main()
