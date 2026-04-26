from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from interp_experiment.io import read_jsonl, write_json, write_jsonl


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _batch_complete(summary_path: Path, expected_claims: int) -> bool:
    if not summary_path.exists():
        return False
    try:
        summary = _read_json(summary_path)
    except json.JSONDecodeError:
        return False
    return (
        summary.get("n_expected_claims", summary.get("n_expected_segments")) == expected_claims
        and summary.get("n_feature_rows") == expected_claims
        and summary.get("n_tokenized_claims", summary.get("n_tokenized_segments")) == expected_claims
        and summary.get("n_self_report_predictions") == expected_claims
        and summary.get("claim_failure_rate", summary.get("segment_failure_rate")) == 0
        and summary.get("self_report_failure_rate") == 0
    )


def _run_batch(args: argparse.Namespace, batch_dir: Path, start: int, end: int) -> None:
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/materialize_factscore_validation.py",
        "--examples-jsonl",
        str(args.examples_jsonl),
        "--claims-jsonl",
        str(args.claims_jsonl),
        "--answer-runs-jsonl",
        str(batch_dir / "answer_runs.jsonl"),
        "--tokenized-claims-jsonl",
        str(batch_dir / "claims_tokenized.jsonl"),
        "--features-npz",
        str(batch_dir / "features_residual.npz"),
        "--raw-dir",
        str(batch_dir / "raw_self_report"),
        "--parsed-jsonl",
        str(batch_dir / "parsed_self_report.jsonl"),
        "--summary-json",
        str(batch_dir / "summary.json"),
        "--log-path",
        str(batch_dir / "remote.log"),
        "--gpu",
        args.gpu,
        "--timeout",
        str(args.timeout),
        "--start-index",
        str(start),
        "--end-index",
        str(end),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--self-report-chunk-size",
        str(args.self_report_chunk_size),
    ]
    subprocess.run(cmd, check=True)


def _merge_batches(args: argparse.Namespace, batch_dirs: list[Path]) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    answer_runs: list[dict[str, Any]] = []
    tokenized_claims: list[dict[str, Any]] = []
    parsed_self_reports: list[dict[str, Any]] = []
    matrices: list[np.ndarray] = []
    claim_ids: list[str] = []
    example_ids: list[str] = []

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    for batch_dir in batch_dirs:
        summary = _read_json(batch_dir / "summary.json")
        summaries.append(summary)
        answer_runs.extend(read_jsonl(batch_dir / "answer_runs.jsonl"))
        tokenized_claims.extend(read_jsonl(batch_dir / "claims_tokenized.jsonl"))
        parsed_self_reports.extend(read_jsonl(batch_dir / "parsed_self_report.jsonl"))
        features = np.load(batch_dir / "features_residual.npz", allow_pickle=False)
        matrices.append(np.asarray(features["matrix"], dtype=np.float32))
        claim_ids.extend(str(item) for item in features["claim_ids"].tolist())
        example_ids.extend(str(item) for item in features["example_ids"].tolist())
        for raw_file in sorted((batch_dir / "raw_self_report").glob("*.txt")):
            shutil.copyfile(raw_file, args.raw_dir / raw_file.name)

    matrix = np.concatenate(matrices, axis=0) if matrices else np.zeros((0, 0), dtype=np.float32)
    write_jsonl(args.answer_runs_jsonl, answer_runs)
    write_jsonl(args.tokenized_claims_jsonl, tokenized_claims)
    write_jsonl(args.parsed_jsonl, parsed_self_reports)
    args.features_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.features_npz, matrix=matrix, claim_ids=np.asarray(claim_ids, dtype=str), example_ids=np.asarray(example_ids, dtype=str))

    expected_claims = sum(int(summary.get("n_expected_claims", summary.get("n_expected_segments", 0))) for summary in summaries)
    feature_rows = sum(int(summary.get("n_feature_rows", 0)) for summary in summaries)
    self_report_predictions = sum(int(summary.get("n_self_report_predictions", 0)) for summary in summaries)
    failures = [
        {**failure, "batch_index": batch_index}
        for batch_index, summary in enumerate(summaries)
        for failure in summary.get("failures", [])
    ]
    normalization_notes = [
        {**note, "batch_index": batch_index}
        for batch_index, summary in enumerate(summaries)
        for note in summary.get("normalization_notes", [])
    ]
    token_span_lengths = [summary.get("token_span_lengths", {}) for summary in summaries]
    merged_summary = {
        "batched": True,
        "batch_size": args.batch_size,
        "n_batches": len(batch_dirs),
        "batch_summaries": [str(batch_dir / "summary.json") for batch_dir in batch_dirs],
        "extractor_name": summaries[0].get("extractor_name") if summaries else None,
        "model_name": summaries[0].get("model_name") if summaries else None,
        "n_examples": sum(int(summary.get("n_examples", 0)) for summary in summaries),
        "n_answer_runs": sum(int(summary.get("n_answer_runs", 0)) for summary in summaries),
        "n_expected_claims": expected_claims,
        "n_expected_segments": expected_claims,
        "n_tokenized_claims": sum(int(summary.get("n_tokenized_claims", summary.get("n_tokenized_segments", 0))) for summary in summaries),
        "n_tokenized_segments": sum(int(summary.get("n_tokenized_claims", summary.get("n_tokenized_segments", 0))) for summary in summaries),
        "n_feature_rows": feature_rows,
        "n_self_report_predictions": self_report_predictions,
        "n_failures": len(failures),
        "failures": failures,
        "n_normalization_notes": len(normalization_notes),
        "normalization_notes": normalization_notes,
        "claim_failure_rate": 0.0 if expected_claims == 0 else 1.0 - (feature_rows / expected_claims),
        "segment_failure_rate": 0.0 if expected_claims == 0 else 1.0 - (feature_rows / expected_claims),
        "self_report_failure_rate": 0.0 if expected_claims == 0 else 1.0 - (self_report_predictions / expected_claims),
        "batch_token_span_length_summaries": token_span_lengths,
        "self_report_chunk_size": args.self_report_chunk_size,
        "transport_note": "Full FActScore materialization is batched to avoid large stdout payload truncation.",
    }
    write_json(args.summary_json, merged_summary)
    return merged_summary


def _batch_range(batch_dir: Path) -> tuple[int, int] | None:
    match = re.match(r"batch_\d+_(\d+)_(\d+)$", batch_dir.name)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _collect_complete_batch_dirs(args: argparse.Namespace, examples: list[dict[str, Any]], claims_by_example: dict[str, int]) -> list[Path]:
    complete: list[tuple[int, int, Path]] = []
    for batch_dir in sorted(args.batch_dir.glob("batch_*_*_*")):
        batch_range = _batch_range(batch_dir)
        if batch_range is None:
            continue
        start, end = batch_range
        batch_examples = examples[start:end]
        expected_claims = sum(claims_by_example.get(row["example_id"], 0) for row in batch_examples)
        if _batch_complete(batch_dir / "summary.json", expected_claims):
            complete.append((start, end, batch_dir))
    complete.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in complete]


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize FActScore validation artifacts in Modal batches.")
    parser.add_argument("--examples-jsonl", type=Path, default=Path("data/factscore/factscore_chatgpt_examples.jsonl"))
    parser.add_argument("--claims-jsonl", type=Path, default=Path("data/factscore/factscore_chatgpt_claims.jsonl"))
    parser.add_argument("--batch-dir", type=Path, default=Path("artifacts/runs/factscore_chatgpt_materialization_batches"))
    parser.add_argument("--batch-size", type=int, default=10)
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
    parser.add_argument("--gpu", default="A100-80GB")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--self-report-chunk-size", type=int, default=20)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=0)
    parser.add_argument("--merge-all-complete", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    examples = list(read_jsonl(args.examples_jsonl))
    claims_by_example: dict[str, int] = {}
    for row in read_jsonl(args.claims_jsonl):
        claims_by_example[row["example_id"]] = claims_by_example.get(row["example_id"], 0) + 1

    args.batch_dir.mkdir(parents=True, exist_ok=True)
    batch_dirs: list[Path] = []
    if args.merge_only:
        batch_dirs = _collect_complete_batch_dirs(args, examples, claims_by_example)
        summary = _merge_batches(args, batch_dirs)
        print(f"Wrote merged FActScore summary to {args.summary_json}")
        print(f"Wrote merged residual feature matrix to {args.features_npz}")
        print(
            "Merged counts: "
            f"{summary['n_examples']} examples, {summary['n_feature_rows']}/{summary['n_expected_claims']} feature rows, "
            f"{summary['n_self_report_predictions']}/{summary['n_expected_claims']} self-report predictions"
        )
        return

    global_start = args.start_index
    global_end = args.end_index if args.end_index else len(examples)
    n_batches = len(range(global_start, global_end, args.batch_size))
    for batch_index, start in enumerate(range(global_start, global_end, args.batch_size)):
        end = min(start + args.batch_size, len(examples))
        batch_examples = examples[start:end]
        expected_claims = sum(claims_by_example.get(row["example_id"], 0) for row in batch_examples)
        batch_dir = args.batch_dir / f"batch_{batch_index:04d}_{start:04d}_{end:04d}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        batch_dirs.append(batch_dir)
        if not args.force and _batch_complete(batch_dir / "summary.json", expected_claims):
            print(f"Reusing completed FActScore batch {batch_index + 1}/{n_batches}: {start}:{end}", flush=True)
            continue
        print(f"Running FActScore batch {batch_index + 1}/{n_batches}: {start}:{end}", flush=True)
        _run_batch(args, batch_dir, start, end)

    if args.merge_all_complete:
        batch_dirs = _collect_complete_batch_dirs(args, examples, claims_by_example)
    summary = _merge_batches(args, batch_dirs)
    print(f"Wrote merged FActScore summary to {args.summary_json}")
    print(f"Wrote merged residual feature matrix to {args.features_npz}")
    print(
        "Merged counts: "
        f"{summary['n_examples']} examples, {summary['n_feature_rows']}/{summary['n_expected_claims']} feature rows, "
        f"{summary['n_self_report_predictions']}/{summary['n_expected_claims']} self-report predictions"
    )


if __name__ == "__main__":
    main()
