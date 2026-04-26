from __future__ import annotations

import argparse
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

from interp_experiment.io import read_json, read_jsonl, write_json, write_jsonl
from interp_experiment.schemas import AnswerRunRow, ClaimRow
from interp_experiment.utils import chunked, ensure_parent


def _load_answer_runs(path: Path) -> list[AnswerRunRow]:
    return [AnswerRunRow.from_dict(row) for row in read_jsonl(path)]


def _load_claims_by_example(path: Path) -> dict[str, list[ClaimRow]]:
    claims_by_example: dict[str, list[ClaimRow]] = defaultdict(list)
    for row in read_jsonl(path):
        claim = ClaimRow.from_dict(row)
        claims_by_example[claim.example_id].append(claim)
    return claims_by_example


def _batch_root(output_path: Path) -> Path:
    return output_path.parent / f"{output_path.stem}_batches"


def _run_batch(
    *,
    batch_index: int,
    batch_runs: list[AnswerRunRow],
    claims_by_example: dict[str, list[ClaimRow]],
    batch_root: Path,
    feature_source: str,
    gpu: str,
    timeout: int,
) -> dict[str, object]:
    batch_dir = batch_root / f"batch_{batch_index:03d}"
    answer_runs_path = batch_dir / "answer_runs.jsonl"
    claims_path = batch_dir / "claims.jsonl"
    output_path = batch_dir / ("features.npz" if feature_source == "sae" else "features.jsonl")
    summary_path = batch_dir / "summary.json"
    log_path = batch_dir / "remote.log"

    if output_path.exists() and summary_path.exists():
        return {
            "batch_index": batch_index,
            "status": "reused",
            "answer_runs_path": str(answer_runs_path),
            "claims_path": str(claims_path),
            "output_path": str(output_path),
            "summary_path": str(summary_path),
            "log_path": str(log_path),
        }

    batch_claims: list[dict[str, object]] = []
    for answer_run in batch_runs:
        batch_claims.extend(claim.as_dict() for claim in claims_by_example[answer_run.example_id])

    write_jsonl(answer_runs_path, [answer_run.as_dict() for answer_run in batch_runs])
    write_jsonl(claims_path, batch_claims)

    cmd = [
        "uv",
        "run",
        "python",
        "scripts/materialize_probe_features.py",
        "--answer-runs-jsonl",
        str(answer_runs_path),
        "--claims-jsonl",
        str(claims_path),
        "--output-jsonl",
        str(output_path),
        "--summary-json",
        str(summary_path),
        "--feature-source",
        feature_source,
        "--gpu",
        gpu,
        "--timeout",
        str(timeout),
        "--log-path",
        str(log_path),
    ]
    result = subprocess.run(cmd, cwd=str(Path.cwd()), capture_output=True, text=True, check=False)
    return {
        "batch_index": batch_index,
        "status": "succeeded" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "answer_runs_path": str(answer_runs_path),
        "claims_path": str(claims_path),
        "output_path": str(output_path),
        "summary_path": str(summary_path),
        "log_path": str(log_path),
        "stdout_tail": result.stdout[-400:],
        "stderr_tail": result.stderr[-400:],
    }


def _merge_sae_batches(batch_results: list[dict[str, object]], output_path: Path) -> int:
    claim_ids: list[str] = []
    example_ids: list[str] = []
    matrices: list[np.ndarray] = []
    for batch_result in batch_results:
        output_file = Path(str(batch_result["output_path"]))
        if batch_result["status"] == "failed" or not output_file.exists():
            continue
        payload = np.load(output_file, allow_pickle=False)
        claim_ids.extend(str(item) for item in payload["claim_ids"].tolist())
        example_ids.extend(str(item) for item in payload["example_ids"].tolist())
        matrices.append(np.asarray(payload["matrix"], dtype=np.float32))
    if matrices:
        matrix = np.concatenate(matrices, axis=0)
    else:
        matrix = np.zeros((0, 0), dtype=np.float32)
    ensure_parent(output_path)
    np.savez_compressed(
        output_path,
        claim_ids=np.asarray(claim_ids),
        example_ids=np.asarray(example_ids),
        matrix=matrix,
    )
    return int(matrix.shape[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize probe features in small resumable batches.")
    parser.add_argument("--answer-runs-jsonl", type=Path, required=True)
    parser.add_argument("--claims-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--feature-source", choices=["residual", "sae"], required=True)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--gpu", default="A100-80GB")
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    answer_runs = _load_answer_runs(args.answer_runs_jsonl)
    claims_by_example = _load_claims_by_example(args.claims_jsonl)
    batch_root = _batch_root(args.output_jsonl)
    ensure_parent(args.output_jsonl)
    batch_root.mkdir(parents=True, exist_ok=True)

    batch_runs_list = list(chunked(answer_runs, args.batch_size))
    if args.max_batches:
        batch_runs_list = batch_runs_list[: args.max_batches]

    batch_results: list[dict[str, object]] = []
    for batch_index, batch_runs in enumerate(batch_runs_list, start=1):
        batch_results.append(
            _run_batch(
                batch_index=batch_index,
                batch_runs=batch_runs,
                claims_by_example=claims_by_example,
                batch_root=batch_root,
                feature_source=args.feature_source,
                gpu=args.gpu,
                timeout=args.timeout,
            )
        )

    failures: list[dict[str, object]] = []
    n_examples_succeeded = 0
    n_examples_failed = 0
    if args.feature_source == "sae":
        for batch_result in batch_results:
            output_file = Path(str(batch_result["output_path"]))
            if batch_result["status"] == "failed" or not output_file.exists():
                failures.append(batch_result)
                n_examples_failed += len(list(read_jsonl(Path(str(batch_result["answer_runs_path"])))))
        n_rows = _merge_sae_batches(batch_results, args.output_jsonl)
    else:
        merged_rows: list[dict[str, object]] = []
        for batch_result in batch_results:
            output_file = Path(str(batch_result["output_path"]))
            if batch_result["status"] == "failed" or not output_file.exists():
                failures.append(batch_result)
                n_examples_failed += len(list(read_jsonl(Path(str(batch_result["answer_runs_path"])))))
                continue
            merged_rows.extend(read_jsonl(output_file))
        write_jsonl(args.output_jsonl, merged_rows)
        n_rows = len(merged_rows)

    for batch_result in batch_results:
        summary_path = Path(str(batch_result["summary_path"]))
        if batch_result["status"] == "failed" or not summary_path.exists():
            continue
        summary = read_json(summary_path)
        n_examples_succeeded += int(summary["n_examples"])

    payload = {
        "feature_source": args.feature_source,
        "n_examples_total": len(answer_runs),
        "n_examples_succeeded": n_examples_succeeded,
        "n_examples_failed": n_examples_failed if failures else len(answer_runs) - n_examples_succeeded,
        "n_rows": n_rows,
        "batch_size": args.batch_size,
        "n_batches": len(batch_results),
        "gpu": args.gpu,
        "timeout": args.timeout,
        "batch_results": batch_results,
        "failures": failures,
    }
    write_json(args.summary_json, payload)
    print(f"Wrote {n_rows} {args.feature_source} feature rows to {args.output_jsonl}")
    print(f"Wrote batched {args.feature_source} feature summary to {args.summary_json}")


if __name__ == "__main__":
    main()
