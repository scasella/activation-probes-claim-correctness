from __future__ import annotations

import argparse
import subprocess
from collections import defaultdict
from pathlib import Path

from interp_experiment.io import read_json, read_jsonl, write_json, write_jsonl
from interp_experiment.schemas import ClaimRow, ExampleRow
from interp_experiment.utils import chunked, ensure_parent


def _load_examples(path: Path) -> list[ExampleRow]:
    return [ExampleRow.from_dict(row) for row in read_jsonl(path)]


def _load_claims_by_example(path: Path) -> dict[str, list[ClaimRow]]:
    claims_by_example: dict[str, list[ClaimRow]] = defaultdict(list)
    for row in read_jsonl(path):
        claim = ClaimRow.from_dict(row)
        claims_by_example[claim.example_id].append(claim)
    return claims_by_example


def _batch_root(
    output_jsonl: Path,
    *,
    batch_size: int,
    n_samples: int,
    n_paraphrases: int,
    override: Path | None = None,
) -> Path:
    if override is not None:
        return override
    suffix = f"bs{batch_size}_ns{n_samples}_np{n_paraphrases}"
    return output_jsonl.parent / f"{output_jsonl.stem}_batches_{suffix}"


def _run_batch(
    *,
    batch_index: int,
    batch_examples: list[ExampleRow],
    claims_by_example: dict[str, list[ClaimRow]],
    batch_root: Path,
    gpu: str,
    timeout: int,
    n_samples: int,
    n_paraphrases: int,
    extractor_backend: str,
) -> dict[str, object]:
    batch_dir = batch_root / f"batch_{batch_index:03d}"
    examples_path = batch_dir / "examples.jsonl"
    claims_path = batch_dir / "claims.jsonl"
    output_path = batch_dir / "targets.jsonl"
    summary_path = batch_dir / "summary.json"
    log_path = batch_dir / "remote.log"

    if output_path.exists() and summary_path.exists():
        return {
            "batch_index": batch_index,
            "status": "reused",
            "examples_path": str(examples_path),
            "claims_path": str(claims_path),
            "output_path": str(output_path),
            "summary_path": str(summary_path),
            "log_path": str(log_path),
        }

    batch_claims: list[dict[str, object]] = []
    for example in batch_examples:
        batch_claims.extend(claim.as_dict() for claim in claims_by_example[example.example_id])

    write_jsonl(examples_path, [example.as_dict() for example in batch_examples])
    write_jsonl(claims_path, batch_claims)

    cmd = [
        "uv",
        "run",
        "python",
        "scripts/materialize_claim_targets.py",
        "--examples-jsonl",
        str(examples_path),
        "--claims-jsonl",
        str(claims_path),
        "--output-jsonl",
        str(output_path),
        "--summary-json",
        str(summary_path),
        "--log-path",
        str(log_path),
        "--gpu",
        gpu,
        "--timeout",
        str(timeout),
        "--n-samples",
        str(n_samples),
        "--n-paraphrases",
        str(n_paraphrases),
        "--extractor-backend",
        extractor_backend,
    ]
    result = subprocess.run(cmd, cwd=str(Path.cwd()), capture_output=True, text=True, check=False)
    return {
        "batch_index": batch_index,
        "status": "succeeded" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "examples_path": str(examples_path),
        "claims_path": str(claims_path),
        "output_path": str(output_path),
        "summary_path": str(summary_path),
        "log_path": str(log_path),
        "stdout_tail": result.stdout[-400:],
        "stderr_tail": result.stderr[-400:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize claim targets in small resumable batches.")
    parser.add_argument("--examples-jsonl", type=Path, required=True)
    parser.add_argument("--claims-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--gpu", default="A100-80GB")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--n-paraphrases", type=int, default=3)
    parser.add_argument("--extractor-backend", choices=["auto", "huggingface"], default="auto")
    parser.add_argument("--batch-root", type=Path, default=None)
    parser.add_argument("--start-batch", type=int, default=1)
    parser.add_argument("--end-batch", type=int, default=0)
    parser.add_argument("--no-merge", action="store_true")
    args = parser.parse_args()
    if args.start_batch < 1:
        raise SystemExit("--start-batch must be >= 1")
    if args.end_batch and args.end_batch < args.start_batch:
        raise SystemExit("--end-batch must be >= --start-batch")

    examples = _load_examples(args.examples_jsonl)
    claims_by_example = _load_claims_by_example(args.claims_jsonl)
    batch_root = _batch_root(
        args.output_jsonl,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
        n_paraphrases=args.n_paraphrases,
        override=args.batch_root,
    )
    ensure_parent(args.output_jsonl)
    batch_root.mkdir(parents=True, exist_ok=True)

    batch_examples_list = list(enumerate(chunked(examples, args.batch_size), start=1))
    if args.max_batches:
        batch_examples_list = batch_examples_list[: args.max_batches]
    batch_examples_list = [
        (batch_index, batch_examples)
        for batch_index, batch_examples in batch_examples_list
        if batch_index >= args.start_batch and (not args.end_batch or batch_index <= args.end_batch)
    ]

    batch_results: list[dict[str, object]] = []
    for batch_index, batch_examples in batch_examples_list:
        batch_results.append(
            _run_batch(
                batch_index=batch_index,
                batch_examples=batch_examples,
                claims_by_example=claims_by_example,
                batch_root=batch_root,
                gpu=args.gpu,
                timeout=args.timeout,
                n_samples=args.n_samples,
                n_paraphrases=args.n_paraphrases,
                extractor_backend=args.extractor_backend,
            )
        )

    merged_rows: list[dict[str, object]] = []
    n_examples_succeeded = 0
    n_examples_failed = 0
    failures: list[dict[str, object]] = []
    for batch_result in batch_results:
        summary_path = Path(str(batch_result["summary_path"]))
        output_path = Path(str(batch_result["output_path"]))
        if batch_result["status"] == "failed" or not summary_path.exists() or not output_path.exists():
            n_examples_failed += len(list(read_jsonl(Path(str(batch_result["examples_path"])))))
            failures.append(batch_result)
            continue
        summary = read_json(summary_path)
        merged_rows.extend(read_jsonl(output_path))
        n_examples_succeeded += int(summary["n_examples"])

    if not args.no_merge:
        write_jsonl(args.output_jsonl, merged_rows)
    payload = {
        "n_examples_total": len(examples),
        "n_examples_succeeded": n_examples_succeeded,
        "n_examples_failed": n_examples_failed,
        "n_rows": len(merged_rows),
        "batch_size": args.batch_size,
        "n_batches": len(batch_results),
        "gpu": args.gpu,
        "timeout": args.timeout,
        "n_samples": args.n_samples,
        "n_paraphrases": args.n_paraphrases,
        "extractor_backend": args.extractor_backend,
        "start_batch": args.start_batch,
        "end_batch": args.end_batch,
        "batch_root": str(batch_root),
        "no_merge": args.no_merge,
        "batch_results": batch_results,
        "failures": failures,
    }
    write_json(args.summary_json, payload)
    if args.no_merge:
        print(f"Materialized {len(batch_results)} claim target batches under {batch_root}")
    else:
        print(f"Wrote {len(merged_rows)} claim target rows to {args.output_jsonl}")
    print(f"Wrote batched claim target summary to {args.summary_json}")


if __name__ == "__main__":
    main()
