from __future__ import annotations

import argparse
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from interp_experiment.io import read_json, read_jsonl, write_json, write_jsonl


def _load_examples(path: Path) -> list[dict[str, object]]:
    return list(read_jsonl(path))


def _load_claims_by_example(path: Path) -> dict[str, list[dict[str, object]]]:
    claims_by_example: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in read_jsonl(path):
        claims_by_example[str(row["example_id"])].append(row)
    return claims_by_example


def _done_ids_from_raw(raw_dir: Path) -> set[str]:
    return {path.stem for path in raw_dir.glob("*.json")}


def _batch_chunks(items: list[str], batch_size: int) -> list[list[str]]:
    return [items[idx : idx + batch_size] for idx in range(0, len(items), batch_size)]


def _build_command(
    *,
    lane: str,
    batch_examples: Path,
    batch_claims: Path,
    batch_output_root: Path,
    raw_dir: Path,
    turn_timeout_sec: float,
    requests_dir: Path | None,
    parsed_dir: Path | None,
) -> list[str]:
    common = [
        sys.executable,
    ]
    if lane == "judge":
        return common + [
            "scripts/materialize_judge_annotations.py",
            "--examples-jsonl",
            str(batch_examples),
            "--claims-jsonl",
            str(batch_claims),
            "--output-jsonl",
            str(batch_output_root.with_suffix(".jsonl")),
            "--summary-json",
            str(batch_output_root.with_name(batch_output_root.name + "_summary.json")),
            "--raw-dir",
            str(raw_dir),
            "--log-dir",
            str(batch_output_root.parent / "logs"),
            "--skip-existing",
            "--turn-timeout-sec",
            str(turn_timeout_sec),
        ]
    if lane == "gpt54":
        if requests_dir is None or parsed_dir is None:
            raise ValueError("requests_dir and parsed_dir are required for gpt54 lane")
        return common + [
            "scripts/materialize_gpt54_baseline.py",
            "--examples-jsonl",
            str(batch_examples),
            "--claims-jsonl",
            str(batch_claims),
            "--requests-dir",
            str(requests_dir),
            "--raw-dir",
            str(raw_dir),
            "--parsed-dir",
            str(parsed_dir),
            "--summary-json",
            str(batch_output_root.with_name(batch_output_root.name + "_summary.json")),
            "--log-dir",
            str(batch_output_root.parent / "logs"),
            "--skip-existing",
            "--turn-timeout-sec",
            str(turn_timeout_sec),
        ]
    raise ValueError(f"Unsupported lane: {lane}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Retry remainder-only batches for judge or GPT full-corpus materialization.")
    parser.add_argument("--lane", choices=["judge", "gpt54"], required=True)
    parser.add_argument("--examples-jsonl", type=Path, required=True)
    parser.add_argument("--claims-jsonl", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--turn-timeout-sec", type=float, default=240.0)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--requests-dir", type=Path, default=None)
    parser.add_argument("--parsed-dir", type=Path, default=None)
    args = parser.parse_args()

    examples = _load_examples(args.examples_jsonl)
    claims_by_example = _load_claims_by_example(args.claims_jsonl)
    done_ids = _done_ids_from_raw(args.raw_dir)
    ordered_ids = [str(row["example_id"]) for row in examples]
    remaining_ids = [example_id for example_id in ordered_ids if example_id not in done_ids]

    args.output_root.mkdir(parents=True, exist_ok=True)
    batches_dir = args.output_root / "batches"
    batches_dir.mkdir(parents=True, exist_ok=True)

    progress_path = args.output_root / "progress.json"
    progress = {
        "lane": args.lane,
        "n_examples_total": len(ordered_ids),
        "n_examples_done_initial": len(done_ids),
        "n_examples_remaining_initial": len(remaining_ids),
        "batch_size": args.batch_size,
        "turn_timeout_sec": args.turn_timeout_sec,
        "batches": [],
    }
    write_json(progress_path, progress)

    if not remaining_ids:
        print(f"No remaining examples for lane={args.lane}")
        return

    batches = _batch_chunks(remaining_ids, args.batch_size)
    if args.max_batches:
        batches = batches[: args.max_batches]

    for batch_index, batch_ids in enumerate(batches, start=1):
        batch_examples_rows = [row for row in examples if str(row["example_id"]) in set(batch_ids)]
        batch_claim_rows = [claim for example_id in batch_ids for claim in claims_by_example[example_id]]
        batch_examples_path = batches_dir / f"batch_{batch_index:03d}_examples.jsonl"
        batch_claims_path = batches_dir / f"batch_{batch_index:03d}_claims.jsonl"
        batch_root = batches_dir / f"batch_{batch_index:03d}"
        batch_log_path = batches_dir / f"batch_{batch_index:03d}.log"
        batch_summary_path = batches_dir / f"batch_{batch_index:03d}_summary.json"

        write_jsonl(batch_examples_path, batch_examples_rows)
        write_jsonl(batch_claims_path, batch_claim_rows)

        before_done = len(_done_ids_from_raw(args.raw_dir))
        cmd = _build_command(
            lane=args.lane,
            batch_examples=batch_examples_path,
            batch_claims=batch_claims_path,
            batch_output_root=batch_root,
            raw_dir=args.raw_dir,
            turn_timeout_sec=args.turn_timeout_sec,
            requests_dir=args.requests_dir,
            parsed_dir=args.parsed_dir,
        )
        result = subprocess.run(cmd, cwd=str(Path.cwd()), capture_output=True, text=True, check=False)
        batch_log_path.write_text(
            result.stdout + ("\n--- STDERR ---\n" + result.stderr if result.stderr else ""),
            encoding="utf-8",
        )
        after_done = len(_done_ids_from_raw(args.raw_dir))
        batch_summary = read_json(batch_summary_path) if batch_summary_path.exists() else None
        progress["batches"].append(
            {
                "batch_index": batch_index,
                "example_ids": batch_ids,
                "returncode": result.returncode,
                "before_done": before_done,
                "after_done": after_done,
                "delta_done": after_done - before_done,
                "batch_log_path": str(batch_log_path),
                "batch_summary_path": str(batch_summary_path),
                "batch_summary": batch_summary,
            }
        )
        write_json(progress_path, progress)
        print(
            f"RETRY_BATCH lane={args.lane} batch={batch_index} "
            f"n_examples={len(batch_ids)} delta_done={after_done - before_done} returncode={result.returncode}"
        )


if __name__ == "__main__":
    main()
