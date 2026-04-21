from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

from interp_experiment.baselines.gpt54_cached import export_gpt54_request_packet
from interp_experiment.baselines.llama_self_report import parse_baseline_claims
from interp_experiment.baselines.utils import (
    build_baseline_output_schema,
    normalize_prediction_claim_ids,
    validate_prediction_claim_ids,
)
from interp_experiment.io import read_jsonl, write_json, write_jsonl
from interp_experiment.schemas import ClaimRow, ExampleRow
from interp_experiment.utils import ensure_parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize GPT-5.4 baseline artifacts via local Codex CLI.")
    parser.add_argument("--examples-jsonl", type=Path, default=Path("data/annotations/maud_pilot_examples.jsonl"))
    parser.add_argument("--claims-jsonl", type=Path, default=Path("data/annotations/maud_pilot_claims.jsonl"))
    parser.add_argument("--requests-dir", type=Path, default=Path("data/cached_baselines/gpt54/requests/maud_pilot"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/cached_baselines/gpt54/raw/maud_pilot"))
    parser.add_argument("--parsed-dir", type=Path, default=Path("data/cached_baselines/gpt54/parsed/maud_pilot"))
    parser.add_argument("--summary-json", type=Path, default=Path("artifacts/runs/maud_gpt54_summary.json"))
    parser.add_argument("--log-dir", type=Path, default=Path("artifacts/runs/maud_gpt54_logs"))
    parser.add_argument("--model", default="gpt-5.4")
    args = parser.parse_args()

    examples = {row.example_id: row for row in (ExampleRow.from_dict(item) for item in read_jsonl(args.examples_jsonl))}
    claims_by_example: dict[str, list[ClaimRow]] = defaultdict(list)
    for claim in (ClaimRow.from_dict(item) for item in read_jsonl(args.claims_jsonl)):
        claims_by_example[claim.example_id].append(claim)

    args.requests_dir.mkdir(parents=True, exist_ok=True)
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.parsed_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    failures: list[dict[str, object]] = []
    parsed_total = 0
    for example_id, example in examples.items():
        claims = claims_by_example[example_id]
        packet_path = args.requests_dir / f"{example_id}.json"
        packet = export_gpt54_request_packet(example, claims, packet_path)
        expected_claim_ids = [claim.claim_id for claim in claims]

        with tempfile.TemporaryDirectory(prefix=f"gpt54-{example_id}-") as tmp_dir:
            schema_path = Path(tmp_dir) / "schema.json"
            output_path = Path(tmp_dir) / "output.json"
            schema_path.write_text(json.dumps(build_baseline_output_schema(expected_claim_ids)), encoding="utf-8")
            prompt_text = packet["messages"][0]["content"] + "\n\n" + packet["messages"][1]["content"]
            result = subprocess.run(
                [
                    "codex",
                    "exec",
                    "-m",
                    args.model,
                    "--sandbox",
                    "read-only",
                    "--output-schema",
                    str(schema_path),
                    "--output-last-message",
                    str(output_path),
                ],
                input=prompt_text,
                text=True,
                capture_output=True,
                check=False,
                cwd=str(Path.cwd()),
            )
            (args.log_dir / f"{example_id}.stdout.log").write_text(result.stdout, encoding="utf-8")
            (args.log_dir / f"{example_id}.stderr.log").write_text(result.stderr, encoding="utf-8")
            if result.returncode != 0 or not output_path.exists():
                failures.append(
                    {
                        "example_id": example_id,
                        "error": f"codex_exec_failed returncode={result.returncode}",
                    }
                )
                continue
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            raw_path = args.raw_dir / f"{example_id}.json"
            raw_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            try:
                predictions = parse_baseline_claims(
                    payload,
                    prompt_version=packet["prompt_version"],
                    model_name=args.model,
                )
                predictions = normalize_prediction_claim_ids(predictions, expected_claim_ids)
                validate_prediction_claim_ids(predictions, expected_claim_ids)
            except Exception as exc:
                failures.append({"example_id": example_id, "error": f"{type(exc).__name__}: {exc}"})
                continue
            write_jsonl(args.parsed_dir / f"{example_id}.jsonl", [row.as_dict() for row in predictions])
            parsed_total += len(predictions)
            print(f"MATERIALIZED_GPT54 {example_id} claims={len(predictions)}")

    summary = {
        "n_examples": len(examples),
        "n_examples_succeeded": len(examples) - len(failures),
        "n_examples_failed": len(failures),
        "n_predictions": parsed_total,
        "model": args.model,
        "failures": failures,
    }
    ensure_parent(args.summary_json)
    write_json(args.summary_json, summary)
    print(f"Wrote GPT-5.4 summary to {args.summary_json}")


if __name__ == "__main__":
    main()
