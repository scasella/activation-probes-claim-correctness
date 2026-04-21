from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from interp_experiment.codex_app_server_client import CodexAppServerClient
from interp_experiment.io import read_jsonl, write_json, write_jsonl
from interp_experiment.schemas import ClaimRow, ExampleRow


def _judge_schema(expected_claim_ids: list[str]) -> dict[str, object]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "minItems": len(expected_claim_ids),
                "maxItems": len(expected_claim_ids),
                "items": {
                    "type": "object",
                    "properties": {
                        "claim_id": {"type": "string", "enum": expected_claim_ids},
                        "correctness_label": {"type": "string", "enum": ["true", "false", "partially_true"]},
                        "load_bearing_label": {"type": "string", "enum": ["yes", "no"]},
                        "flip_evidence_text": {"type": "string"},
                        "notes": {"type": "string"},
                    },
                    "required": [
                        "claim_id",
                        "correctness_label",
                        "load_bearing_label",
                        "flip_evidence_text",
                        "notes",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["claims"],
        "additionalProperties": False,
    }


def _judge_prompt(example: ExampleRow, claims: list[ClaimRow]) -> str:
    claim_block = "\n".join(f"- {claim.claim_id}: {claim.claim_text}" for claim in claims)
    return (
        "You are a proxy legal annotator for a research pilot. "
        "Use only the contract excerpt, the question, the model answer, and the fixed claim list.\n\n"
        "For each claim return:\n"
        "- correctness_label: true, false, or partially_true\n"
        "- load_bearing_label: yes or no\n"
        "- flip_evidence_text: required if load_bearing_label=yes, else empty string\n"
        "- notes: optional short note\n\n"
        f"Contract excerpt:\n{example.excerpt_text}\n\n"
        f"Question:\n{example.question_text}\n\n"
        f"Llama answer:\n{example.llama_answer_text}\n\n"
        f"Canonical claims:\n{claim_block}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize judge-LLM proxy annotations for the MAUD pilot.")
    parser.add_argument("--examples-jsonl", type=Path, default=Path("data/annotations/maud_pilot_examples.jsonl"))
    parser.add_argument("--claims-jsonl", type=Path, default=Path("data/annotations/maud_pilot_claims.jsonl"))
    parser.add_argument("--output-jsonl", type=Path, default=Path("data/annotations/maud_pilot_judge_annotations.jsonl"))
    parser.add_argument("--summary-json", type=Path, default=Path("artifacts/runs/maud_judge_annotation_summary.json"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/annotations/judge_llm_raw/maud_pilot"))
    parser.add_argument("--log-dir", type=Path, default=Path("artifacts/runs/maud_judge_logs"))
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--annotator-id", default="judge_gpt54")
    parser.add_argument("--annotation-version", default="proxy_v1")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-examples", type=int, default=0)
    args = parser.parse_args()

    examples = {row.example_id: row for row in (ExampleRow.from_dict(item) for item in read_jsonl(args.examples_jsonl))}
    claims_by_example: dict[str, list[ClaimRow]] = defaultdict(list)
    for claim in (ClaimRow.from_dict(item) for item in read_jsonl(args.claims_jsonl)):
        claims_by_example[claim.example_id].append(claim)

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    annotation_rows: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    processed = 0
    with CodexAppServerClient(cwd=Path.cwd(), model=args.model) as client:
        for example_id, example in examples.items():
            if args.max_examples and processed >= args.max_examples:
                break
            claims = claims_by_example[example_id]
            raw_path = args.raw_dir / f"{example_id}.json"
            if args.skip_existing and raw_path.exists():
                payload = json.loads(raw_path.read_text(encoding="utf-8"))
            else:
                try:
                    raw_text = client.run_prompt(
                        _judge_prompt(example, claims),
                        output_schema=_judge_schema([claim.claim_id for claim in claims]),
                    )
                    payload = json.loads(raw_text)
                except Exception as exc:
                    failures.append({"example_id": example_id, "error": f"{type(exc).__name__}: {exc}"})
                    continue
                raw_path.write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            try:
                items = payload["claims"]
            except Exception as exc:
                failures.append({"example_id": example_id, "error": f"{type(exc).__name__}: {exc}"})
                continue
            for item in items:
                annotation_rows.append(
                    {
                        "annotator_id": args.annotator_id,
                        "annotation_version": args.annotation_version,
                        "example_id": example.example_id,
                        "source_corpus": example.source_corpus,
                        "task_family": example.task_family,
                        "contract_id": example.contract_id,
                        "question_text": example.question_text,
                        "excerpt_text": example.excerpt_text,
                        "llama_answer_text": example.llama_answer_text,
                        "claim_id": item["claim_id"],
                        "claim_text": next(claim.claim_text for claim in claims if claim.claim_id == item["claim_id"]),
                        "correctness_label": item["correctness_label"],
                        "load_bearing_label": item["load_bearing_label"],
                        "flip_evidence_text": item["flip_evidence_text"],
                        "notes": item["notes"],
                    }
                )
            processed += 1
            print(f"MATERIALIZED_JUDGE {example_id} claims={len(items)}")
        if client.stderr_text:
            (args.log_dir / "app_server.stderr.log").write_text(client.stderr_text, encoding="utf-8")

    write_jsonl(args.output_jsonl, annotation_rows)
    summary = {
        "n_examples": len(examples),
        "n_examples_succeeded": len(examples) - len(failures),
        "n_examples_failed": len(failures),
        "n_rows": len(annotation_rows),
        "n_examples_processed_this_run": processed,
        "annotator_id": args.annotator_id,
        "annotation_version": args.annotation_version,
        "model": args.model,
        "proxy_only": True,
        "failures": failures,
    }
    write_json(args.summary_json, summary)
    print(f"Wrote judge annotations to {args.output_jsonl}")
    print(f"Wrote judge summary to {args.summary_json}")


if __name__ == "__main__":
    main()
