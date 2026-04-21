from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from interp_experiment.activations.extractors import build_extractor
from interp_experiment.baselines.llama_self_report import parse_baseline_claims
from interp_experiment.baselines.prompts import render_self_report_prompt
from interp_experiment.baselines.utils import (
    extract_json_object,
    normalize_prediction_claim_ids,
    validate_prediction_claim_ids,
)
from interp_experiment.env import load_repo_env
from interp_experiment.io import read_jsonl
from interp_experiment.schemas import ClaimRow, ExampleRow


def _emit_jsonl_section(label: str, rows: list[dict[str, object]]) -> None:
    print(f"===BEGIN_{label}===")
    for row in rows:
        print(json.dumps(row, ensure_ascii=True))
    print(f"===END_{label}===")


def _emit_json_section(label: str, payload: dict[str, object]) -> None:
    print(f"===BEGIN_{label}===")
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))
    print(f"===END_{label}===")


def _build_prompt_text(example: ExampleRow, claims: list[ClaimRow]) -> tuple[str, str, str]:
    rendered = render_self_report_prompt(example, claims)
    prompt_text = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{rendered['system']}<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"{rendered['user']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )
    return prompt_text, rendered["prompt_version"], rendered["user"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Llama self-report baseline generation inside a remote sandbox.")
    parser.add_argument("--examples-jsonl", type=Path, required=True)
    parser.add_argument("--claims-jsonl", type=Path, required=True)
    parser.add_argument("--model-name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--layer-index", type=int, default=19)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    args = parser.parse_args()

    load_repo_env()
    extractor = build_extractor(args.model_name, layer_index=args.layer_index, device=args.device)
    examples = {row.example_id: row for row in (ExampleRow.from_dict(item) for item in read_jsonl(args.examples_jsonl))}
    claims_by_example: dict[str, list[ClaimRow]] = defaultdict(list)
    for claim in (ClaimRow.from_dict(item) for item in read_jsonl(args.claims_jsonl)):
        claims_by_example[claim.example_id].append(claim)

    raw_rows: list[dict[str, object]] = []
    parsed_by_example: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for example_id, example in examples.items():
        claims = claims_by_example[example_id]
        prompt_text, prompt_version, user_prompt = _build_prompt_text(example, claims)
        raw_text = extractor.generate_text(prompt_text, max_new_tokens=args.max_new_tokens)
        raw_row = {
            "example_id": example_id,
            "model_name": args.model_name,
            "prompt_version": prompt_version,
            "user_prompt": user_prompt,
            "raw_text": raw_text,
        }
        raw_rows.append(raw_row)
        try:
            payload = extract_json_object(raw_text)
            predictions = parse_baseline_claims(
                payload,
                prompt_version=prompt_version,
                model_name="llama-3.1-8b-instruct-self-report",
            )
            predictions = normalize_prediction_claim_ids(predictions, [claim.claim_id for claim in claims])
            validate_prediction_claim_ids(predictions, [claim.claim_id for claim in claims])
        except Exception as exc:
            failures.append({"example_id": example_id, "error": f"{type(exc).__name__}: {exc}"})
            continue
        parsed_by_example.append(
            {
                "example_id": example_id,
                "predictions": [prediction.as_dict() for prediction in predictions],
            }
        )
        print(f"REMOTE_SELF_REPORT_OK {example_id} claims={len(predictions)}")

    summary = {
        "n_examples": len(examples),
        "n_examples_parsed": len(examples) - len(failures),
        "n_examples_failed": len(failures),
        "n_predictions": sum(len(row["predictions"]) for row in parsed_by_example),
        "failures": failures,
        "model_name": args.model_name,
    }
    _emit_json_section("SUMMARY_JSON", summary)
    _emit_jsonl_section("RAW_JSONL", raw_rows)
    _emit_jsonl_section("PARSED_BY_EXAMPLE_JSONL", parsed_by_example)


if __name__ == "__main__":
    main()
