from __future__ import annotations

import argparse
import json
from pathlib import Path

from interp_experiment.activations.extractors import build_extractor_with_info
from interp_experiment.data.claims import build_canonical_claims
from interp_experiment.env import load_repo_env
from interp_experiment.generation.answers import build_deterministic_answer_prompt, clean_deterministic_answer
from interp_experiment.io import read_jsonl
from interp_experiment.schemas import ExampleRow


def _emit_jsonl_section(label: str, rows: list[dict[str, object]]) -> None:
    print(f"===BEGIN_{label}===")
    for row in rows:
        print(json.dumps(row, ensure_ascii=True))
    print(f"===END_{label}===")


def _emit_json_section(label: str, payload: dict[str, object]) -> None:
    print(f"===BEGIN_{label}===")
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))
    print(f"===END_{label}===")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic pilot answer generation inside a remote sandbox.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--model-name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--layer-index", type=int, default=19)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--annotation-version", default="v1")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()

    load_repo_env()
    build_info = build_extractor_with_info(args.model_name, layer_index=args.layer_index, device=args.device)
    extractor = build_info.extractor
    examples = [ExampleRow.from_dict(row) for row in read_jsonl(args.input_jsonl)]

    updated_examples: list[dict[str, object]] = []
    all_claims: list[dict[str, object]] = []
    answer_lengths: list[int] = []
    for example in examples:
        raw_answer = extractor.generate_text(
            build_deterministic_answer_prompt(example),
            max_new_tokens=args.max_new_tokens,
        )
        example.llama_answer_text = clean_deterministic_answer(raw_answer)
        example.validate()
        updated_examples.append(example.as_dict())
        claims = build_canonical_claims(example, annotation_version=args.annotation_version)
        all_claims.extend(claim.as_dict() for claim in claims)
        answer_lengths.append(len(example.llama_answer_text))
        print(f"REMOTE_GENERATED {example.example_id} claims={len(claims)} chars={len(example.llama_answer_text)}")

    summary = {
        "n_examples": len(updated_examples),
        "n_claims": len(all_claims),
        "mean_answer_chars": (sum(answer_lengths) / len(answer_lengths)) if answer_lengths else 0.0,
        "extractor_name": build_info.extractor.__class__.__name__,
        "transformer_lens_primary_succeeded": build_info.primary_succeeded,
        "transformer_lens_primary_error": build_info.primary_error,
        "model_name": args.model_name,
        "max_new_tokens": args.max_new_tokens,
        "first_example_id": updated_examples[0]["example_id"] if updated_examples else None,
    }
    _emit_json_section("SUMMARY_JSON", summary)
    _emit_jsonl_section("EXAMPLES_JSONL", updated_examples)
    _emit_jsonl_section("CLAIMS_JSONL", all_claims)


if __name__ == "__main__":
    main()
