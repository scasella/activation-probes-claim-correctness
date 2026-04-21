from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from interp_experiment.activations.entropy_targets import claim_presence_probability, claim_resampling_entropy
from interp_experiment.activations.extractors import build_extractor
from interp_experiment.activations.perturbations import paraphrase_prompt
from interp_experiment.data.claims import split_answer_into_claims
from interp_experiment.env import load_repo_env
from interp_experiment.generation.answers import (
    build_deterministic_answer_prompt,
    clean_deterministic_answer,
    ensure_non_empty_answer,
)
from interp_experiment.io import read_jsonl
from interp_experiment.schemas import ClaimRow, ExampleRow
from interp_experiment.utils import normalize_whitespace


def _emit_jsonl_section(label: str, rows: list[dict[str, object]]) -> None:
    print(f"===BEGIN_{label}===")
    for row in rows:
        print(json.dumps(row, ensure_ascii=True))
    print(f"===END_{label}===")


def _emit_json_section(label: str, payload: dict[str, object]) -> None:
    print(f"===BEGIN_{label}===")
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))
    print(f"===END_{label}===")


def _clean_paraphrase_text(text: str) -> str:
    text = text.replace("<|eot_id|>", " ")
    text = normalize_whitespace(text.strip().strip('"'))
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute claim-level entropy and stability targets in a remote sandbox.")
    parser.add_argument("--examples-jsonl", type=Path, required=True)
    parser.add_argument("--claims-jsonl", type=Path, required=True)
    parser.add_argument("--model-name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--layer-index", type=int, default=19)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--n-paraphrases", type=int, default=3)
    parser.add_argument("--max-answer-tokens", type=int, default=96)
    parser.add_argument("--max-paraphrase-tokens", type=int, default=64)
    args = parser.parse_args()

    load_repo_env()
    extractor = build_extractor(args.model_name, layer_index=args.layer_index, device=args.device)
    examples = {row.example_id: row for row in (ExampleRow.from_dict(item) for item in read_jsonl(args.examples_jsonl))}
    claims_by_example: dict[str, list[ClaimRow]] = defaultdict(list)
    for claim in (ClaimRow.from_dict(item) for item in read_jsonl(args.claims_jsonl)):
        claims_by_example[claim.example_id].append(claim)

    rows: list[dict[str, object]] = []
    for example_id, example in examples.items():
        claims = claims_by_example[example_id]
        prompt_text = build_deterministic_answer_prompt(example)
        sampled_claims: list[str] = []
        for _ in range(args.n_samples):
            sampled_answer = extractor.generate_text(
                prompt_text,
                max_new_tokens=args.max_answer_tokens,
                temperature=0.8,
                do_sample=True,
            )
            cleaned = ensure_non_empty_answer(example.task_family, clean_deterministic_answer(sampled_answer))
            sampled_claims.extend(split_answer_into_claims(cleaned))

        paraphrase_claims: list[str] = []
        for paraphrase_index in range(args.n_paraphrases):
            paraphrase_text = extractor.generate_text(
                paraphrase_prompt(example.question_text, paraphrase_index),
                max_new_tokens=args.max_paraphrase_tokens,
                temperature=0.0,
                do_sample=False,
            )
            paraphrased_question = _clean_paraphrase_text(paraphrase_text) or example.question_text
            paraphrased_example = ExampleRow(
                example_id=example.example_id,
                source_corpus=example.source_corpus,
                task_family=example.task_family,
                contract_id=example.contract_id,
                contract_group=example.contract_group,
                excerpt_text=example.excerpt_text,
                question_text=paraphrased_question,
                public_seed_answer=example.public_seed_answer,
                llama_answer_text=example.llama_answer_text,
                split=example.split,
                cross_dist_group=example.cross_dist_group,
            ).validate()
            answer = extractor.generate_text(
                build_deterministic_answer_prompt(paraphrased_example),
                max_new_tokens=args.max_answer_tokens,
                temperature=0.0,
                do_sample=False,
            )
            cleaned_answer = ensure_non_empty_answer(paraphrased_example.task_family, clean_deterministic_answer(answer))
            paraphrase_claims.extend(split_answer_into_claims(cleaned_answer))

        for claim in claims:
            stability_probability = claim_presence_probability(claim.claim_text, paraphrase_claims, threshold=0.55)
            rows.append(
                {
                    "claim_id": claim.claim_id,
                    "example_id": claim.example_id,
                    "correctness_target": claim_resampling_entropy(claim.claim_text, sampled_claims, threshold=0.55),
                    "stability_target": 1 if stability_probability >= 0.5 else 0,
                    "stability_probability": stability_probability,
                    "n_samples": args.n_samples,
                    "n_paraphrases": args.n_paraphrases,
                }
            )
        print(f"REMOTE_TARGETS_OK {example_id} claims={len(claims)}")

    summary = {
        "n_examples": len(examples),
        "n_rows": len(rows),
        "n_samples": args.n_samples,
        "n_paraphrases": args.n_paraphrases,
        "model_name": args.model_name,
    }
    _emit_json_section("SUMMARY_JSON", summary)
    _emit_jsonl_section("TARGETS_JSONL", rows)


if __name__ == "__main__":
    main()
