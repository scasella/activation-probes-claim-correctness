from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

from interp_experiment.io import read_jsonl, write_json
from interp_experiment.schemas import ClaimRow, ExampleRow
from interp_experiment.utils import tokenize_for_matching

SPECIAL_TOKEN_RE = re.compile(r"<\|[^>]+\|>")
NUMERIC_FRAGMENT_RE = re.compile(r"^(?:\d+(?:\.\d+)*\.?|[A-Z]\.)$")
SECTION_HEADER_RE = re.compile(r"^(?:Section|Article|Exhibit)\b", re.IGNORECASE)
SIGNER_FRAGMENT_RE = re.compile(r"\b(?:Name:|Title:|By:|/s/)\b")


def _jaccard(left: str, right: str) -> float:
    left_tokens = tokenize_for_matching(left)
    right_tokens = tokenize_for_matching(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _claim_junk_reasons(claim_text: str) -> list[str]:
    reasons: list[str] = []
    normalized = claim_text.strip()
    if not normalized:
        reasons.append("empty")
        return reasons
    if SPECIAL_TOKEN_RE.search(normalized):
        reasons.append("special_token")
    if NUMERIC_FRAGMENT_RE.match(normalized):
        reasons.append("numeric_fragment")
    if SECTION_HEADER_RE.match(normalized):
        reasons.append("section_header")
    if SIGNER_FRAGMENT_RE.search(normalized):
        reasons.append("signer_fragment")
    if len(normalized) <= 4:
        reasons.append("very_short")
    if normalized.endswith(":"):
        reasons.append("header_like")
    return reasons


def _answer_suspect_reasons(example: ExampleRow) -> list[str]:
    reasons: list[str] = []
    answer = example.llama_answer_text.strip()
    if SPECIAL_TOKEN_RE.search(answer):
        reasons.append("special_token")
    if not answer:
        reasons.append("empty_answer")
        return reasons
    if example.task_family == "field_extraction":
        overlap = _jaccard(answer, example.public_seed_answer)
        if overlap < 0.08:
            reasons.append("low_seed_overlap")
        if len(answer) > 260:
            reasons.append("long_extractive_answer")
        if SECTION_HEADER_RE.match(answer) or NUMERIC_FRAGMENT_RE.match(answer):
            reasons.append("header_like_answer")
    return reasons


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit a machine-readable readiness summary for the pilot artifacts.")
    parser.add_argument("--examples-jsonl", type=Path, required=True)
    parser.add_argument("--claims-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    examples = [ExampleRow.from_dict(row) for row in read_jsonl(args.examples_jsonl)]
    claims = [ClaimRow.from_dict(row) for row in read_jsonl(args.claims_jsonl)]

    contract_counts_by_source: dict[str, Counter[str]] = defaultdict(Counter)
    for example in examples:
        contract_counts_by_source[example.source_corpus][example.contract_id] += 1

    claims_by_example: dict[str, list[ClaimRow]] = defaultdict(list)
    junk_claim_rows: list[dict[str, object]] = []
    junk_examples: set[str] = set()
    for claim in claims:
        claims_by_example[claim.example_id].append(claim)
        reasons = _claim_junk_reasons(claim.claim_text)
        if reasons:
            junk_examples.add(claim.example_id)
            junk_claim_rows.append(
                {
                    "example_id": claim.example_id,
                    "claim_id": claim.claim_id,
                    "claim_text": claim.claim_text,
                    "reasons": reasons,
                }
            )

    cuad_review_queue: list[dict[str, object]] = []
    for example in examples:
        if example.source_corpus != "cuad":
            continue
        answer_reasons = _answer_suspect_reasons(example)
        claim_reasons = [row for row in junk_claim_rows if row["example_id"] == example.example_id]
        cuad_review_queue.append(
            {
                "example_id": example.example_id,
                "question_text": example.question_text,
                "public_seed_answer": example.public_seed_answer,
                "llama_answer_text": example.llama_answer_text,
                "answer_reasons": answer_reasons,
                "junk_claim_count": len(claim_reasons),
                "claim_count": len(claims_by_example.get(example.example_id, [])),
                "seed_overlap": round(_jaccard(example.llama_answer_text, example.public_seed_answer), 4),
            }
        )

    summary = {
        "n_examples": len(examples),
        "n_claims": len(claims),
        "split_integrity": {
            "by_split": dict(Counter(example.split for example in examples)),
            "n_test_rows": sum(example.split == "test" for example in examples),
            "ok": all(example.split != "test" for example in examples),
        },
        "contract_diversity": {
            source: {
                "unique_contract_count": len(counter),
                "max_rows_per_contract": max(counter.values(), default=0),
                "top_repeated_contracts": counter.most_common(5),
            }
            for source, counter in contract_counts_by_source.items()
        },
        "artifact_hygiene": {
            "special_token_answers": sum(bool(SPECIAL_TOKEN_RE.search(example.llama_answer_text)) for example in examples),
            "junk_claim_count": len(junk_claim_rows),
            "junk_example_count": len(junk_examples),
        },
        "cuad_answer_relevance": {
            "n_examples": len(cuad_review_queue),
            "n_examples_with_suspect_answers": sum(bool(row["answer_reasons"]) for row in cuad_review_queue),
            "n_examples_with_junk_claims": sum(row["junk_claim_count"] > 0 for row in cuad_review_queue),
            "review_queue": cuad_review_queue,
        },
        "ready_for_annotation_heuristic": (
            all(example.split != "test" for example in examples)
            and sum(bool(SPECIAL_TOKEN_RE.search(example.llama_answer_text)) for example in examples) == 0
            and sum(row["junk_claim_count"] > 0 for row in cuad_review_queue) <= 3
            and sum(bool(row["answer_reasons"]) for row in cuad_review_queue) <= 3
        ),
    }
    write_json(args.output_json, summary)
    print(f"Wrote pilot readiness summary to {args.output_json}")


if __name__ == "__main__":
    main()
