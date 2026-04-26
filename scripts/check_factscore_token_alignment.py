from __future__ import annotations

import argparse
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from interp_experiment.io import read_jsonl, write_json, write_jsonl
from interp_experiment.env import load_repo_env


def _llama_chat_prompt(system: str, user: str) -> str:
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )


def _answer_prompt(question_text: str) -> str:
    return _llama_chat_prompt(
        "You are a helpful assistant. Answer the user's question directly.",
        question_text,
    )


def _token_span(char_start: int, char_end: int, offsets: list[tuple[int, int]]) -> tuple[int, int] | None:
    token_indices = [
        idx
        for idx, (start, end) in enumerate(offsets)
        if end > char_start and start < char_end
    ]
    if not token_indices:
        return None
    return min(token_indices), max(token_indices)


def _quantiles(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "p25": None, "median": None, "p75": None, "max": None, "mean": None}
    sorted_values = sorted(values)
    return {
        "min": sorted_values[0],
        "p25": float(statistics.quantiles(sorted_values, n=4, method="inclusive")[0]),
        "median": float(statistics.median(sorted_values)),
        "p75": float(statistics.quantiles(sorted_values, n=4, method="inclusive")[2]),
        "max": sorted_values[-1],
        "mean": float(statistics.mean(sorted_values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check FActScore claim char spans against Llama tokenizer offsets.")
    parser.add_argument("--examples-jsonl", type=Path, default=Path("data/factscore/factscore_chatgpt_examples.jsonl"))
    parser.add_argument("--claims-jsonl", type=Path, default=Path("data/factscore/factscore_chatgpt_claims.jsonl"))
    parser.add_argument("--model-name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--summary-json", type=Path, default=Path("artifacts/runs/factscore_chatgpt_token_alignment_summary.json"))
    parser.add_argument("--inspection-jsonl", type=Path, default=Path("data/factscore/factscore_chatgpt_token_alignment_inspection.jsonl"))
    parser.add_argument("--inspection-size", type=int, default=20)
    parser.add_argument("--max-examples", type=int, default=0)
    args = parser.parse_args()
    load_repo_env()

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("transformers is required. Run with `uv run --extra inference python scripts/check_factscore_token_alignment.py`.") from exc

    examples = list(read_jsonl(args.examples_jsonl))
    if args.max_examples:
        examples = examples[: args.max_examples]
    example_ids = {row["example_id"] for row in examples}
    claims_by_example: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for claim in read_jsonl(args.claims_jsonl):
        if claim["example_id"] in example_ids:
            claims_by_example[claim["example_id"]].append(claim)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    token_lengths: list[int] = []
    exact_decoded_answer_matches = 0
    answer_mismatch_examples: list[dict[str, Any]] = []
    alignment_failures: list[dict[str, Any]] = []
    inspection_rows: list[dict[str, Any]] = []
    span_sources: Counter[str] = Counter()

    for example in examples:
        prompt = _answer_prompt(example["question_text"])
        answer_text = example["llama_answer_text"]
        prompt_batch = tokenizer(prompt, return_tensors="pt")
        full_batch = tokenizer(prompt + answer_text, return_tensors="pt")
        prompt_len = int(prompt_batch["input_ids"].shape[1])
        answer_ids = full_batch["input_ids"][0][prompt_len:]
        if int(answer_ids.numel()) == 0:
            answer_ids = tokenizer(answer_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        decoded_answer = tokenizer.decode(answer_ids, skip_special_tokens=True)
        if decoded_answer == answer_text:
            exact_decoded_answer_matches += 1
        elif len(answer_mismatch_examples) < 10:
            answer_mismatch_examples.append(
                {
                    "example_id": example["example_id"],
                    "expected_preview": answer_text[:240],
                    "decoded_preview": decoded_answer[:240],
                    "expected_chars": len(answer_text),
                    "decoded_chars": len(decoded_answer),
                }
            )
        offsets = [
            (int(start), int(end))
            for start, end in tokenizer(decoded_answer, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
        ]
        for claim in claims_by_example[example["example_id"]]:
            span_sources[str(claim.get("span_source"))] += 1
            span = _token_span(int(claim["char_start"]), int(claim["char_end"]), offsets)
            if span is None:
                alignment_failures.append({"claim_id": claim["claim_id"], "example_id": claim["example_id"], "reason": "no_token_overlap"})
                continue
            token_start, token_end = span
            length = token_end - token_start + 1
            token_lengths.append(length)
            if len(inspection_rows) < args.inspection_size:
                char_start = int(claim["char_start"])
                char_end = int(claim["char_end"])
                covered_start = offsets[token_start][0]
                covered_end = offsets[token_end][1]
                inspection_rows.append(
                    {
                        "claim_id": claim["claim_id"],
                        "example_id": claim["example_id"],
                        "claim_text": claim["claim_text"],
                        "source_sentence_text": claim.get("source_sentence_text"),
                        "span_source": claim.get("span_source"),
                        "char_start": char_start,
                        "char_end": char_end,
                        "token_start": token_start,
                        "token_end": token_end,
                        "token_span_length": length,
                        "char_span_text": decoded_answer[char_start:char_end],
                        "token_covered_text": decoded_answer[covered_start:covered_end],
                    }
                )

    short_spans = sum(1 for value in token_lengths if value <= 2)
    summary = {
        "model_name": args.model_name,
        "n_examples_checked": len(examples),
        "n_claims_checked": sum(len(claims_by_example[example["example_id"]]) for example in examples),
        "n_claims_aligned": len(token_lengths),
        "n_alignment_failures": len(alignment_failures),
        "alignment_failure_rate": (
            len(alignment_failures) / sum(len(claims_by_example[example["example_id"]]) for example in examples)
            if examples
            else None
        ),
        "token_span_lengths": _quantiles(token_lengths),
        "n_token_spans_length_le_2": short_spans,
        "token_spans_length_le_2_rate": short_spans / len(token_lengths) if token_lengths else None,
        "span_source_counts": dict(span_sources),
        "n_exact_decoded_answer_matches": exact_decoded_answer_matches,
        "n_decoded_answer_mismatches": len(examples) - exact_decoded_answer_matches,
        "answer_mismatch_examples": answer_mismatch_examples,
        "alignment_failures": alignment_failures[:100],
        "note": (
            "FActScore atomic fact text rarely appears verbatim in the generation. The adapter maps atomic labels "
            "to parent-sentence character spans; this script checks those spans against Llama tokenizer offsets."
        ),
    }
    write_json(summary_path := args.summary_json, summary)
    write_jsonl(args.inspection_jsonl, inspection_rows)
    print(f"Wrote token alignment summary to {summary_path}")
    print(f"Wrote token alignment inspection rows to {args.inspection_jsonl}")
    print(
        "Alignment: "
        f"{summary['n_claims_aligned']}/{summary['n_claims_checked']} claims, "
        f"median token span {summary['token_span_lengths']['median']}, "
        f"<=2 token span rate {summary['token_spans_length_le_2_rate']:.3f}"
    )


if __name__ == "__main__":
    main()
