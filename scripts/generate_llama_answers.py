from __future__ import annotations

import argparse
from pathlib import Path

from interp_experiment.activations.extractors import build_extractor
from interp_experiment.env import load_repo_env
from interp_experiment.generation.answers import (
    build_deterministic_answer_prompt,
    clean_deterministic_answer,
    ensure_non_empty_answer,
)
from interp_experiment.io import read_jsonl, write_jsonl
from interp_experiment.schemas import AnswerRunRow, ExampleRow


def build_prompt(example: ExampleRow) -> str:
    return build_deterministic_answer_prompt(example)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic Llama answers for source-pool examples.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--model-name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--layer-index", type=int, default=19)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--output-answer-runs-jsonl", type=Path, default=None)
    args = parser.parse_args()

    load_repo_env()
    extractor = build_extractor(args.model_name, layer_index=args.layer_index, device=args.device)
    examples = [ExampleRow.from_dict(row) for row in read_jsonl(args.input_jsonl)]
    updated: list[dict[str, object]] = []
    answer_runs: list[dict[str, object]] = []
    for index, example in enumerate(examples):
        if args.limit and index >= args.limit:
            updated.append(example.as_dict())
            continue
        prompt_text = build_prompt(example)
        raw_run = extractor.generate_answer_run(
            example_id=example.example_id,
            source_corpus=example.source_corpus,
            task_family=example.task_family,
            prompt_text=prompt_text,
            max_new_tokens=args.max_new_tokens,
        )
        cleaned_answer = ensure_non_empty_answer(
            example.task_family,
            clean_deterministic_answer(raw_run.answer_text),
        )
        answer_run = extractor.retokenize_answer_run(
            example_id=example.example_id,
            source_corpus=example.source_corpus,
            task_family=example.task_family,
            prompt_text=prompt_text,
            answer_text=cleaned_answer,
        )
        example.llama_answer_text = answer_run.answer_text
        updated.append(example.validate().as_dict())
        answer_runs.append(answer_run.as_dict())
        print(f"Generated answer for {example.example_id}")
    write_jsonl(args.output_jsonl, updated)
    if args.output_answer_runs_jsonl is not None:
        write_jsonl(args.output_answer_runs_jsonl, answer_runs)


if __name__ == "__main__":
    main()
