from __future__ import annotations

import argparse
from pathlib import Path

from interp_experiment.activations.extractors import build_extractor
from interp_experiment.env import load_repo_env
from interp_experiment.generation.answers import build_deterministic_answer_prompt, clean_deterministic_answer
from interp_experiment.io import read_jsonl, write_jsonl
from interp_experiment.schemas import ExampleRow


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
    args = parser.parse_args()

    load_repo_env()
    extractor = build_extractor(args.model_name, layer_index=args.layer_index, device=args.device)
    examples = [ExampleRow.from_dict(row) for row in read_jsonl(args.input_jsonl)]
    updated: list[dict[str, object]] = []
    for index, example in enumerate(examples):
        if args.limit and index >= args.limit:
            updated.append(example.as_dict())
            continue
        raw_answer = extractor.generate_text(build_prompt(example), max_new_tokens=args.max_new_tokens)
        example.llama_answer_text = clean_deterministic_answer(raw_answer)
        updated.append(example.validate().as_dict())
        print(f"Generated answer for {example.example_id}")
    write_jsonl(args.output_jsonl, updated)


if __name__ == "__main__":
    main()
