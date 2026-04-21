from __future__ import annotations

import argparse
from pathlib import Path

from interp_experiment.activations.extractors import build_extractor
from interp_experiment.env import load_repo_env
from interp_experiment.io import read_jsonl, write_jsonl
from interp_experiment.schemas import ExampleRow


def build_prompt(example: ExampleRow) -> str:
    return (
        "Read the contract excerpt and answer the legal question.\n\n"
        f"Contract excerpt:\n{example.excerpt_text}\n\n"
        f"Question:\n{example.question_text}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic Llama answers for source-pool examples.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--model-name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--layer-index", type=int, default=19)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    load_repo_env()
    extractor = build_extractor(args.model_name, layer_index=args.layer_index, device=args.device)
    examples = [ExampleRow.from_dict(row) for row in read_jsonl(args.input_jsonl)]
    updated: list[dict[str, object]] = []
    for index, example in enumerate(examples):
        if args.limit and index >= args.limit:
            updated.append(example.as_dict())
            continue
        generated = extractor.generate_with_activations(build_prompt(example))
        example.llama_answer_text = generated.answer_text
        updated.append(example.validate().as_dict())
        print(f"Generated answer for {example.example_id} using {generated.extractor_name}")
    write_jsonl(args.output_jsonl, updated)


if __name__ == "__main__":
    main()
