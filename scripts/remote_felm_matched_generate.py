from __future__ import annotations

import argparse
import json
from pathlib import Path

from interp_experiment.env import load_repo_env
from interp_experiment.io import read_jsonl
from interp_experiment.schemas import ExampleRow
from interp_experiment.utils import normalize_whitespace


def _emit_json_section(label: str, payload: dict[str, object]) -> None:
    print(f"===BEGIN_{label}===")
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))
    print(f"===END_{label}===")


def _emit_jsonl_section(label: str, rows: list[dict[str, object]]) -> None:
    print(f"===BEGIN_{label}===")
    for row in rows:
        print(json.dumps(row, ensure_ascii=True))
    print(f"===END_{label}===")


def _llama_chat_prompt(system: str, user: str) -> str:
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )


def _answer_prompt(example: ExampleRow) -> str:
    return _llama_chat_prompt(
        "You are a helpful assistant. Answer the user's question directly and concisely. "
        "Use 1 to 4 sentences. Do not use bullets, headings, or numbered lists.",
        example.question_text,
    )


def _clean_answer(text: str) -> str:
    text = text.replace("<|eot_id|>", " ").replace("<|end_of_text|>", " ")
    text = normalize_whitespace(text)
    return text if text else "I do not know."


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate matched Llama answers for FELM prompts.")
    parser.add_argument("--examples-jsonl", type=Path, required=True)
    parser.add_argument("--model-name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--layer-index", type=int, default=19)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", default="huggingface", choices=["auto", "huggingface"])
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=0, help="Exclusive example index; 0 means no explicit end.")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    args = parser.parse_args()

    load_repo_env()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    model.to(args.device)
    examples = [ExampleRow.from_dict(row) for row in read_jsonl(args.examples_jsonl)]
    if args.start_index or args.end_index:
        end_index = args.end_index if args.end_index else None
        examples = examples[args.start_index : end_index]
    if args.max_examples:
        examples = examples[: args.max_examples]

    generated_examples: list[dict[str, object]] = []
    answer_runs: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for example in examples:
        try:
            prompt_text = _answer_prompt(example)
            batch = tokenizer(prompt_text, return_tensors="pt")
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(args.device)
            with torch.no_grad():
                sequences = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )[0]
            answer_ids = sequences[int(input_ids.shape[1]) :]
            answer_text = _clean_answer(tokenizer.decode(answer_ids, skip_special_tokens=True))
            retokenized = tokenizer(answer_text, return_offsets_mapping=True, add_special_tokens=False)
            token_ids = [int(item) for item in retokenized["input_ids"]]
            offsets = [(int(start), int(end)) for start, end in retokenized["offset_mapping"]]
            answer_runs.append(
                {
                    "example_id": example.example_id,
                    "source_corpus": example.source_corpus,
                    "task_family": example.task_family,
                    "prompt_text": prompt_text,
                    "answer_text": answer_text,
                    "model_name": args.model_name,
                    "extractor_name": "huggingface_direct_generation",
                    "token_ids": token_ids,
                    "token_offsets": offsets,
                }
            )
            example.llama_answer_text = answer_text
            generated_examples.append(example.validate().as_dict())
            print(f"REMOTE_FELM_MATCHED_GENERATED {example.example_id} chars={len(answer_text)}")
        except Exception as exc:
            failures.append({"example_id": example.example_id, "error": f"{type(exc).__name__}: {exc}"})

    summary = {
        "n_examples_requested": len(examples),
        "n_examples_succeeded": len(generated_examples),
        "n_examples_failed": len(failures),
        "failures": failures,
        "extractor_name": "huggingface_direct_generation",
        "transformer_lens_primary_attempted": False,
        "transformer_lens_primary_succeeded": False,
        "transformer_lens_primary_error": None,
        "model_name": args.model_name,
        "max_new_tokens": args.max_new_tokens,
        "temperature": 0.0,
        "do_sample": False,
    }
    _emit_json_section("SUMMARY_JSON", summary)
    _emit_jsonl_section("EXAMPLES_JSONL", generated_examples)
    _emit_jsonl_section("ANSWER_RUNS_JSONL", answer_runs)


if __name__ == "__main__":
    main()
