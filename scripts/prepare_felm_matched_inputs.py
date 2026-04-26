from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import requests
from huggingface_hub import hf_hub_download

from interp_experiment.io import read_jsonl, write_json, write_jsonl
from interp_experiment.schemas import ExampleRow


def _read_source_rows() -> dict[str, dict[str, Any]]:
    source_path = Path(hf_hub_download("hkust-nlp/felm", "all.jsonl", repo_type="dataset"))
    rows: dict[str, dict[str, Any]] = {}
    with source_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("domain") == "wk":
                rows[f"felm-wk-{int(row['index']):04d}"] = row
    return rows


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [value]
    return []


def _fetch_url(url: str, timeout: float) -> tuple[str | None, str | None]:
    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "interp-felm-matched-generation/0.1"},
        )
        response.raise_for_status()
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    text = response.text.strip()
    return text if text else None, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare FELM matched-generation seed examples and reference cache.")
    parser.add_argument("--base-examples-jsonl", type=Path, default=Path("data/felm/felm_wk_examples.jsonl"))
    parser.add_argument("--output-examples-jsonl", type=Path, default=Path("data/felm/felm_wk_matched_seed_examples.jsonl"))
    parser.add_argument("--reference-cache-jsonl", type=Path, default=Path("data/felm/felm_wk_reference_cache.jsonl"))
    parser.add_argument("--summary-json", type=Path, default=Path("artifacts/runs/felm_wk_matched_input_summary.json"))
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--fetch-missing", action="store_true")
    parser.add_argument("--fetch-timeout-sec", type=float, default=15.0)
    args = parser.parse_args()

    source_rows = _read_source_rows()
    base_examples = [ExampleRow.from_dict(row) for row in read_jsonl(args.base_examples_jsonl)]
    if args.max_examples:
        base_examples = base_examples[: args.max_examples]

    matched_examples: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for base in base_examples:
        source = source_rows.get(base.example_id)
        if source is None:
            failures.append({"example_id": base.example_id, "reason": "missing_source_row"})
            continue

        matched_id = base.example_id.replace("felm-", "felm-matched-", 1)
        refs = [str(item) for item in _as_list(source.get("ref")) if str(item).strip()]
        raw_contents = _as_list(source.get("ref_contents"))
        reference_entries: list[dict[str, Any]] = []
        for idx, url in enumerate(refs):
            dataset_content = str(raw_contents[idx]).strip() if idx < len(raw_contents) and raw_contents[idx] is not None else ""
            content = dataset_content
            source_kind = "dataset_ref_contents" if content else "missing"
            fetch_error = None
            if not content and args.fetch_missing:
                fetched, fetch_error = _fetch_url(url, args.fetch_timeout_sec)
                if fetched:
                    content = fetched
                    source_kind = "live_fetch"
            reference_entries.append(
                {
                    "url": url,
                    "content": content,
                    "content_chars": len(content),
                    "source": source_kind,
                    "fetch_error": fetch_error,
                }
            )

        if refs and not any(entry["content"] for entry in reference_entries):
            failures.append({"example_id": matched_id, "reason": "no_reference_content"})

        matched = ExampleRow(
            example_id=matched_id,
            source_corpus="felm_matched_generation",
            task_family="generative_qa",
            contract_id=base.contract_id,
            contract_group=base.contract_group,
            excerpt_text="FELM matched-generation example; correctness is judged against cached FELM reference evidence.",
            question_text=base.question_text,
            public_seed_answer=base.public_seed_answer,
            llama_answer_text="PENDING_GENERATION",
            split=base.split,
            cross_dist_group="felm_wk_matched_generation",
        ).validate()
        matched_examples.append(matched.as_dict())
        reference_rows.append(
            {
                "example_id": matched_id,
                "base_example_id": base.example_id,
                "source_dataset": "hkust-nlp/felm",
                "source_index": source.get("index"),
                "domain": source.get("domain"),
                "references": reference_entries,
            }
        )

    write_jsonl(args.output_examples_jsonl, matched_examples)
    write_jsonl(args.reference_cache_jsonl, reference_rows)
    summary = {
        "n_base_examples": len(base_examples),
        "n_matched_examples": len(matched_examples),
        "split_counts": dict(Counter(row["split"] for row in matched_examples)),
        "n_reference_rows": len(reference_rows),
        "n_examples_with_reference_content": sum(
            1 for row in reference_rows if any(ref["content"] for ref in row["references"])
        ),
        "n_failures": len(failures),
        "failures": failures,
        "reference_source_note": (
            "FELM all.jsonl already includes ref_contents. This script caches that evidence and only live-fetches "
            "missing reference content when --fetch-missing is set."
        ),
    }
    write_json(args.summary_json, summary)
    print(f"Wrote matched seed examples to {args.output_examples_jsonl}")
    print(f"Wrote reference cache to {args.reference_cache_jsonl}")
    print(f"Wrote summary to {args.summary_json}")


if __name__ == "__main__":
    main()
