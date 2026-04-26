from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split

from interp_experiment.io import write_json, write_jsonl
from interp_experiment.schemas import ExampleRow


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _find_segment_spans(response: str, segments: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    spans: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    cursor = 0
    for index, segment in enumerate(segments):
        start = response.find(segment, cursor)
        if start < 0:
            start = response.find(segment)
        if start < 0:
            compact_segment = re.sub(r"\s+", " ", segment).strip()
            compact_response = re.sub(r"\s+", " ", response)
            compact_start = compact_response.find(compact_segment)
            if compact_start >= 0:
                failures.append(
                    {
                        "segment_index": index,
                        "reason": "whitespace_normalized_match_only",
                        "segment_preview": segment[:160],
                    }
                )
            else:
                failures.append(
                    {
                        "segment_index": index,
                        "reason": "no_substring_match",
                        "segment_preview": segment[:160],
                    }
                )
            continue
        end = start + len(segment)
        cursor = end
        spans.append({"segment_index": index, "char_start": start, "char_end": end})
    return spans, failures


def _split_examples(rows: list[dict[str, Any]], seed: int) -> dict[str, str]:
    ids = [row["example_id"] for row in rows]
    strata = []
    for row in rows:
        labels = row["labels"]
        if all(labels):
            strata.append("all_true")
        elif not any(labels):
            strata.append("all_false")
        else:
            strata.append("mixed")
    counts = Counter(strata)
    stratify = strata if min(counts.values()) >= 2 else None
    train_ids, temp_ids, train_strata, temp_strata = train_test_split(
        ids,
        strata,
        test_size=0.30,
        random_state=seed,
        stratify=stratify,
    )
    temp_counts = Counter(temp_strata)
    temp_stratify = temp_strata if min(temp_counts.values()) >= 2 else None
    validation_ids, test_ids = train_test_split(
        temp_ids,
        test_size=0.50,
        random_state=seed + 1,
        stratify=temp_stratify,
    )
    split = {example_id: "train" for example_id in train_ids}
    split.update({example_id: "validation" for example_id in validation_ids})
    split.update({example_id: "test" for example_id in test_ids})
    return split


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert FELM into the repo's canonical example/claim/label files.")
    parser.add_argument("--domain", default="wk", help="FELM domain to use; default is world knowledge (`wk`).")
    parser.add_argument("--include-domain", action="append", default=[], help="Additional FELM domain to include.")
    parser.add_argument("--output-prefix", default="felm_wk")
    parser.add_argument("--examples-jsonl", type=Path, default=None)
    parser.add_argument("--segments-jsonl", type=Path, default=None)
    parser.add_argument("--labels-jsonl", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260424)
    args = parser.parse_args()

    examples_path = args.examples_jsonl or Path(f"data/felm/{args.output_prefix}_examples.jsonl")
    segments_path = args.segments_jsonl or Path(f"data/felm/{args.output_prefix}_segments.jsonl")
    labels_path = args.labels_jsonl or Path(f"data/annotations/{args.output_prefix}_labels.jsonl")
    summary_path = args.summary_json or Path(f"artifacts/runs/{args.output_prefix}_adapter_summary.json")

    source_path = Path(hf_hub_download("hkust-nlp/felm", "all.jsonl", repo_type="dataset"))
    source_rows = _read_jsonl(source_path)
    domains = {args.domain, *args.include_domain}
    filtered = [row for row in source_rows if row.get("domain") in domains]
    normalized_rows: list[dict[str, Any]] = []
    span_failures: dict[str, list[dict[str, Any]]] = {}
    malformed_rows: dict[str, list[dict[str, Any]]] = {}
    for row in filtered:
        example_id = f"felm-{row['domain']}-{int(row['index']):04d}"
        if not isinstance(row.get("response"), str):
            malformed_rows[example_id] = [
                {
                    "reason": "non_string_response",
                    "response_type": type(row.get("response")).__name__,
                    "response_preview": repr(row.get("response"))[:120],
                }
            ]
            continue
        if not isinstance(row.get("segmented_response"), list):
            malformed_rows[example_id] = [
                {
                    "reason": "non_list_segmented_response",
                    "segmented_response_type": type(row.get("segmented_response")).__name__,
                }
            ]
            continue
        labels = [bool(item) for item in row["labels"]]
        if len(row["segmented_response"]) != len(labels):
            span_failures[example_id] = [
                {
                    "reason": "segment_label_length_mismatch",
                    "n_segments": len(row["segmented_response"]),
                    "n_labels": len(labels),
                }
            ]
            continue
        spans, failures = _find_segment_spans(row["response"], row["segmented_response"])
        if failures:
            span_failures[example_id] = failures
        if len(spans) != len(row["segmented_response"]):
            continue
        normalized_rows.append(
            {
                **row,
                "example_id": example_id,
                "labels": labels,
                "spans": spans,
            }
        )

    split_by_id = _split_examples(normalized_rows, args.seed)
    examples: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    for row in normalized_rows:
        example = ExampleRow(
            example_id=row["example_id"],
            source_corpus="felm",
            task_family="generative_qa",
            contract_id=f"felm_{row['index']}",
            contract_group=row["domain"],
            excerpt_text="FELM has no source excerpt in this adapter; factuality is evaluated against FELM human segment labels.",
            question_text=row["prompt"],
            public_seed_answer=row["response"],
            llama_answer_text=row["response"],
            split=split_by_id[row["example_id"]],
            cross_dist_group=f"felm_{row['domain']}",
        ).validate()
        examples.append(example.as_dict())
        spans_by_segment = {span["segment_index"]: span for span in row["spans"]}
        for segment_index, segment_text in enumerate(row["segmented_response"]):
            claim_id = f"{row['example_id']}-seg-{segment_index:02d}"
            span = spans_by_segment[segment_index]
            segments.append(
                {
                    "claim_id": claim_id,
                    "example_id": row["example_id"],
                    "claim_text": segment_text,
                    "char_start": span["char_start"],
                    "char_end": span["char_end"],
                    "segment_index": segment_index,
                    "domain": row["domain"],
                    "source": row["source"],
                    "annotation_version": "felm_human_v1",
                }
            )
            labels.append(
                {
                    "claim_id": claim_id,
                    "example_id": row["example_id"],
                    "correctness_label": "true" if row["labels"][segment_index] else "false",
                    "load_bearing_label": "yes",
                    "flip_evidence_text": str(row.get("comment", [""])[segment_index] if row.get("comment") else ""),
                    "label_source": "felm_human_annotations",
                }
            )

    write_jsonl(examples_path, examples)
    write_jsonl(segments_path, segments)
    write_jsonl(labels_path, labels)
    summary = {
        "source_dataset": "hkust-nlp/felm",
        "source_file": str(source_path),
        "domains": sorted(domains),
        "n_source_rows": len(source_rows),
        "n_filtered_rows": len(filtered),
        "n_examples": len(examples),
        "n_segments": len(segments),
        "n_malformed_rows": len(malformed_rows),
        "malformed_rows": malformed_rows,
        "n_span_failure_examples": len(span_failures),
        "span_failures": span_failures,
        "split_counts": dict(Counter(example["split"] for example in examples)),
        "label_counts": dict(Counter(label["correctness_label"] for label in labels)),
        "segments_by_split": dict(
            Counter(
                split_by_id[segment["example_id"]]
                for segment in segments
            )
        ),
        "note": (
            "FELM responses were generated by ChatGPT, not Llama. This adapter preserves those "
            "responses and treats each annotated segment as a fixed claim."
        ),
    }
    write_json(summary_path, summary)
    print(f"Wrote {len(examples)} FELM examples to {examples_path}")
    print(f"Wrote {len(segments)} FELM segments to {segments_path}")
    print(f"Wrote {len(labels)} FELM labels to {labels_path}")
    print(f"Wrote adapter summary to {summary_path}")


if __name__ == "__main__":
    main()
