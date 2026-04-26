from __future__ import annotations

import argparse
import json
import re
import statistics
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from sklearn.model_selection import train_test_split

from interp_experiment.io import write_json, write_jsonl
from interp_experiment.schemas import ExampleRow


LABEL_MAP = {"S": "true", "NS": "false"}
FACTSCORE_DATA_ZIP_ID = "155exEdKs7R21gZF4G-x54-XN3qswBcPo"
FACTSCORE_RELEASE_DATE = "2023-11-04"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip()).strip("-").lower()
    return slug[:60] or "unknown"


def _ensure_unzipped(zip_path: Path, data_root: Path) -> None:
    labeled_path = data_root / "labeled" / "ChatGPT.jsonl"
    if labeled_path.exists():
        return
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Missing {zip_path}. Download the official FActScore data.zip "
            f"(Google Drive file id {FACTSCORE_DATA_ZIP_ID}) before running the adapter."
        )
    data_root.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(data_root.parent)
    if not labeled_path.exists():
        raise FileNotFoundError(f"Expected {labeled_path} after unzipping {zip_path}")


def _find_sentence_spans(output: str, annotations: list[dict[str, Any]]) -> tuple[dict[int, tuple[int, int]], list[dict[str, Any]]]:
    spans: dict[int, tuple[int, int]] = {}
    failures: list[dict[str, Any]] = []
    cursor = 0
    for index, annotation in enumerate(annotations):
        sentence = str(annotation.get("text") or "")
        start = output.find(sentence, cursor)
        if start < 0:
            start = output.find(sentence)
        if start < 0:
            failures.append({"sentence_index": index, "sentence_preview": sentence[:160], "reason": "no_exact_sentence_match"})
            continue
        end = start + len(sentence)
        spans[index] = (start, end)
        cursor = end
    return spans, failures


def _split_examples(rows: list[dict[str, Any]], seed: int) -> dict[str, str]:
    ids = [row["example_id"] for row in rows]
    strata: list[str] = []
    for row in rows:
        labels = [claim["correctness_label"] == "true" for claim in row["claims"]]
        if all(labels):
            strata.append("all_true")
        elif not any(labels):
            strata.append("all_false")
        else:
            strata.append("mixed")
    counts = Counter(strata)
    stratify = strata if counts and min(counts.values()) >= 2 else None
    train_ids, temp_ids, train_strata, temp_strata = train_test_split(
        ids,
        strata,
        test_size=0.30,
        random_state=seed,
        stratify=stratify,
    )
    temp_counts = Counter(temp_strata)
    temp_stratify = temp_strata if temp_counts and min(temp_counts.values()) >= 2 else None
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
    parser = argparse.ArgumentParser(description="Convert FActScore ChatGPT human annotations to canonical examples/claims/labels.")
    parser.add_argument("--zip-path", type=Path, default=Path("data/factscore/raw/data.zip"))
    parser.add_argument("--data-root", type=Path, default=Path("data/factscore/raw/data"))
    parser.add_argument("--source-jsonl", type=Path, default=None)
    parser.add_argument("--source-model", default="ChatGPT")
    parser.add_argument("--output-prefix", default="factscore_chatgpt")
    parser.add_argument("--examples-jsonl", type=Path, default=None)
    parser.add_argument("--claims-jsonl", type=Path, default=None)
    parser.add_argument("--labels-jsonl", type=Path, default=None)
    parser.add_argument("--alignment-inspection-jsonl", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260424)
    parser.add_argument("--inspection-size", type=int, default=20)
    args = parser.parse_args()

    _ensure_unzipped(args.zip_path, args.data_root)
    source_path = args.source_jsonl or args.data_root / "labeled" / f"{args.source_model}.jsonl"
    examples_path = args.examples_jsonl or Path(f"data/factscore/{args.output_prefix}_examples.jsonl")
    claims_path = args.claims_jsonl or Path(f"data/factscore/{args.output_prefix}_claims.jsonl")
    labels_path = args.labels_jsonl or Path(f"data/annotations/{args.output_prefix}_labels.jsonl")
    inspection_path = args.alignment_inspection_jsonl or Path(
        f"data/factscore/{args.output_prefix}_alignment_inspection.jsonl"
    )
    summary_path = args.summary_json or Path(f"artifacts/runs/{args.output_prefix}_adapter_summary.json")

    source_rows = _read_jsonl(source_path)
    normalized_rows: list[dict[str, Any]] = []
    sentence_failures: dict[str, list[dict[str, Any]]] = {}
    raw_label_counts: Counter[str] = Counter()
    dropped_ir = 0
    dropped_unknown = 0
    exact_atomic_substring_matches = 0
    total_atomic_facts = 0
    source_sentence_lengths: list[int] = []
    fact_text_lengths: list[int] = []

    for row_index, row in enumerate(source_rows):
        annotations = row.get("annotations") or []
        topic = str(row.get("topic") or f"row-{row_index}")
        example_id = f"factscore-chatgpt-{row_index:04d}-{_slug(topic)}"
        output = str(row.get("output") or "")
        spans, failures = _find_sentence_spans(output, annotations)
        if failures:
            sentence_failures[example_id] = failures
        claims: list[dict[str, Any]] = []
        fact_index = 0
        for sentence_index, annotation in enumerate(annotations):
            if sentence_index not in spans:
                continue
            sentence_text = str(annotation.get("text") or "")
            char_start, char_end = spans[sentence_index]
            for fact in annotation.get("human-atomic-facts") or []:
                total_atomic_facts += 1
                raw_label = str(fact.get("label") or "")
                raw_label_counts[raw_label] += 1
                if raw_label == "IR":
                    dropped_ir += 1
                    continue
                if raw_label not in LABEL_MAP:
                    dropped_unknown += 1
                    continue
                fact_text = str(fact.get("text") or "").strip()
                if not fact_text:
                    dropped_unknown += 1
                    continue
                exact_match = output.find(fact_text) >= 0
                if exact_match:
                    exact_atomic_substring_matches += 1
                source_sentence_lengths.append(char_end - char_start)
                fact_text_lengths.append(len(fact_text))
                claims.append(
                    {
                        "claim_id": f"{example_id}-fact-{fact_index:03d}",
                        "example_id": example_id,
                        "claim_text": fact_text,
                        "char_start": char_start,
                        "char_end": char_end,
                        "segment_index": fact_index,
                        "fact_index": fact_index,
                        "source_sentence_index": sentence_index,
                        "source_sentence_text": sentence_text,
                        "source_sentence_is_relevant": bool(annotation.get("is-relevant")),
                        "source_sentence_char_start": char_start,
                        "source_sentence_char_end": char_end,
                        "span_source": "exact_atomic_substring" if exact_match else "parent_sentence",
                        "raw_factscore_label": raw_label,
                        "correctness_label": LABEL_MAP[raw_label],
                        "annotation_version": "factscore_human_atomic_v1",
                    }
                )
                fact_index += 1
        if claims:
            normalized_rows.append({**row, "example_id": example_id, "claims": claims})

    split_by_id = _split_examples(normalized_rows, args.seed)
    examples: list[dict[str, Any]] = []
    claims_out: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    for row in normalized_rows:
        example = ExampleRow(
            example_id=row["example_id"],
            source_corpus="factscore",
            task_family="generative_qa",
            contract_id=str(row.get("topic") or row["example_id"]),
            contract_group="biography",
            excerpt_text="FActScore biography example; labels are original human atomic-fact annotations.",
            question_text=str(row.get("input") or f"Question: Tell me a bio of {row.get('topic', '')}."),
            public_seed_answer=str(row.get("output") or ""),
            llama_answer_text=str(row.get("output") or ""),
            split=split_by_id[row["example_id"]],
            cross_dist_group="factscore_chatgpt_biography",
        ).validate()
        examples.append(example.as_dict())
        for claim in row["claims"]:
            claims_out.append({key: value for key, value in claim.items() if key != "correctness_label"})
            labels.append(
                {
                    "claim_id": claim["claim_id"],
                    "example_id": claim["example_id"],
                    "correctness_label": claim["correctness_label"],
                    "load_bearing_label": "yes",
                    "flip_evidence_text": f"FActScore human label {claim['raw_factscore_label']}",
                    "label_source": "factscore_human_annotations",
                    "raw_factscore_label": claim["raw_factscore_label"],
                }
            )

    inspection_rows = []
    for label in ("true", "false"):
        matching = [claim for claim, label_row in zip(claims_out, labels) if label_row["correctness_label"] == label]
        inspection_rows.extend(matching[: max(1, args.inspection_size // 4)])
    seen = {row["claim_id"] for row in inspection_rows}
    inspection_rows.extend([row for row in claims_out if row["claim_id"] not in seen][: max(0, args.inspection_size - len(inspection_rows))])
    inspection_rows = inspection_rows[: args.inspection_size]

    write_jsonl(examples_path, examples)
    write_jsonl(claims_path, claims_out)
    write_jsonl(labels_path, labels)
    write_jsonl(inspection_path, inspection_rows)
    label_counts = Counter(label["correctness_label"] for label in labels)
    facts_per_example = [len(row["claims"]) for row in normalized_rows]
    span_source_counts = Counter(claim["span_source"] for claim in claims_out)
    summary = {
        "source": "FActScore official Google Drive data.zip",
        "source_url": "https://github.com/shmsw25/FActScore",
        "google_drive_file_id": FACTSCORE_DATA_ZIP_ID,
        "release_date_from_readme": FACTSCORE_RELEASE_DATE,
        "source_file": str(source_path),
        "source_model": args.source_model,
        "n_source_rows": len(source_rows),
        "n_examples_with_claims": len(examples),
        "n_examples_without_claims": len(source_rows) - len(examples),
        "n_claims": len(claims_out),
        "raw_factscore_label_counts": dict(raw_label_counts),
        "label_counts_after_dropping_ir": dict(label_counts),
        "n_ir_facts_dropped": dropped_ir,
        "n_unknown_facts_dropped": dropped_unknown,
        "total_atomic_facts_seen": total_atomic_facts,
        "exact_atomic_substring_matches": exact_atomic_substring_matches,
        "exact_atomic_substring_match_rate": exact_atomic_substring_matches / total_atomic_facts if total_atomic_facts else None,
        "span_source_counts": dict(span_source_counts),
        "span_policy": (
            "FActScore human atomic facts are canonicalized and almost never appear verbatim in the output. "
            "Activation spans therefore use the parent annotated sentence unless an exact atomic substring exists."
        ),
        "n_sentence_span_failure_examples": len(sentence_failures),
        "sentence_span_failures": sentence_failures,
        "split_counts": dict(Counter(example["split"] for example in examples)),
        "claims_by_split": dict(Counter(split_by_id[claim["example_id"]] for claim in claims_out)),
        "facts_per_example": _quantiles(facts_per_example),
        "source_sentence_char_length": _quantiles(source_sentence_lengths),
        "atomic_fact_text_char_length": _quantiles(fact_text_lengths),
        "generation_mismatch_caveat": (
            "These are ChatGPT-generated biographies. Llama activations are extracted over ChatGPT text, "
            "matching the intentional generation-mismatch design of the scoping check."
        ),
    }
    write_json(summary_path, summary)
    print(f"Wrote {len(examples)} FActScore examples to {examples_path}")
    print(f"Wrote {len(claims_out)} FActScore claims to {claims_path}")
    print(f"Wrote {len(labels)} FActScore labels to {labels_path}")
    print(f"Wrote alignment inspection sample to {inspection_path}")
    print(f"Wrote adapter summary to {summary_path}")


if __name__ == "__main__":
    main()
