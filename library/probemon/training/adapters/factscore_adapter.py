from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from probemon.training.dataset import CanonicalClaim, CanonicalDataset, CanonicalExample, write_canonical_dataset


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _label(raw: str) -> str | None:
    if raw in {"S", "Supported", "true"}:
        return "correct"
    if raw in {"NS", "Not-supported", "false"}:
        return "incorrect"
    return None


def convert_factscore_to_canonical(
    *,
    examples_jsonl: str | Path,
    claims_jsonl: str | Path,
    output_path: str | Path | None = None,
) -> CanonicalDataset:
    """Convert FActScore ChatGPT biographies to canonical format.

    FActScore atomic facts are canonicalized and almost never occur verbatim
    in the generated biography. Following the methods paper, this adapter
    uses the parent sentence as `claims[].text` and its character range as
    `char_start` / `char_end`, while keeping the atomic-fact correctness
    label. This sentence-span pooling compromise is also recorded in the
    shipped biography probe metadata and README.
    """

    examples = {row["example_id"]: row for row in _read_jsonl(Path(examples_jsonl))}
    grouped: dict[str, list[CanonicalClaim]] = defaultdict(list)
    for row in _read_jsonl(Path(claims_jsonl)):
        label = _label(str(row.get("raw_factscore_label", "")))
        if label is None:
            continue
        text = str(row.get("source_sentence_text") or row["claim_text"])
        start = int(row["source_sentence_char_start"] if "source_sentence_char_start" in row else row["char_start"])
        end = int(row["source_sentence_char_end"] if "source_sentence_char_end" in row else row["char_end"])
        grouped[row["example_id"]].append(
            CanonicalClaim(
                text=text,
                char_start=start,
                char_end=end,
                label=label,
                metadata={
                    "claim_id": row["claim_id"],
                    "atomic_fact_text": row["claim_text"],
                    "span_source": row.get("span_source", "parent_sentence"),
                    "raw_factscore_label": row.get("raw_factscore_label"),
                },
            )
        )
    canonical = [
        CanonicalExample(
            prompt=example["question_text"],
            generation=example["llama_answer_text"],
            claims=grouped[example_id],
            metadata={
                "example_id": example_id,
                "split": example.get("split", "unassigned"),
                "source": "factscore",
                "label_source": "factscore_human_annotations",
            },
        )
        for example_id, example in examples.items()
        if grouped.get(example_id)
    ]
    dataset = CanonicalDataset(examples=canonical, name="factscore-biographies")
    if output_path is not None:
        write_canonical_dataset(dataset, output_path)
    return dataset
