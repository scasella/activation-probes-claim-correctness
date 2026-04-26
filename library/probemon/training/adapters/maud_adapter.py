from __future__ import annotations

import json
from pathlib import Path

from probemon.training.dataset import CanonicalClaim, CanonicalDataset, CanonicalExample, write_canonical_dataset


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _label(value: str) -> str:
    if value == "true":
        return "correct"
    if value == "partially_true":
        return "partial"
    return "incorrect"


def convert_maud_to_canonical(
    *,
    examples_jsonl: str | Path,
    claims_jsonl: str | Path,
    labels_jsonl: str | Path,
    output_path: str | Path | None = None,
) -> CanonicalDataset:
    """Convert MAUD artifacts to canonical format.

    MAUD claim spans in the methods artifacts are token spans over the
    extracted claim text, not substring spans inside the short Llama answer.
    The canonical example therefore uses each fixed claim as the generation
    text for training-data scaffolding. Runtime monitoring still segments and
    scores normal generations.
    """

    examples = {row["example_id"]: row for row in _read_jsonl(Path(examples_jsonl))}
    labels = {row["claim_id"]: row for row in _read_jsonl(Path(labels_jsonl))}
    canonical: list[CanonicalExample] = []
    for claim in _read_jsonl(Path(claims_jsonl)):
        label_row = labels.get(claim["claim_id"])
        if label_row is None:
            continue
        example = examples[claim["example_id"]]
        text = str(claim["claim_text"])
        canonical.append(
            CanonicalExample(
                prompt=f"{example.get('question_text', '')}\n\n{example.get('excerpt_text', '')}",
                generation=text,
                claims=[
                    CanonicalClaim(
                        text=text,
                        char_start=0,
                        char_end=len(text),
                        label=_label(str(label_row["correctness_label"])),
                        metadata={"claim_id": claim["claim_id"], "example_id": claim["example_id"]},
                    )
                ],
                metadata={
                    "example_id": claim["example_id"],
                    "split": example.get("split", "unassigned"),
                    "source": "maud",
                    "label_source": label_row.get("label_source", "judge_llm_proxy"),
                },
            )
        )
    dataset = CanonicalDataset(examples=canonical, name="maud-legal-qa")
    if output_path is not None:
        write_canonical_dataset(dataset, output_path)
    return dataset
