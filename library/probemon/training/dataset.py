from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


CanonicalLabel = Literal["correct", "incorrect", "partial"]


@dataclass(slots=True)
class CanonicalClaim:
    text: str
    char_start: int
    char_end: int
    label: CanonicalLabel
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CanonicalExample:
    prompt: str
    generation: str
    claims: list[CanonicalClaim]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CanonicalDataset:
    examples: list[CanonicalExample]
    name: str = "custom"

    @property
    def n_claims(self) -> int:
        return sum(len(example.claims) for example in self.examples)

    def iter_claims(self):
        for example in self.examples:
            for claim in example.claims:
                yield example, claim


def _claim_from_dict(payload: dict[str, Any]) -> CanonicalClaim:
    label = payload["label"]
    if label not in {"correct", "incorrect", "partial"}:
        raise ValueError(f"Unsupported canonical label: {label!r}")
    return CanonicalClaim(
        text=str(payload["text"]),
        char_start=int(payload["char_start"]),
        char_end=int(payload["char_end"]),
        label=label,
        metadata=dict(payload.get("metadata", {})),
    )


def load_canonical_dataset(path: str | Path, *, name: str | None = None) -> CanonicalDataset:
    path = Path(path)
    examples: list[CanonicalExample] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            examples.append(
                CanonicalExample(
                    prompt=str(row["prompt"]),
                    generation=str(row["generation"]),
                    claims=[_claim_from_dict(item) for item in row["claims"]],
                    metadata=dict(row.get("metadata", {})),
                )
            )
    return CanonicalDataset(examples=examples, name=name or path.stem)


def write_canonical_dataset(dataset: CanonicalDataset, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in dataset.examples:
            row = {
                "prompt": example.prompt,
                "generation": example.generation,
                "claims": [
                    {
                        "text": claim.text,
                        "char_start": claim.char_start,
                        "char_end": claim.char_end,
                        "label": claim.label,
                        "metadata": claim.metadata,
                    }
                    for claim in example.claims
                ],
                "metadata": example.metadata,
            }
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
