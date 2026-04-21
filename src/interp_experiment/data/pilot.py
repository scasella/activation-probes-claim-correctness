from __future__ import annotations

import random
from collections import defaultdict

from ..schemas import ClaimRow, ExampleRow


def sample_pilot_examples(
    examples: list[ExampleRow],
    pilot_size: int = 30,
    stratify_field: str = "source_corpus",
    seed: int = 11,
) -> list[ExampleRow]:
    if pilot_size <= 0:
        raise ValueError("pilot_size must be positive")
    if pilot_size > len(examples):
        raise ValueError("pilot_size cannot exceed the number of available examples")

    groups: dict[str, list[ExampleRow]] = defaultdict(list)
    for example in examples:
        groups[str(getattr(example, stratify_field))].append(example)
    if not groups:
        raise ValueError("No groups available for pilot sampling")

    rng = random.Random(seed)
    ordered_groups = sorted(groups)
    for group_name in ordered_groups:
        rng.shuffle(groups[group_name])

    base = pilot_size // len(ordered_groups)
    remainder = pilot_size % len(ordered_groups)
    allocation = {
        group_name: base + (1 if idx < remainder else 0)
        for idx, group_name in enumerate(ordered_groups)
    }
    sampled: list[ExampleRow] = []
    for group_name in ordered_groups:
        target = allocation[group_name]
        if target > len(groups[group_name]):
            raise ValueError(f"Not enough examples in group {group_name} for target {target}")
        sampled.extend(groups[group_name][:target])
    return sampled


def build_claim_annotation_packet(
    examples: list[ExampleRow],
    claims: list[ClaimRow],
    annotator_ids: list[str],
    annotation_version: str = "v1",
) -> list[dict[str, object]]:
    example_by_id = {example.example_id: example for example in examples}
    packet_rows: list[dict[str, object]] = []
    for annotator_id in annotator_ids:
        for claim in claims:
            example = example_by_id[claim.example_id]
            packet_rows.append(
                {
                    "annotator_id": annotator_id,
                    "annotation_version": annotation_version,
                    "example_id": example.example_id,
                    "source_corpus": example.source_corpus,
                    "contract_id": example.contract_id,
                    "question_text": example.question_text,
                    "excerpt_text": example.excerpt_text,
                    "llama_answer_text": example.llama_answer_text,
                    "claim_id": claim.claim_id,
                    "claim_text": claim.claim_text,
                    "correctness_label": "",
                    "load_bearing_label": "",
                    "flip_evidence_text": "",
                    "notes": "",
                }
            )
    return packet_rows
