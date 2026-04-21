from __future__ import annotations

import random
from collections import defaultdict

from ..schemas import ExampleRow


def freeze_contract_splits(
    examples: list[ExampleRow],
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    seed: int = 7,
) -> list[ExampleRow]:
    if round(train_ratio + validation_ratio + test_ratio, 6) != 1.0:
        raise ValueError("split ratios must sum to 1.0")

    groups: dict[str, dict[str, list[ExampleRow]]] = defaultdict(lambda: defaultdict(list))
    for example in examples:
        groups[example.cross_dist_group][example.contract_id].append(example)

    rng = random.Random(seed)
    assigned: list[ExampleRow] = []
    for cross_group, contracts in groups.items():
        contract_ids = list(contracts)
        rng.shuffle(contract_ids)
        total = len(contract_ids)
        train_cut = int(total * train_ratio)
        validation_cut = train_cut + int(total * validation_ratio)
        split_map: dict[str, str] = {}
        for idx, contract_id in enumerate(contract_ids):
            if idx < train_cut:
                split_map[contract_id] = "train"
            elif idx < validation_cut:
                split_map[contract_id] = "validation"
            else:
                split_map[contract_id] = "test"
        for contract_id, rows in contracts.items():
            split_name = split_map[contract_id]
            for row in rows:
                assigned.append(
                    ExampleRow(
                        example_id=row.example_id,
                        source_corpus=row.source_corpus,
                        contract_id=row.contract_id,
                        contract_group=row.contract_group,
                        excerpt_text=row.excerpt_text,
                        question_text=row.question_text,
                        public_seed_answer=row.public_seed_answer,
                        llama_answer_text=row.llama_answer_text,
                        split=split_name,
                        cross_dist_group=cross_group,
                    ).validate()
                )
    return assigned
