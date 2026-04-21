from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..io import write_json, write_jsonl
from ..schemas import ExampleRow


def write_source_pool(examples: list[ExampleRow], output_path: Path, manifest_path: Path) -> None:
    write_jsonl(output_path, [example.as_dict() for example in examples])
    by_source_contracts: dict[str, set[str]] = defaultdict(set)
    by_source_contract_rows: dict[str, Counter[str]] = defaultdict(Counter)
    for example in examples:
        by_source_contracts[example.source_corpus].add(example.contract_id)
        by_source_contract_rows[example.source_corpus][example.contract_id] += 1
    manifest: dict[str, Any] = {
        "n_examples": len(examples),
        "by_source": dict(Counter(example.source_corpus for example in examples)),
        "by_task_family": dict(Counter(example.task_family for example in examples)),
        "by_cross_dist_group": dict(Counter(example.cross_dist_group for example in examples)),
        "by_split": dict(Counter(example.split for example in examples)),
        "unique_contract_count_by_source": {
            source: len(contract_ids)
            for source, contract_ids in by_source_contracts.items()
        },
        "max_rows_per_contract_by_source": {
            source: max(counter.values(), default=0)
            for source, counter in by_source_contract_rows.items()
        },
        "top_repeated_contracts_by_source": {
            source: counter.most_common(5)
            for source, counter in by_source_contract_rows.items()
        },
    }
    write_json(manifest_path, manifest)
