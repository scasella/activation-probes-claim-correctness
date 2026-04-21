from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from ..io import write_json, write_jsonl
from ..schemas import ExampleRow


def write_source_pool(examples: list[ExampleRow], output_path: Path, manifest_path: Path) -> None:
    write_jsonl(output_path, [example.as_dict() for example in examples])
    manifest: dict[str, Any] = {
        "n_examples": len(examples),
        "by_source": dict(Counter(example.source_corpus for example in examples)),
        "by_cross_dist_group": dict(Counter(example.cross_dist_group for example in examples)),
        "by_split": dict(Counter(example.split for example in examples)),
    }
    write_json(manifest_path, manifest)
