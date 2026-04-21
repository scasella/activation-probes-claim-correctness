from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..config import load_config
from ..schemas import ExampleRow
from ..utils import normalize_whitespace, stable_hash


def _load_datasets_module():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "datasets is required for source-pool construction. Install with `uv sync --extra data`."
        ) from exc
    return load_dataset


def _iter_all_splits(dataset_obj: Any) -> Iterable[dict[str, Any]]:
    if hasattr(dataset_obj, "keys"):
        for split_name in dataset_obj.keys():
            for row in dataset_obj[split_name]:
                yield dict(row)
        return
    for row in dataset_obj:
        yield dict(row)


def _extract_answers(raw_value: Any) -> str:
    if isinstance(raw_value, str):
        return normalize_whitespace(raw_value)
    if isinstance(raw_value, list):
        return normalize_whitespace("; ".join(str(item) for item in raw_value if str(item).strip()))
    if isinstance(raw_value, dict):
        if "text" in raw_value and isinstance(raw_value["text"], list):
            return normalize_whitespace("; ".join(str(item) for item in raw_value["text"] if str(item).strip()))
        if "text" in raw_value and isinstance(raw_value["text"], str):
            return normalize_whitespace(raw_value["text"])
    return ""


def _maud_row_to_example(row: dict[str, Any], source_cfg: dict[str, Any]) -> ExampleRow | None:
    excerpt = normalize_whitespace(str(row.get(source_cfg["excerpt_field"], "")))
    question = normalize_whitespace(str(row.get(source_cfg["question_field"], "")))
    answer = _extract_answers(row.get(source_cfg["answer_field"], ""))
    contract_id = normalize_whitespace(str(row.get(source_cfg["contract_id_field"], "")))
    if not excerpt or not question or not answer or not contract_id:
        return None
    example_id = stable_hash(
        {
            "source": "maud",
            "contract": contract_id,
            "question": question,
            "answer": answer,
        }
    )
    return ExampleRow(
        example_id=f"maud-{example_id}",
        source_corpus="maud",
        contract_id=contract_id,
        contract_group=source_cfg["contract_group"],
        excerpt_text=excerpt,
        question_text=question,
        public_seed_answer=answer,
        llama_answer_text="",
        split="unassigned",
        cross_dist_group=source_cfg["cross_dist_group"],
    ).validate()


def _cuad_row_to_example(row: dict[str, Any], source_cfg: dict[str, Any]) -> ExampleRow | None:
    excerpt = normalize_whitespace(
        str(row.get(source_cfg.get("excerpt_field", "context")) or row.get("clause") or "")
    )
    question = normalize_whitespace(str(row.get(source_cfg.get("question_field", "question"), "")))
    answer = _extract_answers(row.get(source_cfg.get("answer_field", "answers"), ""))
    contract_id = normalize_whitespace(
        str(row.get(source_cfg.get("contract_id_field", "title")) or row.get("id") or "")
    )
    if not excerpt or not question or not answer or not contract_id:
        return None
    example_id = stable_hash(
        {
            "source": "cuad",
            "contract": contract_id,
            "question": question,
            "answer": answer,
        }
    )
    return ExampleRow(
        example_id=f"cuad-{example_id}",
        source_corpus="cuad",
        contract_id=contract_id,
        contract_group=source_cfg["contract_group"],
        excerpt_text=excerpt,
        question_text=question,
        public_seed_answer=answer,
        llama_answer_text="",
        split="unassigned",
        cross_dist_group=source_cfg["cross_dist_group"],
    ).validate()


ADAPTERS = {
    "maud": _maud_row_to_example,
    "cuad_qa": _cuad_row_to_example,
}


def build_hybrid_source_pool(dataset_config_path: str | None = None) -> list[ExampleRow]:
    dataset_cfg = load_config("dataset.yaml") if dataset_config_path is None else load_config(dataset_config_path)
    load_dataset = _load_datasets_module()
    source_rows: list[ExampleRow] = []
    seen_ids: set[str] = set()

    for source_name, source_cfg in dataset_cfg["sources"].items():
        adapter_name = source_cfg["adapter"]
        adapter = ADAPTERS[adapter_name]
        dataset_obj = load_dataset(source_cfg["hf_dataset"])
        target_examples = int(source_cfg["target_examples"])
        for row in _iter_all_splits(dataset_obj):
            example = adapter(row, source_cfg)
            if example is None or example.example_id in seen_ids:
                continue
            seen_ids.add(example.example_id)
            source_rows.append(example)
            if sum(item.source_corpus == source_name for item in source_rows) >= target_examples:
                break

    return source_rows
