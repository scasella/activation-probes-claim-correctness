from interp_experiment.data.seed_corpora import _sample_contract_diverse_rows
from interp_experiment.schemas import ExampleRow


def _example(source: str, contract_id: str, idx: int) -> ExampleRow:
    return ExampleRow(
        example_id=f"{source}-{contract_id}-{idx}",
        source_corpus=source,
        task_family="field_extraction" if source == "cuad" else "generative_qa",
        contract_id=contract_id,
        contract_group="group",
        excerpt_text="Excerpt",
        question_text="Question?",
        public_seed_answer="Seed",
        llama_answer_text="",
        split="unassigned",
        cross_dist_group=f"{source}_group",
    ).validate()


def test_contract_diverse_sampling_respects_cap() -> None:
    rows = []
    for contract_id in ("a", "b", "c", "d"):
        for idx in range(4):
            rows.append(_example("cuad", contract_id, idx))
    sampled = _sample_contract_diverse_rows(rows, target_examples=8, max_rows_per_contract=2, seed=4)
    counts = {}
    for row in sampled:
        counts[row.contract_id] = counts.get(row.contract_id, 0) + 1
    assert all(count <= 2 for count in counts.values())
