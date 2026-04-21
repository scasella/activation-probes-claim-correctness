from interp_experiment.data.split_freeze import freeze_contract_splits
from interp_experiment.schemas import ExampleRow


def _make_example(example_id: str, contract_id: str, cross_group: str) -> ExampleRow:
    return ExampleRow(
        example_id=example_id,
        source_corpus="maud" if cross_group == "maud_merger" else "cuad",
        contract_id=contract_id,
        contract_group="merger_agreement" if cross_group == "maud_merger" else "commercial_contract",
        excerpt_text="Excerpt",
        question_text="Question?",
        public_seed_answer="Answer",
        llama_answer_text="",
        split="unassigned",
        cross_dist_group=cross_group,
    ).validate()


def test_freeze_contract_splits_keeps_contracts_together() -> None:
    examples = [
        _make_example("a1", "contract-a", "maud_merger"),
        _make_example("a2", "contract-a", "maud_merger"),
        _make_example("b1", "contract-b", "maud_merger"),
        _make_example("c1", "contract-c", "cuad_non_merger"),
        _make_example("c2", "contract-c", "cuad_non_merger"),
        _make_example("d1", "contract-d", "cuad_non_merger"),
    ]
    frozen = freeze_contract_splits(examples, 0.5, 0.25, 0.25, seed=11)
    by_contract = {}
    for row in frozen:
        by_contract.setdefault(row.contract_id, set()).add(row.split)
    assert all(len(splits) == 1 for splits in by_contract.values())
