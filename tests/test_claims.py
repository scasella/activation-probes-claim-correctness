from interp_experiment.data.claims import build_canonical_claims, split_answer_into_claims
from interp_experiment.schemas import ExampleRow


def test_split_answer_into_claims_is_deterministic() -> None:
    answer = "California law governs the agreement. Termination requires notice; liability survives."
    expected = [
        "California law governs the agreement.",
        "Termination requires notice;",
        "liability survives.",
    ]
    assert split_answer_into_claims(answer) == expected


def test_build_canonical_claims_produces_ids_and_token_spans() -> None:
    example = ExampleRow(
        example_id="ex-1",
        source_corpus="cuad",
        contract_id="contract-a",
        contract_group="commercial_contract",
        excerpt_text="Text",
        question_text="Question?",
        public_seed_answer="Seed",
        llama_answer_text="California law governs the agreement. Liability survives.",
        split="train",
        cross_dist_group="cuad_non_merger",
    ).validate()
    claims = build_canonical_claims(example)
    assert len(claims) == 2
    assert claims[0].token_start == 0
    assert claims[0].token_end >= claims[0].token_start
