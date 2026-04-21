from interp_experiment.data.claims import build_canonical_claims, split_answer_into_claims
from interp_experiment.schemas import AnswerRunRow


def test_split_answer_into_claims_is_deterministic() -> None:
    answer = "California law governs the agreement. Termination requires notice; liability survives."
    expected = [
        "California law governs the agreement.",
        "Termination requires notice;",
        "liability survives.",
    ]
    assert split_answer_into_claims(answer) == expected


def test_build_canonical_claims_produces_ids_and_token_spans() -> None:
    answer_run = AnswerRunRow(
        example_id="ex-1",
        source_corpus="cuad",
        task_family="field_extraction",
        prompt_text="Prompt",
        answer_text="California law governs the agreement. Liability survives.",
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        extractor_name="transformer_lens",
        token_ids=[10, 11, 12, 13, 14, 15],
        token_offsets=[(0, 10), (11, 14), (15, 18), (19, 22), (23, 33), (34, 43)],
    ).validate()
    claims = build_canonical_claims(answer_run)
    assert len(claims) == 2
    assert claims[0].token_start == 0
    assert claims[0].token_end >= claims[0].token_start
