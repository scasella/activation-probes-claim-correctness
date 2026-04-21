from interp_experiment.schemas import BaselinePrediction, ClaimFeatureRow, ClaimRow, ExampleRow


def test_example_row_validation_accepts_expected_payload() -> None:
    row = ExampleRow(
        example_id="ex-1",
        source_corpus="maud",
        contract_id="contract-a",
        contract_group="merger_agreement",
        excerpt_text="Clause text",
        question_text="Question?",
        public_seed_answer="Seed answer",
        llama_answer_text="",
        split="unassigned",
        cross_dist_group="maud_merger",
    ).validate()
    assert row.example_id == "ex-1"


def test_claim_row_rejects_bad_label() -> None:
    try:
        ClaimRow(
            claim_id="claim-1",
            example_id="ex-1",
            claim_text="Claim text",
            token_start=0,
            token_end=1,
            correctness_label="maybe",
            load_bearing_label="yes",
            flip_evidence_text="",
            annotation_version="v1",
        ).validate()
    except ValueError as exc:
        assert "correctness_label" in str(exc)
    else:
        raise AssertionError("Expected validation to fail")


def test_baseline_prediction_requires_probabilities() -> None:
    prediction = BaselinePrediction(
        claim_id="claim-1",
        correctness_confidence=0.8,
        load_bearing_label="yes",
        load_bearing_confidence=0.6,
        flip_evidence_text="Contrary clause language.",
        raw_json={"claim_id": "claim-1"},
        prompt_version="v1",
        model_name="gpt-5.4",
    ).validate()
    assert prediction.load_bearing_confidence == 0.6


def test_claim_feature_row_coerces_numeric_vector() -> None:
    row = ClaimFeatureRow(
        claim_id="claim-1",
        example_id="ex-1",
        feature_source="residual",
        vector=[1, 2, 3],
        correctness_target=0.5,
        load_bearing_target=1,
        stability_target=0,
    ).validate()
    assert row.vector == [1.0, 2.0, 3.0]
