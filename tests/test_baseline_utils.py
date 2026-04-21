from interp_experiment.baselines.llama_self_report import parse_baseline_claims
from interp_experiment.baselines.utils import (
    build_baseline_output_schema,
    extract_json_object,
    normalize_prediction_claim_ids,
    validate_prediction_claim_ids,
)


def test_extract_json_object_finds_embedded_payload() -> None:
    payload = extract_json_object("preface {\"claims\": []} suffix")
    assert payload == {"claims": []}


def test_validate_prediction_claim_ids_accepts_matching_ids() -> None:
    predictions = parse_baseline_claims(
        {
            "claims": [
                {
                    "claim_id": "claim-1",
                    "correctness_confidence": 0.5,
                    "load_bearing_label": "yes",
                    "load_bearing_confidence": 0.8,
                    "flip_evidence_text": "Clause X",
                }
            ]
        },
        prompt_version="v1",
        model_name="gpt-5.4",
    )
    validate_prediction_claim_ids(predictions, ["claim-1"])


def test_build_baseline_output_schema_uses_claim_ids() -> None:
    schema = build_baseline_output_schema(["claim-a", "claim-b"])
    assert schema["properties"]["claims"]["maxItems"] == 2
    assert schema["properties"]["claims"]["items"]["properties"]["claim_id"]["enum"] == ["claim-a", "claim-b"]


def test_normalize_prediction_claim_ids_restores_prefix() -> None:
    predictions = parse_baseline_claims(
        {
            "claims": [
                {
                    "claim_id": "abc123",
                    "correctness_confidence": 0.5,
                    "load_bearing_label": "yes",
                    "load_bearing_confidence": 0.8,
                    "flip_evidence_text": "Clause X",
                }
            ]
        },
        prompt_version="v1",
        model_name="llama",
    )
    normalized = normalize_prediction_claim_ids(predictions, ["claim-abc123"])
    assert normalized[0].claim_id == "claim-abc123"
