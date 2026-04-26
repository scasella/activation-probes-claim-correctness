from __future__ import annotations

import json
from json import JSONDecodeError
from typing import Any

from ..schemas import BaselinePrediction


def extract_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[idx:])
        except JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("Could not locate a JSON object in model output")


def validate_prediction_claim_ids(predictions: list[BaselinePrediction], expected_claim_ids: list[str]) -> None:
    observed = [prediction.claim_id for prediction in predictions]
    if len(observed) != len(expected_claim_ids):
        raise ValueError(f"Expected {len(expected_claim_ids)} claims, found {len(observed)}")
    if sorted(observed) != sorted(expected_claim_ids):
        raise ValueError(
            f"Prediction claim ids do not match expected ids. observed={sorted(observed)} expected={sorted(expected_claim_ids)}"
        )


def normalize_prediction_claim_ids(
    predictions: list[BaselinePrediction],
    expected_claim_ids: list[str],
) -> list[BaselinePrediction]:
    expected_set = set(expected_claim_ids)
    if len(predictions) == 1 and len(expected_claim_ids) == 1:
        predictions[0].claim_id = expected_claim_ids[0]
        return predictions
    normalized: list[BaselinePrediction] = []
    for prediction in predictions:
        claim_id = prediction.claim_id
        if claim_id not in expected_set:
            prefixed = f"claim-{claim_id}"
            if prefixed in expected_set:
                prediction.claim_id = prefixed
        normalized.append(prediction)
    return normalized


def build_baseline_output_schema(expected_claim_ids: list[str]) -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "minItems": len(expected_claim_ids),
                "maxItems": len(expected_claim_ids),
                "items": {
                    "type": "object",
                    "properties": {
                        "claim_id": {"type": "string", "enum": expected_claim_ids},
                        "correctness_confidence": {"type": "number"},
                        "load_bearing_label": {"type": "string", "enum": ["yes", "no"]},
                        "load_bearing_confidence": {"type": "number"},
                        "flip_evidence_text": {"type": "string"},
                    },
                    "required": [
                        "claim_id",
                        "correctness_confidence",
                        "load_bearing_label",
                        "load_bearing_confidence",
                        "flip_evidence_text",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["claims"],
        "additionalProperties": False,
    }
