from __future__ import annotations

from typing import Any

from ..schemas import BaselinePrediction


def parse_baseline_claims(raw_payload: dict[str, Any], prompt_version: str, model_name: str) -> list[BaselinePrediction]:
    claims = raw_payload.get("claims", [])
    if not isinstance(claims, list):
        raise ValueError("Expected raw_payload['claims'] to be a list")
    parsed: list[BaselinePrediction] = []
    for item in claims:
        if not isinstance(item, dict):
            raise ValueError("Each claim payload must be a dict")
        parsed.append(
            BaselinePrediction(
                claim_id=item["claim_id"],
                correctness_confidence=float(item["correctness_confidence"]),
                load_bearing_label=item["load_bearing_label"],
                load_bearing_confidence=float(item["load_bearing_confidence"]),
                flip_evidence_text=item.get("flip_evidence_text", ""),
                raw_json=item,
                prompt_version=prompt_version,
                model_name=model_name,
            ).validate()
        )
    return parsed
