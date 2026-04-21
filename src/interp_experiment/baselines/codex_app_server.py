from __future__ import annotations

from pathlib import Path
from typing import Any

from ..io import write_json
from ..utils import ensure_parent


def build_request_packet(
    example_id: str,
    system_prompt: str,
    user_prompt: str,
    prompt_version: str,
) -> dict[str, Any]:
    return {
        "example_id": example_id,
        "transport": "codex_app_server_cached_artifact",
        "prompt_version": prompt_version,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "expected_schema": {
            "claims": [
                {
                    "claim_id": "string",
                    "correctness_confidence": "float[0,1]",
                    "load_bearing_label": "yes|no",
                    "load_bearing_confidence": "float[0,1]",
                    "flip_evidence_text": "string",
                }
            ]
        },
    }


def write_request_packet(packet: dict[str, Any], path: Path) -> None:
    ensure_parent(path)
    write_json(path, packet)
