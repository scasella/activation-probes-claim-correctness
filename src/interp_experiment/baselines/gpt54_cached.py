from __future__ import annotations

from pathlib import Path
from typing import Any

from ..io import read_json, write_jsonl
from ..schemas import BaselinePrediction, ClaimRow, ExampleRow
from .codex_app_server import build_request_packet, write_request_packet
from .llama_self_report import parse_baseline_claims
from .prompts import render_gpt54_prompt


def export_gpt54_request_packet(example: ExampleRow, claims: list[ClaimRow], output_path: Path) -> dict[str, Any]:
    prompt = render_gpt54_prompt(example, claims)
    packet = build_request_packet(
        example_id=example.example_id,
        system_prompt=prompt["system"],
        user_prompt=prompt["user"],
        prompt_version=prompt["prompt_version"],
    )
    write_request_packet(packet, output_path)
    return packet


def ingest_gpt54_cached_output(raw_path: Path, parsed_output_path: Path, prompt_version: str) -> list[BaselinePrediction]:
    raw_payload = read_json(raw_path)
    parsed = parse_baseline_claims(raw_payload, prompt_version=prompt_version, model_name="gpt-5.4")
    write_jsonl(parsed_output_path, [row.as_dict() for row in parsed])
    return parsed
