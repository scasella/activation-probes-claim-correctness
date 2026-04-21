from __future__ import annotations

import re

from ..schemas import AnswerRunRow, ClaimRow
from ..utils import normalize_whitespace, stable_hash

CLAIM_SPLIT_RE = re.compile(r"(?:\n+|(?<=;)\s+|(?<=[.:])\s+(?=[A-Z0-9]))")


def split_answer_into_claims(answer_text: str) -> list[str]:
    normalized = normalize_whitespace(answer_text.replace("\r\n", "\n"))
    if not normalized:
        return []
    candidates = [normalize_whitespace(chunk) for chunk in CLAIM_SPLIT_RE.split(answer_text) if normalize_whitespace(chunk)]
    claims: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in claims:
            claims.append(candidate)
    return claims


def locate_claim_span(
    answer_text: str,
    claim_text: str,
    token_offsets: list[tuple[int, int]],
) -> tuple[int, int]:
    start_char = answer_text.find(claim_text)
    if start_char < 0:
        raise ValueError(f"Could not locate claim span in answer: {claim_text!r}")
    end_char = start_char + len(claim_text)
    start_token = None
    end_token = None
    for idx, (token_start, token_end) in enumerate(token_offsets):
        if start_token is None and token_end > start_char:
            start_token = idx
        if token_start < end_char:
            end_token = idx
        if token_start >= end_char:
            break
    if start_token is None or end_token is None:
        raise ValueError(f"Could not map claim span to token offsets: {claim_text!r}")
    return start_token, end_token


def build_canonical_claims(
    answer_run: AnswerRunRow,
    annotation_version: str = "v1",
) -> list[ClaimRow]:
    if not answer_run.answer_text.strip():
        raise ValueError("Answer run must include answer_text before building canonical claims")
    claims = split_answer_into_claims(answer_run.answer_text)
    rows: list[ClaimRow] = []
    for claim_text in claims:
        token_start, token_end = locate_claim_span(answer_run.answer_text, claim_text, answer_run.token_offsets)
        claim_id = stable_hash({"example_id": answer_run.example_id, "claim_text": claim_text})
        rows.append(
            ClaimRow(
                claim_id=f"claim-{claim_id}",
                example_id=answer_run.example_id,
                claim_text=claim_text,
                token_start=token_start,
                token_end=token_end,
                annotation_version=annotation_version,
            ).validate()
        )
    return rows
