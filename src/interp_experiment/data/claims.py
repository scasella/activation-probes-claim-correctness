from __future__ import annotations

import re
from dataclasses import dataclass

from ..schemas import ClaimRow, ExampleRow
from ..utils import normalize_whitespace, stable_hash

CLAIM_SPLIT_RE = re.compile(r"(?:\n+|(?<=;)\s+|(?<=[.:])\s+(?=[A-Z0-9]))")


@dataclass(slots=True)
class TokenizedText:
    text: str
    tokens: list[str]
    offsets: list[tuple[int, int]]


def whitespace_tokenize(text: str) -> TokenizedText:
    tokens: list[str] = []
    offsets: list[tuple[int, int]] = []
    for match in re.finditer(r"\S+", text):
        tokens.append(match.group(0))
        offsets.append((match.start(), match.end()))
    return TokenizedText(text=text, tokens=tokens, offsets=offsets)


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


def locate_claim_span(answer_text: str, claim_text: str, tokenized: TokenizedText | None = None) -> tuple[int, int]:
    tokenized = tokenized or whitespace_tokenize(answer_text)
    start_char = answer_text.find(claim_text)
    if start_char < 0:
        raise ValueError(f"Could not locate claim span in answer: {claim_text!r}")
    end_char = start_char + len(claim_text)
    start_token = None
    end_token = None
    for idx, (token_start, token_end) in enumerate(tokenized.offsets):
        if start_token is None and token_end > start_char:
            start_token = idx
        if token_start < end_char:
            end_token = idx
        if token_start >= end_char:
            break
    if start_token is None or end_token is None:
        raise ValueError(f"Could not translate claim span to token span: {claim_text!r}")
    return start_token, end_token


def build_canonical_claims(
    example: ExampleRow,
    annotation_version: str = "v1",
) -> list[ClaimRow]:
    if not example.llama_answer_text.strip():
        raise ValueError("Example must include llama_answer_text before building canonical claims")
    tokenized = whitespace_tokenize(example.llama_answer_text)
    claims = split_answer_into_claims(example.llama_answer_text)
    rows: list[ClaimRow] = []
    for claim_text in claims:
        token_start, token_end = locate_claim_span(example.llama_answer_text, claim_text, tokenized)
        claim_id = stable_hash({"example_id": example.example_id, "claim_text": claim_text})
        rows.append(
            ClaimRow(
                claim_id=f"claim-{claim_id}",
                example_id=example.example_id,
                claim_text=claim_text,
                token_start=token_start,
                token_end=token_end,
                correctness_label="false",
                load_bearing_label="no",
                flip_evidence_text="",
                annotation_version=annotation_version,
            ).validate()
        )
    return rows
