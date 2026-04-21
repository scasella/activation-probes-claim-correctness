from __future__ import annotations

from ..config import load_config
from ..schemas import ClaimRow, ExampleRow


def canonical_claims_block(claims: list[ClaimRow]) -> str:
    return "\n".join(f"- {claim.claim_id}: {claim.claim_text}" for claim in claims)


def render_self_report_prompt(example: ExampleRow, claims: list[ClaimRow]) -> dict[str, str]:
    cfg = load_config("prompts.yaml")
    return {
        "system": cfg["self_report_system"].strip(),
        "user": cfg["self_report_user_template"].format(
            excerpt_text=example.excerpt_text,
            question_text=example.question_text,
            llama_answer_text=example.llama_answer_text,
            canonical_claims_block=canonical_claims_block(claims),
        ).strip(),
        "prompt_version": cfg["prompt_version"],
    }


def render_gpt54_prompt(example: ExampleRow, claims: list[ClaimRow]) -> dict[str, str]:
    cfg = load_config("prompts.yaml")
    return {
        "system": cfg["gpt54_system"].strip(),
        "user": cfg["gpt54_user_template"].format(
            excerpt_text=example.excerpt_text,
            question_text=example.question_text,
            llama_answer_text=example.llama_answer_text,
            canonical_claims_block=canonical_claims_block(claims),
        ).strip(),
        "prompt_version": cfg["prompt_version"],
    }
