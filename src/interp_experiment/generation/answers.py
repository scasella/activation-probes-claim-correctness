from __future__ import annotations

import re

from ..schemas import ExampleRow
from ..utils import normalize_whitespace

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
LEADING_LABEL_RE = re.compile(r"^(answer|response)\s*:\s*", re.IGNORECASE)
EXACT_SPAN_RE = re.compile(r"^[A-Z0-9][A-Za-z0-9 .,'\"()/-]{1,200}$")
SPECIAL_TOKEN_RE = re.compile(r"<\|[^>]+\|>")

CUAD_LABEL_GUIDANCE = {
    "Agreement Date": "the date the agreement was made or signed",
    "Anti-Assignment": "whether assignment is restricted, prohibited, or requires consent",
    "Audit Rights": "a right to inspect books, records, accounts, or compliance through an audit",
    "Document Name": "the title or formal name of the contract",
    "Effective Date": "the date the agreement becomes effective",
    "Ip Ownership Assignment": "ownership or assignment of intellectual property, inventions, or work product",
    "Minimum Commitment": "a minimum purchase, sales, volume, exclusivity, or effort commitment",
    "Parties": "the names of the parties entering into the agreement",
    "Post-Termination Services": "services, support, or transition obligations that continue after termination",
    "Rofr/Rofo/Rofn": "a right of first refusal, first offer, or first notice",
    "Warranty Duration": "the length or duration of a warranty period",
}


def build_deterministic_answer_prompt(example: ExampleRow) -> str:
    system_message = (
        "You are a careful legal contract analyst. "
        "Use only the provided excerpt. "
        "Do not invent facts. "
        "Do not use headings, bullets, or numbered lists."
    )
    if example.source_corpus == "cuad":
        guidance = CUAD_LABEL_GUIDANCE.get(example.question_text, "the contract concept named by the label")
        user_instruction = (
            f"The question names a CUAD contract field label meaning: {guidance}. "
            "Find the shortest exact excerpt span that directly answers that label. "
            "If the answer is absent, say 'Not stated in the excerpt.'"
        )
    else:
        user_instruction = (
            "Answer the legal question directly in 1 to 2 sentences. "
            "State the key condition once and stop."
        )
    user_message = (
        f"{user_instruction}\n\n"
        f"Contract excerpt:\n{example.excerpt_text}\n\n"
        f"Question:\n{example.question_text}"
    )
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )


def clean_deterministic_answer(answer_text: str, max_sentences: int = 3) -> str:
    text = answer_text.replace("\r\n", "\n").strip()
    text = SPECIAL_TOKEN_RE.sub(" ", text)
    text = LEADING_LABEL_RE.sub("", text)
    text = normalize_whitespace(text)
    if not text:
        return text

    if EXACT_SPAN_RE.match(text) and len(text) <= 220 and len(SENTENCE_RE.split(text)) == 1:
        return text

    sentences = [segment.strip() for segment in SENTENCE_RE.split(text) if segment.strip()]
    deduped: list[str] = []
    seen: set[str] = set()
    for sentence in sentences:
        normalized = sentence.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(sentence)
        if len(deduped) >= max_sentences:
            break

    cleaned = " ".join(deduped)
    return cleaned if cleaned else text
