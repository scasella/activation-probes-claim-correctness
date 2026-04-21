from __future__ import annotations


def paraphrase_prompt(question_text: str, paraphrase_index: int) -> str:
    return (
        f"Rewrite the legal question below without changing meaning. "
        f"Produce paraphrase #{paraphrase_index + 1}.\n\n{question_text}"
    )
