from interp_experiment.generation.answers import build_deterministic_answer_prompt, clean_deterministic_answer
from interp_experiment.schemas import ExampleRow


def test_clean_deterministic_answer_strips_label_and_duplicates() -> None:
    raw = "Answer: Yes, the buyer may terminate. Yes, the buyer may terminate. Delaware law governs."
    cleaned = clean_deterministic_answer(raw, max_sentences=3)
    assert cleaned == "Yes, the buyer may terminate. Delaware law governs."


def test_build_deterministic_answer_prompt_contains_question_and_excerpt() -> None:
    example = ExampleRow(
        example_id="ex-1",
        source_corpus="maud",
        contract_id="c-1",
        contract_group="merger_agreement",
        excerpt_text="Clause text",
        question_text="Can buyer terminate?",
        public_seed_answer="Seed",
        llama_answer_text="",
        split="train",
        cross_dist_group="maud_merger",
    ).validate()
    prompt = build_deterministic_answer_prompt(example)
    assert "Clause text" in prompt
    assert "Can buyer terminate?" in prompt
