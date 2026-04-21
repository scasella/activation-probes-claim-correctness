from interp_experiment.data.pilot import build_claim_annotation_packet, sample_pilot_examples
from interp_experiment.schemas import ClaimRow, ExampleRow


def _example(example_id: str, source: str) -> ExampleRow:
    return ExampleRow(
        example_id=example_id,
        source_corpus=source,
        contract_id=f"{source}-{example_id}",
        contract_group="group",
        excerpt_text="Excerpt",
        question_text="Question?",
        public_seed_answer="Seed",
        llama_answer_text="Answer.",
        split="train",
        cross_dist_group=f"{source}_group",
    ).validate()


def test_sample_pilot_examples_balances_two_sources() -> None:
    examples = [_example(f"m{i}", "maud") for i in range(20)] + [_example(f"c{i}", "cuad") for i in range(20)]
    sampled = sample_pilot_examples(examples, pilot_size=10, stratify_field="source_corpus", seed=5)
    counts = {}
    for item in sampled:
        counts[item.source_corpus] = counts.get(item.source_corpus, 0) + 1
    assert counts == {"cuad": 5, "maud": 5}


def test_build_claim_annotation_packet_multiplies_by_annotators() -> None:
    examples = [_example("ex-1", "maud")]
    claims = [
        ClaimRow(
            claim_id="claim-1",
            example_id="ex-1",
            claim_text="Claim text",
            token_start=0,
            token_end=1,
            correctness_label="false",
            load_bearing_label="no",
            flip_evidence_text="",
            annotation_version="v1",
        ).validate()
    ]
    packet = build_claim_annotation_packet(examples, claims, annotator_ids=["a1", "a2"])
    assert len(packet) == 2
    assert {row["annotator_id"] for row in packet} == {"a1", "a2"}
