from interp_experiment.data.annotation import (
    compute_annotation_agreement,
    is_completed_annotation_row,
    validate_annotation_row,
)


def _row(annotator_id: str, claim_id: str, *, correctness: str = "", load: str = "", flip: str = "") -> dict[str, str]:
    return {
        "annotator_id": annotator_id,
        "annotation_version": "v1",
        "example_id": "ex-1",
        "source_corpus": "maud",
        "task_family": "generative_qa",
        "contract_id": "contract-1",
        "question_text": "Question?",
        "excerpt_text": "Excerpt text",
        "llama_answer_text": "Answer text",
        "claim_id": claim_id,
        "claim_text": "Claim text",
        "correctness_label": correctness,
        "load_bearing_label": load,
        "flip_evidence_text": flip,
        "notes": "",
    }


def test_validate_annotation_row_allows_blank_labels_in_packet_template() -> None:
    row = validate_annotation_row(_row("a1", "claim-1"))
    assert row["correctness_label"] == ""


def test_is_completed_annotation_row_requires_flip_evidence_for_load_bearing_yes() -> None:
    row = validate_annotation_row(_row("a1", "claim-1", correctness="true", load="yes", flip="Clause X"))
    assert is_completed_annotation_row(row) is True


def test_compute_annotation_agreement_reports_incomplete_when_labels_missing() -> None:
    rows = [
        validate_annotation_row(_row("a1", "claim-1")),
        validate_annotation_row(_row("a2", "claim-1")),
    ]
    payload = compute_annotation_agreement(rows)
    assert payload["gate"]["status"] == "incomplete"
    assert payload["n_complete_pairs"] == 0


def test_compute_annotation_agreement_reports_pass_for_perfect_agreement() -> None:
    rows = [
        validate_annotation_row(_row("a1", "claim-1", correctness="true", load="yes", flip="Clause X")),
        validate_annotation_row(_row("a2", "claim-1", correctness="true", load="yes", flip="Clause X")),
        validate_annotation_row(_row("a1", "claim-2", correctness="false", load="no")),
        validate_annotation_row(_row("a2", "claim-2", correctness="false", load="no")),
    ]
    payload = compute_annotation_agreement(rows)
    assert payload["gate"]["status"] == "pass"
    assert payload["agreement"]["load_bearing_kappa"] == 1.0
    assert payload["agreement"]["correctness_kappa"] == 1.0


def test_compute_annotation_agreement_extracts_disagreements() -> None:
    rows = [
        validate_annotation_row(_row("a1", "claim-1", correctness="true", load="yes", flip="Clause X")),
        validate_annotation_row(_row("a2", "claim-1", correctness="false", load="no")),
        validate_annotation_row(_row("a1", "claim-2", correctness="false", load="no")),
        validate_annotation_row(_row("a2", "claim-2", correctness="false", load="no")),
    ]
    payload = compute_annotation_agreement(rows)
    assert payload["gate"]["status"] == "fail_revise_once"
    assert payload["disagreements"]
    assert payload["disagreements"][0]["claim_id"] == "claim-1"
