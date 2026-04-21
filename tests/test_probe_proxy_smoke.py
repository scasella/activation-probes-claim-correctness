import json

from interp_experiment.io import write_jsonl


def test_label_loading_contract_for_proxy_smoke(tmp_path) -> None:
    rows = [
        {
            "annotator_id": "judge_gpt54",
            "annotation_version": "proxy_v1",
            "example_id": "ex-1",
            "source_corpus": "maud",
            "task_family": "generative_qa",
            "contract_id": "contract-1",
            "question_text": "Question?",
            "excerpt_text": "Excerpt",
            "llama_answer_text": "Answer",
            "claim_id": "claim-1",
            "claim_text": "Claim",
            "correctness_label": "true",
            "load_bearing_label": "yes",
            "flip_evidence_text": "Clause X",
            "notes": "",
        }
    ]
    path = tmp_path / "labels.jsonl"
    write_jsonl(path, rows)
    loaded = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert loaded[0]["claim_id"] == "claim-1"
