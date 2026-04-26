from __future__ import annotations

import json

import numpy as np

from probemon.training import CanonicalClaim, CanonicalDataset, CanonicalExample, fit_probe, load_canonical_dataset
from probemon.training.adapters.factscore_adapter import convert_factscore_to_canonical
from probemon.training.adapters.maud_adapter import convert_maud_to_canonical


def test_fit_probe_on_precomputed_features(tmp_path) -> None:
    dataset = CanonicalDataset(
        name="toy",
        examples=[
            CanonicalExample("p", "a", [CanonicalClaim("a", 0, 1, "incorrect")], {"split": "train"}),
            CanonicalExample("p", "b", [CanonicalClaim("b", 0, 1, "correct")], {"split": "train"}),
            CanonicalExample("p", "c", [CanonicalClaim("c", 0, 1, "incorrect")], {"split": "validation"}),
            CanonicalExample("p", "d", [CanonicalClaim("d", 0, 1, "correct")], {"split": "validation"}),
        ],
    )
    features = np.asarray([[0.0, 0.0], [2.0, 2.0], [0.1, 0.0], [2.1, 2.0]], dtype=float)
    result = fit_probe(model=None, dataset=dataset, features=features, output_path=tmp_path / "toy.npz")
    assert result.validation_auroc == 1.0
    assert (tmp_path / "toy.npz").exists()


def test_load_canonical_dataset(tmp_path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text(
        json.dumps(
            {
                "prompt": "p",
                "generation": "hello",
                "claims": [{"text": "hello", "char_start": 0, "char_end": 5, "label": "correct"}],
                "metadata": {"split": "train"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = load_canonical_dataset(path)
    assert dataset.n_claims == 1


def test_factscore_adapter_outputs_parent_sentence_spans(tmp_path) -> None:
    examples = tmp_path / "examples.jsonl"
    claims = tmp_path / "claims.jsonl"
    examples.write_text(
        json.dumps(
            {
                "example_id": "e1",
                "question_text": "Tell me a bio.",
                "llama_answer_text": "Alice is a painter. She lives in Rome.",
                "split": "train",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    claims.write_text(
        json.dumps(
            {
                "claim_id": "c1",
                "example_id": "e1",
                "claim_text": "Alice is a painter.",
                "source_sentence_text": "Alice is a painter.",
                "source_sentence_char_start": 0,
                "source_sentence_char_end": 19,
                "raw_factscore_label": "S",
                "span_source": "parent_sentence",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = convert_factscore_to_canonical(examples_jsonl=examples, claims_jsonl=claims)
    assert dataset.examples[0].claims[0].label == "correct"
    assert dataset.examples[0].claims[0].char_end == 19


def test_maud_adapter_uses_fixed_claim_as_generation(tmp_path) -> None:
    examples = tmp_path / "examples.jsonl"
    claims = tmp_path / "claims.jsonl"
    labels = tmp_path / "labels.jsonl"
    examples.write_text(
        json.dumps({"example_id": "e1", "question_text": "Q", "excerpt_text": "E", "split": "train"}) + "\n",
        encoding="utf-8",
    )
    claims.write_text(
        json.dumps({"claim_id": "c1", "example_id": "e1", "claim_text": "Fixed claim."}) + "\n",
        encoding="utf-8",
    )
    labels.write_text(
        json.dumps({"claim_id": "c1", "correctness_label": "partially_true", "label_source": "judge"}) + "\n",
        encoding="utf-8",
    )
    dataset = convert_maud_to_canonical(examples_jsonl=examples, claims_jsonl=claims, labels_jsonl=labels)
    assert dataset.examples[0].generation == "Fixed claim."
    assert dataset.examples[0].claims[0].label == "partial"
