from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

from probemon import ModelMismatchError, Probe, load_probe, score_generation
from probemon.core.activations import GenerationWithActivations


class DummyExtractor:
    model_name = "meta-llama/Llama-3.1-8B-Instruct"

    def encode_answer_with_activations(self, prompt: str, generation: str) -> GenerationWithActivations:
        return GenerationWithActivations(
            answer_text=generation,
            residual_stream=np.ones((4, 4096), dtype=float),
            token_offsets=[(0, 4), (4, 8), (8, 12), (12, len(generation))],
            model_name=self.model_name,
            layer=19,
        )


def test_pretrained_probe_loads() -> None:
    probe = load_probe("legal-qa-llama-3.1-8b-v1")
    assert probe.direction.shape == (4096,)
    assert probe.metadata["test_auroc"] >= 0.77


def test_model_mismatch_error() -> None:
    probe = load_probe("legal-qa-llama-3.1-8b-v1")
    with pytest.raises(ModelMismatchError):
        probe.check_model_name("other/model")


def test_score_generation_with_dummy_extractor() -> None:
    probe = load_probe("legal-qa-llama-3.1-8b-v1")
    result = score_generation(
        model=DummyExtractor.model_name,
        activation_extractor=DummyExtractor(),
        probe=probe,
        prompt="Agreement fiduciary duty superior proposal notice parent board.",
        generation="One claim is here. Another claim follows.",
        suppress_ood_warning=True,
    )
    assert len(result.scores) == 2
    assert all(0.0 <= score.calibrated_score <= 1.0 for score in result.scores)


def test_ood_warning_fires() -> None:
    probe = load_probe("legal-qa-llama-3.1-8b-v1")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        score_generation(
            model=DummyExtractor.model_name,
            activation_extractor=DummyExtractor(),
            probe=probe,
            prompt="recipe pasta tomato basil",
            generation="Cook the pasta. Add sauce.",
        )
    assert any("training distribution" in str(item.message) for item in caught)


def test_probe_artifact_round_trip(tmp_path) -> None:
    metadata = {
        "probe_id": "tiny",
        "model_name": "m",
        "layer": 19,
        "pooling": "mean",
        "training_dataset": "d",
        "training_dataset_size": 2,
        "training_label_source": "labels",
        "validation_auroc": 1.0,
        "test_auroc": 1.0,
        "version": "0.1.0",
        "domain_description": "tiny",
    }
    probe = Probe("tiny", np.ones(3), 0.0, 1.0, 0.0, metadata)
    path = tmp_path / "probe.npz"
    probe.save(path)
    loaded = load_probe(path)
    assert loaded.probe_id == "tiny"
    assert loaded.score_vectors(np.ones((1, 3))).shape == (1,)
