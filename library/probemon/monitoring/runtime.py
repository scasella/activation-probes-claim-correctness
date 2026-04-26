from __future__ import annotations

import re
import warnings
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from probemon.core.activations import (
    GenerationWithActivations,
    HuggingFaceActivationExtractor,
    mean_pool_char_span,
)
from probemon.core.probe import Probe, load_probe
from probemon.core.segmentation import TextSpan, segment_sentences


@dataclass(frozen=True, slots=True)
class ClaimScore:
    text: str
    char_start: int
    char_end: int
    raw_score: float
    calibrated_score: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MonitoringResult:
    prompt: str
    generation: str
    probe_id: str
    model_name: str | None
    scores: list[ClaimScore]
    warnings: list[str]

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["scores"] = [score.as_dict() for score in self.scores]
        return payload


def _model_name(model: Any, extractor: Any | None = None) -> str | None:
    for obj in (extractor, model):
        if obj is None:
            continue
        if isinstance(obj, str):
            return obj
        for attr in ("model_name", "name_or_path"):
            value = getattr(obj, attr, None)
            if value:
                return str(value)
        config = getattr(obj, "config", None)
        value = getattr(config, "name_or_path", None)
        if value:
            return str(value)
    return None


def _extractor_for(model: Any, probe: Probe, extractor: Any | None) -> Any:
    if extractor is not None:
        return extractor
    if hasattr(model, "encode_answer_with_activations"):
        return model
    if isinstance(model, str):
        return HuggingFaceActivationExtractor(model, layer=probe.layer)
    raise TypeError(
        "score_generation requires either an activation extractor, a model with "
        "encode_answer_with_activations(prompt, generation), or a model name string."
    )


def _domain_term_fraction(prompt: str, terms: list[str]) -> float:
    if not terms:
        return 1.0
    words = set(re.findall(r"[a-z0-9][a-z0-9_-]+", prompt.lower()))
    if not words:
        return 0.0
    return sum(1 for term in terms if term.lower() in words) / len(terms)


def _ood_warnings(prompt: str, probe: Probe) -> list[str]:
    metadata = probe.metadata
    messages: list[str] = []
    mean = metadata.get("training_prompt_length_mean")
    std = metadata.get("training_prompt_length_std")
    if mean is not None and std:
        z = abs((len(prompt) - float(mean)) / max(float(std), 1.0))
        if z > 3.0:
            messages.append(
                "Prompt appears to differ from this probe's training distribution. Scores may not be reliable."
            )
    terms = list(metadata.get("domain_indicator_terms", []))
    if _domain_term_fraction(prompt, terms) < 0.1:
        warning = "Prompt appears to differ from this probe's training distribution. Scores may not be reliable."
        if warning not in messages:
            messages.append(warning)
    return messages


def _spans_from_claims(generation: str, claim_spans: list[dict[str, Any]] | list[TextSpan] | None) -> list[TextSpan]:
    if claim_spans is None:
        return segment_sentences(generation)
    spans: list[TextSpan] = []
    for item in claim_spans:
        if isinstance(item, TextSpan):
            spans.append(item)
        else:
            start = int(item["char_start"])
            end = int(item["char_end"])
            spans.append(TextSpan(text=str(item.get("text", generation[start:end])), char_start=start, char_end=end))
    return spans


def score_generation(
    *,
    model: Any,
    probe: Probe | str,
    prompt: str,
    generation: str,
    claim_spans: list[dict[str, Any]] | list[TextSpan] | None = None,
    activation_extractor: Any | None = None,
    suppress_ood_warning: bool = False,
) -> MonitoringResult:
    loaded_probe = load_probe(probe) if isinstance(probe, (str, bytes)) else probe
    extractor = _extractor_for(model, loaded_probe, activation_extractor)
    model_name = _model_name(model, extractor)
    loaded_probe.check_model_name(model_name)
    warnings_list = [] if suppress_ood_warning else _ood_warnings(prompt, loaded_probe)
    for message in warnings_list:
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    encoded: GenerationWithActivations = extractor.encode_answer_with_activations(prompt, generation)
    spans = _spans_from_claims(generation, claim_spans)
    vectors = np.vstack(
        [
            mean_pool_char_span(encoded.residual_stream, encoded.token_offsets, span.char_start, span.char_end)
            for span in spans
        ]
    )
    raw_logits = loaded_probe.raw_logits(vectors)
    scores = loaded_probe.score_vectors(vectors)
    claim_scores = [
        ClaimScore(
            text=span.text,
            char_start=span.char_start,
            char_end=span.char_end,
            raw_score=float(raw),
            calibrated_score=float(score),
        )
        for span, raw, score in zip(spans, raw_logits, scores, strict=True)
    ]
    return MonitoringResult(
        prompt=prompt,
        generation=generation,
        probe_id=loaded_probe.probe_id,
        model_name=model_name,
        scores=claim_scores,
        warnings=warnings_list,
    )


def generate_with_monitoring(
    *,
    model: Any,
    probe: Probe | str,
    prompt: str,
    max_new_tokens: int = 512,
    activation_extractor: Any | None = None,
    suppress_ood_warning: bool = False,
) -> MonitoringResult:
    loaded_probe = load_probe(probe) if isinstance(probe, (str, bytes)) else probe
    extractor = _extractor_for(model, loaded_probe, activation_extractor)
    if hasattr(extractor, "generate_text"):
        generation = extractor.generate_text(prompt, max_new_tokens=max_new_tokens)
    else:
        raise TypeError("generate_with_monitoring requires an extractor with generate_text().")
    return score_generation(
        model=model,
        probe=loaded_probe,
        prompt=prompt,
        generation=generation,
        activation_extractor=extractor,
        suppress_ood_warning=suppress_ood_warning,
    )
