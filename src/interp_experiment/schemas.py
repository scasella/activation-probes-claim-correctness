from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

LabelCorrectness = Literal["true", "false", "partially_true"]
LabelLoadBearing = Literal["yes", "no"]


class ValidationError(ValueError):
    pass


def _require_non_empty(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{field_name} must be a non-empty string")
    return value


def _require_probability(value: Any, field_name: str) -> float:
    if not isinstance(value, (float, int)):
        raise ValidationError(f"{field_name} must be numeric")
    numeric = float(value)
    if numeric < 0.0 or numeric > 1.0:
        raise ValidationError(f"{field_name} must be in [0, 1]")
    return numeric


@dataclass(slots=True)
class ExampleRow:
    example_id: str
    source_corpus: str
    contract_id: str
    contract_group: str
    excerpt_text: str
    question_text: str
    public_seed_answer: str
    llama_answer_text: str
    split: str
    cross_dist_group: str

    def validate(self) -> "ExampleRow":
        self.example_id = _require_non_empty(self.example_id, "example_id")
        self.source_corpus = _require_non_empty(self.source_corpus, "source_corpus")
        self.contract_id = _require_non_empty(self.contract_id, "contract_id")
        self.contract_group = _require_non_empty(self.contract_group, "contract_group")
        self.excerpt_text = _require_non_empty(self.excerpt_text, "excerpt_text")
        self.question_text = _require_non_empty(self.question_text, "question_text")
        self.public_seed_answer = _require_non_empty(self.public_seed_answer, "public_seed_answer")
        if not isinstance(self.llama_answer_text, str):
            raise ValidationError("llama_answer_text must be a string")
        self.cross_dist_group = _require_non_empty(self.cross_dist_group, "cross_dist_group")
        if self.split and self.split not in {"train", "validation", "test", "pilot", "unassigned"}:
            raise ValidationError("split must be one of train/validation/test/pilot/unassigned")
        return self

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExampleRow":
        return cls(**payload).validate()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ClaimRow:
    claim_id: str
    example_id: str
    claim_text: str
    token_start: int
    token_end: int
    correctness_label: LabelCorrectness | str
    load_bearing_label: LabelLoadBearing | str
    flip_evidence_text: str
    annotation_version: str

    def validate(self) -> "ClaimRow":
        self.claim_id = _require_non_empty(self.claim_id, "claim_id")
        self.example_id = _require_non_empty(self.example_id, "example_id")
        self.claim_text = _require_non_empty(self.claim_text, "claim_text")
        if not isinstance(self.token_start, int) or self.token_start < 0:
            raise ValidationError("token_start must be a non-negative int")
        if not isinstance(self.token_end, int) or self.token_end < self.token_start:
            raise ValidationError("token_end must be >= token_start")
        if self.correctness_label not in {"true", "false", "partially_true"}:
            raise ValidationError("correctness_label must be true/false/partially_true")
        if self.load_bearing_label not in {"yes", "no"}:
            raise ValidationError("load_bearing_label must be yes/no")
        if not isinstance(self.flip_evidence_text, str):
            raise ValidationError("flip_evidence_text must be a string")
        self.annotation_version = _require_non_empty(self.annotation_version, "annotation_version")
        return self

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ClaimRow":
        return cls(**payload).validate()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BaselinePrediction:
    claim_id: str
    correctness_confidence: float
    load_bearing_label: LabelLoadBearing | str
    load_bearing_confidence: float
    flip_evidence_text: str
    raw_json: dict[str, Any]
    prompt_version: str
    model_name: str

    def validate(self) -> "BaselinePrediction":
        self.claim_id = _require_non_empty(self.claim_id, "claim_id")
        self.correctness_confidence = _require_probability(self.correctness_confidence, "correctness_confidence")
        if self.load_bearing_label not in {"yes", "no"}:
            raise ValidationError("load_bearing_label must be yes/no")
        self.load_bearing_confidence = _require_probability(
            self.load_bearing_confidence,
            "load_bearing_confidence",
        )
        if not isinstance(self.flip_evidence_text, str):
            raise ValidationError("flip_evidence_text must be a string")
        if not isinstance(self.raw_json, dict):
            raise ValidationError("raw_json must be a dict")
        self.prompt_version = _require_non_empty(self.prompt_version, "prompt_version")
        self.model_name = _require_non_empty(self.model_name, "model_name")
        return self

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BaselinePrediction":
        return cls(**payload).validate()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ClaimFeatureRow:
    claim_id: str
    example_id: str
    feature_source: str
    vector: list[float]
    correctness_target: float | None
    load_bearing_target: int | None
    stability_target: int | None

    def validate(self) -> "ClaimFeatureRow":
        self.claim_id = _require_non_empty(self.claim_id, "claim_id")
        self.example_id = _require_non_empty(self.example_id, "example_id")
        self.feature_source = _require_non_empty(self.feature_source, "feature_source")
        if not isinstance(self.vector, list) or not self.vector or not all(isinstance(x, (float, int)) for x in self.vector):
            raise ValidationError("vector must be a non-empty numeric list")
        self.vector = [float(x) for x in self.vector]
        for field_name in ("load_bearing_target", "stability_target"):
            value = getattr(self, field_name)
            if value is not None and value not in {0, 1}:
                raise ValidationError(f"{field_name} must be 0/1 or None")
        if self.correctness_target is not None and not isinstance(self.correctness_target, (float, int)):
            raise ValidationError("correctness_target must be numeric or None")
        if self.correctness_target is not None:
            self.correctness_target = float(self.correctness_target)
        return self

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ClaimFeatureRow":
        return cls(**payload).validate()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MetricSummary:
    metric_name: str
    point_estimate: float
    ci_low: float
    ci_high: float
    n_items: int

    def validate(self) -> "MetricSummary":
        self.metric_name = _require_non_empty(self.metric_name, "metric_name")
        for name in ("point_estimate", "ci_low", "ci_high"):
            value = getattr(self, name)
            if not isinstance(value, (float, int)):
                raise ValidationError(f"{name} must be numeric")
            setattr(self, name, float(value))
        if not isinstance(self.n_items, int) or self.n_items <= 0:
            raise ValidationError("n_items must be a positive int")
        return self

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
