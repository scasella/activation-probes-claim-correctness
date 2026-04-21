from __future__ import annotations

import math

from ..utils import tokenize_for_matching


def jaccard_similarity(left: str, right: str) -> float:
    left_tokens = tokenize_for_matching(left)
    right_tokens = tokenize_for_matching(right)
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    return overlap / union


def claim_presence_probability(canonical_claim: str, sampled_claims: list[str], threshold: float = 0.55) -> float:
    if not sampled_claims:
        return 0.0
    hits = sum(jaccard_similarity(canonical_claim, candidate) >= threshold for candidate in sampled_claims)
    return hits / len(sampled_claims)


def binary_entropy(probability: float) -> float:
    probability = min(max(probability, 1e-6), 1 - 1e-6)
    return -(probability * math.log(probability) + (1 - probability) * math.log(1 - probability))


def claim_resampling_entropy(canonical_claim: str, sampled_claims: list[str], threshold: float = 0.55) -> float:
    return binary_entropy(claim_presence_probability(canonical_claim, sampled_claims, threshold=threshold))
