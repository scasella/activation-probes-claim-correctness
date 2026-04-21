from __future__ import annotations

from collections import Counter


def cohens_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    if len(labels_a) != len(labels_b):
        raise ValueError("Both annotator label lists must have the same length")
    if not labels_a:
        raise ValueError("Need at least one label")
    observed = sum(a == b for a, b in zip(labels_a, labels_b)) / len(labels_a)
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    categories = set(counts_a) | set(counts_b)
    expected = sum((counts_a[c] / len(labels_a)) * (counts_b[c] / len(labels_b)) for c in categories)
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def load_bearing_gate_passed(kappa: float, threshold: float = 0.6) -> bool:
    return kappa >= threshold
