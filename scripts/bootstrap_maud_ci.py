from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from interp_experiment.evaluation.metrics import auroc
from interp_experiment.io import read_jsonl, write_json
from interp_experiment.schemas import BaselinePrediction, ClaimFeatureRow


METHOD_LABELS = {
    "llama_self_report": "Llama self-report",
    "gpt54_cached": "GPT-5.4 external scorer",
    "residual_probe": "Residual activation probe",
    "sae_probe": "SAE feature probe",
}


def _load_labels(path: Path) -> dict[str, dict[str, Any]]:
    return {row["claim_id"]: row for row in read_jsonl(path)}


def _load_claim_order(path: Path) -> list[str]:
    return [row["claim_id"] for row in read_jsonl(path)]


def _load_human_labels(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    rows = {}
    for row in read_jsonl(path):
        if row.get("correctness_label") in {"true", "partially_true", "false"}:
            rows[row["claim_id"]] = row
    return rows


def _load_predictions(path: Path) -> dict[str, BaselinePrediction]:
    rows: list[BaselinePrediction] = []
    if path.is_dir():
        for file_path in sorted(path.glob("*.jsonl")):
            rows.extend(BaselinePrediction.from_dict(row) for row in read_jsonl(file_path))
    else:
        rows.extend(BaselinePrediction.from_dict(row) for row in read_jsonl(path))
    return {row.claim_id: row for row in rows}


def _load_feature_arrays(path: Path) -> tuple[list[str], np.ndarray]:
    if path.suffix.lower() == ".npz":
        payload = np.load(path, allow_pickle=False)
        return [str(item) for item in payload["claim_ids"].tolist()], np.asarray(payload["matrix"], dtype=float)
    rows = [ClaimFeatureRow.from_dict(row) for row in read_jsonl(path)]
    return [row.claim_id for row in rows], np.asarray([row.vector for row in rows], dtype=float)


def _loo_probe_scores(feature_path: Path, fit_labels: dict[str, dict[str, Any]], candidate_claim_ids: set[str]) -> dict[str, float]:
    claim_ids, x_all = _load_feature_arrays(feature_path)
    shared_idx = [idx for idx, claim_id in enumerate(claim_ids) if claim_id in fit_labels and claim_id in candidate_claim_ids]
    shared_claim_ids = [claim_ids[idx] for idx in shared_idx]
    x = x_all[shared_idx]
    y_fit = np.asarray([1 if fit_labels[claim_id]["correctness_label"] == "true" else 0 for claim_id in shared_claim_ids], dtype=int)
    if len(set(y_fit.tolist())) < 2:
        raise ValueError("Probe fit labels contain a single class")
    probs = np.zeros(len(shared_claim_ids), dtype=float)
    for train_idx, test_idx in LeaveOneOut().split(x):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, solver="liblinear", max_iter=5000))
        model.fit(x[train_idx], y_fit[train_idx])
        probs[test_idx[0]] = model.predict_proba(x[test_idx])[0, 1]
    return {claim_id: float(prob) for claim_id, prob in zip(shared_claim_ids, probs)}


def _percentile_ci(values: list[float], alpha: float = 0.05) -> dict[str, float | int]:
    values = sorted(values)
    if not values:
        return {"ci_low": None, "ci_high": None, "n_valid_resamples": 0}
    low_idx = max(0, int((alpha / 2) * len(values)))
    high_idx = min(len(values) - 1, int((1 - alpha / 2) * len(values)) - 1)
    return {"ci_low": values[low_idx], "ci_high": values[high_idx], "n_valid_resamples": len(values)}


def _bootstrap_auroc(
    claim_ids: list[str],
    labels: dict[str, dict[str, Any]],
    scores: dict[str, float],
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    shared = [claim_id for claim_id in claim_ids if claim_id in labels and claim_id in scores]
    y = [1 if labels[claim_id]["correctness_label"] == "true" else 0 for claim_id in shared]
    s = [scores[claim_id] for claim_id in shared]
    point = auroc(y, s)
    rng = random.Random(seed)
    samples = []
    for _ in range(n_resamples):
        idxs = [rng.randrange(len(shared)) for _ in shared]
        yy = [y[i] for i in idxs]
        ss = [s[i] for i in idxs]
        try:
            samples.append(auroc(yy, ss))
        except ValueError:
            continue
    return {"n_claims": len(shared), "auroc": point, **_percentile_ci(samples)}


def _bootstrap_delta(
    claim_ids: list[str],
    labels: dict[str, dict[str, Any]],
    scores_a: dict[str, float],
    scores_b: dict[str, float],
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    shared = [claim_id for claim_id in claim_ids if claim_id in labels and claim_id in scores_a and claim_id in scores_b]
    y = [1 if labels[claim_id]["correctness_label"] == "true" else 0 for claim_id in shared]
    a = [scores_a[claim_id] for claim_id in shared]
    b = [scores_b[claim_id] for claim_id in shared]
    point = auroc(y, a) - auroc(y, b)
    rng = random.Random(seed)
    samples = []
    for _ in range(n_resamples):
        idxs = [rng.randrange(len(shared)) for _ in shared]
        yy = [y[i] for i in idxs]
        aa = [a[i] for i in idxs]
        bb = [b[i] for i in idxs]
        try:
            samples.append(auroc(yy, aa) - auroc(yy, bb))
        except ValueError:
            continue
    ci = _percentile_ci(samples)
    excludes_zero = None
    if ci["ci_low"] is not None and ci["ci_high"] is not None:
        excludes_zero = bool(ci["ci_low"] > 0 or ci["ci_high"] < 0)
    return {
        "n_claims": len(shared),
        "delta_auroc": point,
        **ci,
        "ci_excludes_zero": excludes_zero,
    }


def _score_maps(
    *,
    llama_path: Path,
    gpt54_path: Path,
    residual_path: Path,
    sae_path: Path,
    fit_labels: dict[str, dict[str, Any]],
    candidate_claim_ids: set[str],
) -> dict[str, dict[str, float]]:
    llama = _load_predictions(llama_path)
    gpt54 = _load_predictions(gpt54_path)
    return {
        "llama_self_report": {claim_id: row.correctness_confidence for claim_id, row in llama.items()},
        "gpt54_cached": {claim_id: row.correctness_confidence for claim_id, row in gpt54.items()},
        "residual_probe": _loo_probe_scores(residual_path, fit_labels, candidate_claim_ids),
        "sae_probe": _loo_probe_scores(sae_path, fit_labels, candidate_claim_ids),
    }


def _context_payload(
    *,
    name: str,
    claim_ids: list[str],
    labels: dict[str, dict[str, Any]],
    scores: dict[str, dict[str, float]],
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    aurocs = {
        method: _bootstrap_auroc(claim_ids, labels, method_scores, n_resamples=n_resamples, seed=seed + idx * 101)
        for idx, (method, method_scores) in enumerate(scores.items())
    }
    paired = {
        "residual_minus_sae": _bootstrap_delta(
            claim_ids,
            labels,
            scores["residual_probe"],
            scores["sae_probe"],
            n_resamples=n_resamples,
            seed=seed + 1001,
        ),
        "gpt54_minus_residual": _bootstrap_delta(
            claim_ids,
            labels,
            scores["gpt54_cached"],
            scores["residual_probe"],
            n_resamples=n_resamples,
            seed=seed + 1002,
        ),
        "residual_minus_self_report": _bootstrap_delta(
            claim_ids,
            labels,
            scores["residual_probe"],
            scores["llama_self_report"],
            n_resamples=n_resamples,
            seed=seed + 1003,
        ),
    }
    return {"name": name, "n_claims": len(claim_ids), "aurocs": aurocs, "paired_deltas": paired}


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap MAUD headline AUROC CIs and paired deltas.")
    parser.add_argument("--judge1-labels-jsonl", type=Path, default=Path("data/annotations/maud_full_judge_annotations.jsonl"))
    parser.add_argument("--judge2-labels-jsonl", type=Path, default=Path("data/annotations/maud_full_judge_annotations_v2.jsonl"))
    parser.add_argument("--human-labels-jsonl", type=Path, default=Path("data/annotations/maud_human_audit_labels.jsonl"))
    parser.add_argument("--llama-predictions", type=Path, default=Path("data/cached_baselines/llama_self_report/parsed/maud_full/_all_predictions.jsonl"))
    parser.add_argument("--gpt54-predictions", type=Path, default=Path("data/cached_baselines/gpt54/parsed/maud_full"))
    parser.add_argument("--residual-features-jsonl", type=Path, default=Path("artifacts/runs/maud_full_probe_features_residual.jsonl"))
    parser.add_argument("--sae-features-path", type=Path, default=Path("artifacts/runs/maud_full_probe_features_sae.npz"))
    parser.add_argument("--agreement-json", type=Path, default=Path("artifacts/runs/maud_judge_agreement_analysis.json"))
    parser.add_argument("--output-json", type=Path, default=Path("artifacts/runs/maud_bootstrap_ci.json"))
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=240424)
    args = parser.parse_args()

    judge1 = _load_labels(args.judge1_labels_jsonl)
    judge2 = _load_labels(args.judge2_labels_jsonl)
    judge1_order = _load_claim_order(args.judge1_labels_jsonl)
    agreement = json.loads(args.agreement_json.read_text(encoding="utf-8"))
    agreement_ids = list(agreement["agreement_claim_ids"])
    all_ids = [claim_id for claim_id in judge1_order if claim_id in judge2]
    all_ids_judge2_order = sorted(set(judge1) & set(judge2))
    human = _load_human_labels(args.human_labels_jsonl)
    human_ids = sorted(set(human) & set(judge1) & set(judge2))

    scores = _score_maps(
        llama_path=args.llama_predictions,
        gpt54_path=args.gpt54_predictions,
        residual_path=args.residual_features_jsonl,
        sae_path=args.sae_features_path,
        fit_labels=judge1,
        candidate_claim_ids=set(all_ids),
    )
    payload = {
        "method_labels": METHOD_LABELS,
        "bootstrap": {
            "n_resamples": args.n_resamples,
            "confidence_level": 0.95,
            "method": "paired claim-level bootstrap, percentile interval",
            "seed": args.seed,
        },
        "contexts": {
            "judge1_gpt54": _context_payload(
                name="judge1_gpt54",
                claim_ids=all_ids,
                labels=judge1,
                scores=scores,
                n_resamples=args.n_resamples,
                seed=args.seed,
            ),
            "judge2_kimi": _context_payload(
                name="judge2_kimi",
                claim_ids=all_ids_judge2_order,
                labels=judge2,
                scores=scores,
                n_resamples=args.n_resamples,
                seed=args.seed + 2000,
            ),
            "judge_agreement_set": _context_payload(
                name="judge_agreement_set",
                claim_ids=agreement_ids,
                labels=judge1,
                scores=scores,
                n_resamples=args.n_resamples,
                seed=args.seed + 4000,
            ),
        },
        "human_audit_subset": {
            "status": "pending_human_labels",
            "n_valid_human_labels": len(human_ids),
        },
    }
    if human_ids:
        payload["human_audit_subset"] = _context_payload(
            name="human_audit_subset",
            claim_ids=human_ids,
            labels=human,
            scores=scores,
            n_resamples=args.n_resamples,
            seed=args.seed + 6000,
        )
    write_json(args.output_json, payload)
    print(f"Wrote bootstrap CIs to {args.output_json}")


if __name__ == "__main__":
    main()
