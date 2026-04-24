from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from interp_experiment.evaluation.metrics import auroc, brier_score
from interp_experiment.io import read_jsonl, write_json
from interp_experiment.schemas import BaselinePrediction, ClaimFeatureRow


LABELS = ["true", "partially_true", "false"]


def _load_labels(path: Path) -> dict[str, dict[str, Any]]:
    return {row["claim_id"]: row for row in read_jsonl(path)}


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
        claim_ids = [str(item) for item in payload["claim_ids"].tolist()]
        matrix = np.asarray(payload["matrix"], dtype=float)
        return claim_ids, matrix
    rows = [ClaimFeatureRow.from_dict(row) for row in read_jsonl(path)]
    return [row.claim_id for row in rows], np.asarray([row.vector for row in rows], dtype=float)


def _safe_auroc(y_true: list[int], y_score: list[float]) -> float | None:
    try:
        return auroc(y_true, y_score)
    except ValueError:
        return None


def _metric_block(y_true: list[int], y_score: list[float]) -> dict[str, Any]:
    return {
        "auroc": _safe_auroc(y_true, y_score),
        "brier": brier_score(y_true, y_score) if y_true else None,
    }


def _score_baseline(
    predictions: dict[str, BaselinePrediction],
    labels: dict[str, dict[str, Any]],
    claim_ids: list[str] | None = None,
) -> dict[str, Any]:
    ids = claim_ids or sorted(labels)
    shared = [claim_id for claim_id in ids if claim_id in labels and claim_id in predictions]
    y = [1 if labels[claim_id]["correctness_label"] == "true" else 0 for claim_id in shared]
    scores = [predictions[claim_id].correctness_confidence for claim_id in shared]
    return {"n_claims": len(shared), "correctness": _metric_block(y, scores)}


def _loo_probe_scores(
    feature_path: Path,
    fit_labels: dict[str, dict[str, Any]],
    candidate_claim_ids: set[str],
) -> dict[str, float]:
    claim_ids, x_all = _load_feature_arrays(feature_path)
    shared_idx = [idx for idx, claim_id in enumerate(claim_ids) if claim_id in fit_labels and claim_id in candidate_claim_ids]
    shared_claim_ids = [claim_ids[idx] for idx in shared_idx]
    x = x_all[shared_idx]
    y_fit = np.asarray([1 if fit_labels[claim_id]["correctness_label"] == "true" else 0 for claim_id in shared_claim_ids], dtype=int)
    if len(set(y_fit.tolist())) < 2:
        raise ValueError("Probe fit labels contain a single class")
    probs = np.zeros(len(shared_claim_ids), dtype=float)
    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(x):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, solver="liblinear", max_iter=5000))
        model.fit(x[train_idx], y_fit[train_idx])
        probs[test_idx[0]] = model.predict_proba(x[test_idx])[0, 1]
    return {claim_id: float(prob) for claim_id, prob in zip(shared_claim_ids, probs)}


def _score_probe_scores(
    probe_scores: dict[str, float],
    labels: dict[str, dict[str, Any]],
    claim_ids: list[str] | None = None,
) -> dict[str, Any]:
    ids = claim_ids or sorted(labels)
    shared = [claim_id for claim_id in ids if claim_id in labels and claim_id in probe_scores]
    y = [1 if labels[claim_id]["correctness_label"] == "true" else 0 for claim_id in shared]
    scores = [probe_scores[claim_id] for claim_id in shared]
    return {"n_claims": len(shared), "correctness": _metric_block(y, scores)}


def _cohen_kappa(pairs: list[tuple[str, str]]) -> dict[str, float]:
    n = len(pairs)
    if n == 0:
        return {"kappa": float("nan"), "observed_agreement": float("nan"), "expected_agreement": float("nan")}
    observed = sum(1 for a, b in pairs if a == b) / n
    left = Counter(a for a, _ in pairs)
    right = Counter(b for _, b in pairs)
    expected = sum((left[label] / n) * (right[label] / n) for label in LABELS)
    if expected == 1.0:
        kappa = 1.0 if observed == 1.0 else float("nan")
    else:
        kappa = (observed - expected) / (1.0 - expected)
    return {"kappa": kappa, "observed_agreement": observed, "expected_agreement": expected}


def _judge_agreement(v1_labels: dict[str, dict[str, Any]], v2_labels: dict[str, dict[str, Any]]) -> dict[str, Any]:
    shared_ids = sorted(set(v1_labels) & set(v2_labels))
    pairs = [(v1_labels[claim_id]["correctness_label"], v2_labels[claim_id]["correctness_label"]) for claim_id in shared_ids]
    confusion = {
        left: {right: 0 for right in LABELS}
        for left in LABELS
    }
    for left, right in pairs:
        confusion[left][right] += 1
    by_v1_label = {}
    for label in LABELS:
        label_pairs = [(a, b) for a, b in pairs if a == label]
        by_v1_label[label] = {
            "n_claims": len(label_pairs),
            "agreement_rate": (sum(1 for a, b in label_pairs if a == b) / len(label_pairs)) if label_pairs else None,
        }
    agreement_ids = [claim_id for claim_id in shared_ids if v1_labels[claim_id]["correctness_label"] == v2_labels[claim_id]["correctness_label"]]
    kappa = _cohen_kappa(pairs)
    return {
        "n_claims_judge1": len(v1_labels),
        "n_claims_judge2": len(v2_labels),
        "n_shared_claims": len(shared_ids),
        "n_missing_from_judge2": len(set(v1_labels) - set(v2_labels)),
        "raw_agreement_rate": kappa["observed_agreement"],
        "cohen_kappa": kappa["kappa"],
        "expected_agreement": kappa["expected_agreement"],
        "confusion_matrix": confusion,
        "agreement_by_judge1_label": by_v1_label,
        "agreement_claim_ids": agreement_ids,
        "judge1_label_counts": dict(Counter(v1_labels[claim_id]["correctness_label"] for claim_id in shared_ids)),
        "judge2_label_counts": dict(Counter(v2_labels[claim_id]["correctness_label"] for claim_id in shared_ids)),
    }


def _ordering(metrics: dict[str, dict[str, Any]]) -> list[str]:
    return [
        item[0]
        for item in sorted(
            metrics.items(),
            key=lambda kv: (-1 if kv[1]["correctness"]["auroc"] is None else kv[1]["correctness"]["auroc"]),
            reverse=True,
        )
    ]


def _outcome(
    *,
    judge1: dict[str, dict[str, Any]],
    judge2: dict[str, dict[str, Any]],
    agreement: dict[str, dict[str, Any]],
    kappa: float,
) -> dict[str, str]:
    order1 = _ordering(judge1)
    order2 = _ordering(judge2)
    order_agreement = _ordering(agreement)
    gpt_drop = (judge1["gpt54_cached"]["correctness"]["auroc"] or 0) - (judge2["gpt54_cached"]["correctness"]["auroc"] or 0)
    if kappa < 0.4:
        return {
            "outcome": "D",
            "label": "Judges broadly disagree",
            "rationale": "Cohen's kappa is below 0.4, so portability of the judge-proxy target is the main result.",
        }
    if order1 == order2 == order_agreement:
        return {
            "outcome": "A",
            "label": "Story holds",
            "rationale": "The AUROC ordering is stable across judge 1, judge 2, and the agreement set.",
        }
    if gpt_drop >= 0.15:
        return {
            "outcome": "B",
            "label": "GPT-5.4 external scorer drops under the second judge",
            "rationale": "The GPT-5.4 scorer's AUROC drops by at least 0.15 under judge 2, consistent with same-family coupling.",
        }
    return {
        "outcome": "C",
        "label": "Self-report or probes shift meaningfully",
        "rationale": "Judge 2 changes the method ordering or probe/self-report interpretation without collapsing judge agreement below 0.4.",
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _write_report(
    path: Path,
    *,
    second_judge_model: str,
    model_choice_path: Path,
    judge1_metrics: dict[str, dict[str, Any]],
    judge2_metrics: dict[str, dict[str, Any]],
    agreement_metrics: dict[str, dict[str, Any]],
    agreement_analysis: dict[str, Any],
    outcome: dict[str, str],
) -> None:
    method_labels = {
        "llama_self_report": "Llama self-report",
        "gpt54_cached": "GPT-5.4 external scorer",
        "residual_probe": "Residual activation probe",
        "sae_probe": "SAE feature probe",
    }
    lines = [
        "# Second-Judge Sensitivity Study: MAUD Judge-Proxy Results",
        "",
        "## Header numbers fixed before interpretation",
        "",
        "| Method | AUROC under judge 1 (GPT-5.4) | AUROC under judge 2 | AUROC on judge-agreement set |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in method_labels:
        lines.append(
            f"| {method_labels[key]} | "
            f"{_fmt(judge1_metrics[key]['correctness']['auroc'])} | "
            f"{_fmt(judge2_metrics[key]['correctness']['auroc'])} | "
            f"{_fmt(agreement_metrics[key]['correctness']['auroc'])} |"
        )
    lines.extend(
        [
            "",
            "## Second-judge model choice",
            "",
            f"The second judge is `{second_judge_model}` via Prime Intellect Inference. It was selected because Kimi K2.6 was available at runtime, it is from the Moonshot/Kimi family rather than the GPT family, and it is the highest-priority available model under the requested Kimi -> Qwen -> GLM selection order.",
            "",
            f"The recorded model-choice note is `{model_choice_path}`.",
            "",
            "## Judge agreement",
            "",
            f"- Shared claims: {agreement_analysis['n_shared_claims']}",
            f"- Missing v2 labels: {agreement_analysis['n_missing_from_judge2']}",
            f"- Raw agreement: {_fmt(agreement_analysis['raw_agreement_rate'])}",
            f"- Cohen's kappa: {_fmt(agreement_analysis['cohen_kappa'])}",
            "",
            "| Judge 1 label | Judge 2 true | Judge 2 partially_true | Judge 2 false |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for label in LABELS:
        row = agreement_analysis["confusion_matrix"][label]
        lines.append(f"| {label} | {row['true']} | {row['partially_true']} | {row['false']} |")
    lines.extend(
        [
            "",
            "Agreement by GPT-5.4 judge label:",
            "",
        ]
    )
    for label in LABELS:
        block = agreement_analysis["agreement_by_judge1_label"][label]
        lines.append(f"- `{label}`: n={block['n_claims']}, agreement={_fmt(block['agreement_rate'])}")
    lines.extend(
        [
            "",
            "## Outcome call",
            "",
            f"Outcome {outcome['outcome']} - {outcome['label']}. {outcome['rationale']}",
            "",
            "## What this result does and does not show",
            "",
            "This sensitivity pass tests whether the first MAUD judge-proxy story survives a change in LLM judge family. It does not turn either judge into legal ground truth, and it does not validate the labels against human legal judgment. Stable ordering across two LLM judges would make the proxy result more credible; broad judge disagreement would instead show that the target label itself is unstable.",
            "",
            "The probe numbers should be read as fixed-artifact sensitivity scores: the probe scoring path is fit only from judge-1 labels and then compared with judge-2 labels or the judge-agreement subset. Judge-2 labels are not used to retrain the probes.",
            "",
            "To move from stable across two LLM judges to stable against human legal judgment, the next step is a small source-bound human audit: sample claims from agreement and disagreement regions, have legal annotators score the frozen claim list against the same excerpt/question/answer bundle, and report where both LLM judges agree or diverge from the human labels.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze second-judge sensitivity for the frozen MAUD proxy study.")
    parser.add_argument("--judge1-labels-jsonl", type=Path, default=Path("data/annotations/maud_full_judge_annotations.jsonl"))
    parser.add_argument("--judge2-labels-jsonl", type=Path, default=Path("data/annotations/maud_full_judge_annotations_v2.jsonl"))
    parser.add_argument("--llama-predictions", type=Path, default=Path("data/cached_baselines/llama_self_report/parsed/maud_full/_all_predictions.jsonl"))
    parser.add_argument("--gpt54-predictions", type=Path, default=Path("data/cached_baselines/gpt54/parsed/maud_full"))
    parser.add_argument("--residual-features-jsonl", type=Path, default=Path("artifacts/runs/maud_full_probe_features_residual.jsonl"))
    parser.add_argument("--sae-features-path", type=Path, default=Path("artifacts/runs/maud_full_probe_features_sae.npz"))
    parser.add_argument("--judge1-baseline-json", type=Path, default=Path("artifacts/runs/maud_full_proxy_baseline_eval.json"))
    parser.add_argument("--judge1-probe-json", type=Path, default=Path("artifacts/runs/maud_full_probe_proxy_smoke.json"))
    parser.add_argument("--judge2-baseline-json", type=Path, default=Path("artifacts/runs/maud_full_proxy_baseline_eval_v2.json"))
    parser.add_argument("--judge2-probe-json", type=Path, default=Path("artifacts/runs/maud_full_probe_proxy_smoke_v2.json"))
    parser.add_argument("--agreement-json", type=Path, default=Path("artifacts/runs/maud_judge_agreement_analysis.json"))
    parser.add_argument("--agreement-eval-json", type=Path, default=Path("artifacts/runs/maud_agreement_set_eval.json"))
    parser.add_argument("--report-md", type=Path, default=Path("docs/second_judge_findings.md"))
    parser.add_argument("--model-choice-md", type=Path, default=Path("docs/second_judge_model_choice.md"))
    parser.add_argument("--second-judge-model", default="moonshotai/kimi-k2.6")
    args = parser.parse_args()

    v1_labels = _load_labels(args.judge1_labels_jsonl)
    v2_labels = _load_labels(args.judge2_labels_jsonl)
    llama = _load_predictions(args.llama_predictions)
    gpt54 = _load_predictions(args.gpt54_predictions)
    candidate_claim_ids = set(v1_labels) & set(v2_labels)
    residual_scores = _loo_probe_scores(args.residual_features_jsonl, v1_labels, candidate_claim_ids)
    sae_scores = _loo_probe_scores(args.sae_features_path, v1_labels, candidate_claim_ids)

    judge1_existing_baseline = json.loads(args.judge1_baseline_json.read_text(encoding="utf-8"))
    judge1_existing_probe = json.loads(args.judge1_probe_json.read_text(encoding="utf-8"))
    judge1_metrics = {
        "llama_self_report": judge1_existing_baseline["llama_self_report"],
        "gpt54_cached": judge1_existing_baseline["gpt54_cached"],
        "residual_probe": judge1_existing_probe["residual"],
        "sae_probe": judge1_existing_probe["sae"],
    }
    judge2_metrics = {
        "llama_self_report": _score_baseline(llama, v2_labels),
        "gpt54_cached": _score_baseline(gpt54, v2_labels),
        "residual_probe": _score_probe_scores(residual_scores, v2_labels),
        "sae_probe": _score_probe_scores(sae_scores, v2_labels),
    }
    agreement_analysis = _judge_agreement(v1_labels, v2_labels)
    agreement_claim_ids = agreement_analysis["agreement_claim_ids"]
    agreement_metrics = {
        "llama_self_report": _score_baseline(llama, v1_labels, agreement_claim_ids),
        "gpt54_cached": _score_baseline(gpt54, v1_labels, agreement_claim_ids),
        "residual_probe": _score_probe_scores(residual_scores, v1_labels, agreement_claim_ids),
        "sae_probe": _score_probe_scores(sae_scores, v1_labels, agreement_claim_ids),
    }
    outcome = _outcome(
        judge1=judge1_metrics,
        judge2=judge2_metrics,
        agreement=agreement_metrics,
        kappa=float(agreement_analysis["cohen_kappa"]),
    )

    write_json(args.agreement_json, agreement_analysis)
    write_json(
        args.agreement_eval_json,
        {
            "label_source": "judge_agreement_set",
            "n_claims": len(agreement_claim_ids),
            "metrics": agreement_metrics,
            "method_ordering": _ordering(agreement_metrics),
            "judge1_full_ordering": _ordering(judge1_metrics),
            "judge2_full_ordering": _ordering(judge2_metrics),
            "matches_judge1_full_ordering": _ordering(agreement_metrics) == _ordering(judge1_metrics),
            "matches_judge2_full_ordering": _ordering(agreement_metrics) == _ordering(judge2_metrics),
            "probe_fit_label_source": "judge_llm_proxy_v1",
        },
    )
    write_json(
        args.judge2_baseline_json,
        {
            "label_source": "judge_llm_proxy_v2",
            "llama_self_report": judge2_metrics["llama_self_report"],
            "gpt54_cached": judge2_metrics["gpt54_cached"],
        },
    )
    write_json(
        args.judge2_probe_json,
        {
            "label_source": "judge_llm_proxy_v2",
            "fit_label_source": "judge_llm_proxy_v1",
            "proxy_only": True,
            "residual": judge2_metrics["residual_probe"],
            "sae": judge2_metrics["sae_probe"],
            "note": "leave_one_out_probe_scores_fit_on_judge1_labels_scored_against_judge2_labels",
        },
    )
    _write_report(
        args.report_md,
        second_judge_model=args.second_judge_model,
        model_choice_path=args.model_choice_md,
        judge1_metrics=judge1_metrics,
        judge2_metrics=judge2_metrics,
        agreement_metrics=agreement_metrics,
        agreement_analysis=agreement_analysis,
        outcome=outcome,
    )
    print(f"Wrote judge agreement analysis to {args.agreement_json}")
    print(f"Wrote agreement-set eval to {args.agreement_eval_json}")
    print(f"Wrote sensitivity report to {args.report_md}")


if __name__ == "__main__":
    main()
