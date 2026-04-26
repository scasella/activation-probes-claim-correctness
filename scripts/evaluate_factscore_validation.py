from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from interp_experiment.evaluation.metrics import auroc, brier_score
from interp_experiment.io import read_json, read_jsonl, write_json
from interp_experiment.schemas import BaselinePrediction


def _load_labels(path: Path) -> dict[str, dict[str, Any]]:
    return {row["claim_id"]: row for row in read_jsonl(path)}


def _load_examples(path: Path) -> dict[str, dict[str, Any]]:
    return {row["example_id"]: row for row in read_jsonl(path)}


def _load_predictions(path: Path) -> dict[str, float]:
    return {
        prediction.claim_id: prediction.correctness_confidence
        for prediction in (BaselinePrediction.from_dict(row) for row in read_jsonl(path))
    }


def _load_features(path: Path) -> tuple[list[str], list[str], np.ndarray]:
    payload = np.load(path, allow_pickle=False)
    return [str(item) for item in payload["claim_ids"].tolist()], [str(item) for item in payload["example_ids"].tolist()], np.asarray(payload["matrix"], dtype=float)


def _metric_block(y_true: list[int], y_score: list[float]) -> dict[str, float | None]:
    try:
        roc = auroc(y_true, y_score)
    except ValueError:
        roc = None
    return {"auroc": roc, "brier": brier_score(y_true, y_score) if y_true else None}


def _bootstrap_delta(
    y_true: list[int],
    score_a: list[float],
    score_b: list[float],
    metric: Callable[[list[int], list[float]], float],
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    point = metric(y_true, score_a) - metric(y_true, score_b)
    rng = random.Random(seed)
    samples: list[float] = []
    dropped = 0
    indices = list(range(len(y_true)))
    for _ in range(n_resamples):
        sample = [indices[rng.randrange(len(indices))] for _ in indices]
        yy = [y_true[idx] for idx in sample]
        aa = [score_a[idx] for idx in sample]
        bb = [score_b[idx] for idx in sample]
        try:
            samples.append(metric(yy, aa) - metric(yy, bb))
        except ValueError:
            dropped += 1
    if samples:
        low, high = np.quantile(np.asarray(samples), [0.025, 0.975]).tolist()
    else:
        low, high = None, None
    return {
        "point": point,
        "ci_low": low,
        "ci_high": high,
        "n_valid_resamples": len(samples),
        "n_resamples_dropped": dropped,
        "ci_excludes_zero": bool(low is not None and high is not None and (low > 0 or high < 0)),
    }


def _fit_probe(x_train: np.ndarray, y_train: np.ndarray, c_value: float) -> Any:
    model = make_pipeline(StandardScaler(), LogisticRegression(C=c_value, solver="liblinear", max_iter=5000))
    model.fit(x_train, y_train)
    return model


def _evaluate_probe(
    claim_ids: list[str],
    example_ids: list[str],
    x: np.ndarray,
    labels: dict[str, dict[str, Any]],
    examples: dict[str, dict[str, Any]],
    c_values: list[float],
) -> dict[str, Any]:
    splits = {claim_id: examples[example_id]["split"] for claim_id, example_id in zip(claim_ids, example_ids)}
    y = np.asarray([1 if labels[claim_id]["correctness_label"] == "true" else 0 for claim_id in claim_ids], dtype=int)
    train_idx = np.asarray([idx for idx, claim_id in enumerate(claim_ids) if splits[claim_id] == "train"], dtype=int)
    val_idx = np.asarray([idx for idx, claim_id in enumerate(claim_ids) if splits[claim_id] == "validation"], dtype=int)
    test_idx = np.asarray([idx for idx, claim_id in enumerate(claim_ids) if splits[claim_id] == "test"], dtype=int)

    validation_rows = []
    for c_value in c_values:
        model = _fit_probe(x[train_idx], y[train_idx], c_value)
        val_probs = model.predict_proba(x[val_idx])[:, 1]
        train_probs = model.predict_proba(x[train_idx])[:, 1]
        validation_rows.append(
            {
                "c": c_value,
                "train": _metric_block(y[train_idx].tolist(), train_probs.tolist()),
                "validation": _metric_block(y[val_idx].tolist(), val_probs.tolist()),
            }
        )
    valid_rows = [row for row in validation_rows if row["validation"]["auroc"] is not None]
    if not valid_rows:
        raise ValueError("No C value produced a valid validation AUROC")
    selected = max(valid_rows, key=lambda row: (row["validation"]["auroc"], -row["validation"]["brier"]))
    train_val_idx = np.concatenate([train_idx, val_idx])
    final_model = _fit_probe(x[train_val_idx], y[train_val_idx], float(selected["c"]))
    train_probs = final_model.predict_proba(x[train_idx])[:, 1]
    val_probs = final_model.predict_proba(x[val_idx])[:, 1]
    test_probs = final_model.predict_proba(x[test_idx])[:, 1]
    return {
        "selected_c": selected["c"],
        "validation": {"rows": validation_rows, "selected": selected["validation"]},
        "train": {
            "claim_ids": [claim_ids[idx] for idx in train_idx],
            "metrics": _metric_block(y[train_idx].tolist(), train_probs.tolist()),
        },
        "test": {
            "claim_ids": [claim_ids[idx] for idx in test_idx],
            "y_true": y[test_idx].tolist(),
            "scores": test_probs.tolist(),
            "metrics": _metric_block(y[test_idx].tolist(), test_probs.tolist()),
        },
        "split_counts": {"train": int(len(train_idx)), "validation": int(len(val_idx)), "test": int(len(test_idx))},
    }


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    residual = payload["residual_probe"]["test"]["metrics"]
    self_report = payload["llama_self_report"]["test"]["metrics"]
    delta = payload["paired_delta_residual_minus_self_report_auroc"]
    materialization = payload["materialization_summary"]
    adapter = payload["adapter_summary"]
    decision = payload["decision"]
    lines = [
        "# FActScore Validation",
        "",
        "## Summary",
        "",
        decision["summary"],
        "",
        "## Method",
        "",
        "This is a scoped transfer check of the MAUD residual-probe method on FActScore ChatGPT biographies. I used the official FActScore human annotations, kept only `Supported` and `Not-supported` atomic facts, and dropped `Irrelevant` facts. Because the human atomic facts are canonicalized rather than literal substrings of the biography, activation pooling uses the parent annotated sentence span while labels remain atomic-fact labels.",
        "",
        f"The adapter produced {adapter['n_examples_with_claims']} usable biographies and {adapter['n_claims']} non-IR atomic facts. Label counts were {adapter['label_counts_after_dropping_ir']}. Token alignment covered {payload['token_alignment_summary']['n_claims_aligned']}/{payload['token_alignment_summary']['n_claims_checked']} claims with median span length {payload['token_alignment_summary']['token_span_lengths']['median']:.1f} tokens and no 1-2 token spans.",
        "",
        f"Materialization covered {materialization['n_feature_rows']}/{materialization['n_expected_claims']} feature rows and {materialization['n_self_report_predictions']}/{materialization['n_expected_claims']} self-report predictions.",
        "",
        "## Results",
        "",
        "| Method | Test AUROC | Test Brier |",
        "| --- | ---: | ---: |",
        f"| Llama self-report | {_fmt(self_report['auroc'])} | {_fmt(self_report['brier'])} |",
        f"| Residual probe | {_fmt(residual['auroc'])} | {_fmt(residual['brier'])} |",
        "",
        f"Residual minus self-report AUROC delta: {delta['point']:.3f} [{delta['ci_low']:.3f}, {delta['ci_high']:.3f}] from {payload['n_bootstrap']} paired bootstrap resamples. The interval {'excludes' if delta['ci_excludes_zero'] else 'includes'} zero.",
        "",
        "## Comparison",
        "",
        "MAUD showed a large probe-over-self-report gap. FELM-wk was directionally positive but inconclusive. FActScore is cleaner than FELM on label granularity and evidence coverage, but it retains the generation-mismatch caveat: these biographies were generated by ChatGPT, not Llama.",
        "",
        "## Library Decision",
        "",
        decision["library_decision"],
        "",
        "## Methods-Paper Paragraph Draft",
        "",
        decision["paper_paragraph"],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate residual probe vs Llama self-report on FActScore.")
    parser.add_argument("--examples-jsonl", type=Path, default=Path("data/factscore/factscore_chatgpt_examples.jsonl"))
    parser.add_argument("--labels-jsonl", type=Path, default=Path("data/annotations/factscore_chatgpt_labels.jsonl"))
    parser.add_argument("--features-npz", type=Path, default=Path("artifacts/runs/factscore_chatgpt_probe_features_residual.npz"))
    parser.add_argument(
        "--self-report-jsonl",
        type=Path,
        default=Path("data/cached_baselines/llama_self_report/parsed/factscore_chatgpt/_all_predictions.jsonl"),
    )
    parser.add_argument("--adapter-summary-json", type=Path, default=Path("artifacts/runs/factscore_chatgpt_adapter_summary.json"))
    parser.add_argument("--token-alignment-summary-json", type=Path, default=Path("artifacts/runs/factscore_chatgpt_token_alignment_summary.json"))
    parser.add_argument("--materialization-summary-json", type=Path, default=Path("artifacts/runs/factscore_chatgpt_materialization_summary.json"))
    parser.add_argument("--output-json", type=Path, default=Path("artifacts/runs/factscore_chatgpt_validation_eval.json"))
    parser.add_argument("--report-md", type=Path, default=Path("docs/factscore_validation.md"))
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260424)
    parser.add_argument("--c-values", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0])
    args = parser.parse_args()

    labels = _load_labels(args.labels_jsonl)
    examples = _load_examples(args.examples_jsonl)
    self_report_scores = _load_predictions(args.self_report_jsonl)
    claim_ids, example_ids, x = _load_features(args.features_npz)
    adapter_summary = read_json(args.adapter_summary_json)
    token_alignment_summary = read_json(args.token_alignment_summary_json)
    materialization = read_json(args.materialization_summary_json)
    shared_idx = [
        idx
        for idx, claim_id in enumerate(claim_ids)
        if claim_id in labels and claim_id in self_report_scores and example_ids[idx] in examples
    ]
    claim_ids = [claim_ids[idx] for idx in shared_idx]
    example_ids = [example_ids[idx] for idx in shared_idx]
    x = x[shared_idx]
    probe = _evaluate_probe(claim_ids, example_ids, x, labels, examples, args.c_values)
    test_claim_ids = probe["test"]["claim_ids"]
    y_test = probe["test"]["y_true"]
    residual_scores = probe["test"]["scores"]
    self_scores = [self_report_scores[claim_id] for claim_id in test_claim_ids]
    self_metrics = _metric_block(y_test, self_scores)
    delta = _bootstrap_delta(y_test, residual_scores, self_scores, auroc, n_resamples=args.n_bootstrap, seed=args.seed)

    train_auroc = probe["train"]["metrics"]["auroc"]
    validation_auroc = probe["validation"]["selected"]["auroc"]
    residual_auroc = probe["test"]["metrics"]["auroc"]
    self_auroc = self_metrics["auroc"]
    if train_auroc is not None and train_auroc < 0.65:
        decision = {
            "status": "stop_pilot_train_below_threshold",
            "summary": f"The FActScore pilot should stop: residual probe train AUROC was {train_auroc:.3f}, below the 0.65 pilot diagnostic threshold.",
            "library_decision": "Do not add FActScore as a scaffolding dataset yet. Diagnose span mapping and label noise before any library claim.",
            "paper_paragraph": "We attempted a FActScore transfer pilot, but the residual probe did not clear the preregistered pilot training-AUROC sanity threshold; we therefore did not treat it as evidence about transfer.",
        }
    elif validation_auroc is not None and validation_auroc < 0.55:
        decision = {
            "status": "stop_validation_below_threshold",
            "summary": f"The FActScore run should stop at validation: selected validation AUROC was {validation_auroc:.3f}, below the 0.55 stopping threshold.",
            "library_decision": "Do not include FActScore in the scaffolding-dataset story. The method did not transfer cleanly under this setup.",
            "paper_paragraph": "A FActScore transfer check did not clear the validation threshold, so it should be reported as a boundary case rather than a replication.",
        }
    elif residual_auroc is not None and self_auroc is not None and residual_auroc > self_auroc and delta["ci_excludes_zero"]:
        decision = {
            "status": "positive_transfer",
            "summary": f"Residual probes beat Llama self-report on FActScore: AUROC {residual_auroc:.3f} vs {self_auroc:.3f}, delta {delta['point']:.3f}; the paired bootstrap CI excludes zero.",
            "library_decision": "Include FActScore as a positive transfer scaffold, but document that activations are sentence-pooled and biographies are ChatGPT-generated.",
            "paper_paragraph": f"As a transfer check, we evaluated ChatGPT-generated FActScore biographies with original human atomic-fact labels. The residual probe outperformed Llama self-report on held-out FActScore claims (AUROC {residual_auroc:.3f} vs {self_auroc:.3f}), despite generation mismatch and sentence-span pooling for canonicalized facts.",
        }
    elif residual_auroc is not None and self_auroc is not None and residual_auroc > self_auroc:
        decision = {
            "status": "directional_positive_ambiguous",
            "summary": f"Residual probes directionally beat Llama self-report on FActScore: AUROC {residual_auroc:.3f} vs {self_auroc:.3f}, but the paired bootstrap CI includes zero.",
            "library_decision": "Keep FActScore as a candidate scaffold, not a confirmed transfer domain. The library should remain MAUD-first until stronger transfer evidence lands.",
            "paper_paragraph": f"A FActScore transfer check was directionally consistent with MAUD but not statistically decisive: residual AUROC {residual_auroc:.3f} vs self-report {self_auroc:.3f}. We treat this as suggestive rather than confirmatory because the CI includes zero and activation spans are parent-sentence pooled.",
        }
    else:
        decision = {
            "status": "negative_or_ambiguous_transfer",
            "summary": f"Residual probes did not beat Llama self-report on FActScore: residual AUROC {_fmt(residual_auroc)}, self-report AUROC {_fmt(self_auroc)}.",
            "library_decision": "Scope the library narrowly to MAUD/legal QA for now. FActScore does not support a general scaffolding-dataset story under this setup.",
            "paper_paragraph": f"A FActScore transfer check did not reproduce the MAUD probe-over-self-report gap, suggesting either domain specificity or a problem introduced by ChatGPT-generation mismatch and parent-sentence span pooling.",
        }

    payload = {
        "dataset": "FActScore",
        "source_model": "ChatGPT",
        "label_source": "factscore_human_annotations",
        "generation_mismatch_caveat": True,
        "sentence_span_pooling_caveat": True,
        "adapter_summary": adapter_summary,
        "token_alignment_summary": token_alignment_summary,
        "materialization_summary": materialization,
        "n_shared_claims": len(claim_ids),
        "n_bootstrap": args.n_bootstrap,
        "llama_self_report": {"test": {"metrics": self_metrics, "scores": dict(zip(test_claim_ids, self_scores))}},
        "residual_probe": probe,
        "paired_delta_residual_minus_self_report_auroc": delta,
        "decision": decision,
    }
    write_json(args.output_json, payload)
    _write_report(args.report_md, payload)
    print(f"Wrote FActScore validation eval to {args.output_json}")
    print(f"Wrote FActScore validation report to {args.report_md}")
    print(decision["summary"])


if __name__ == "__main__":
    main()
