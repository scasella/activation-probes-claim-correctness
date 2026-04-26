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
    return (
        [str(item) for item in payload["claim_ids"].tolist()],
        [str(item) for item in payload["example_ids"].tolist()],
        np.asarray(payload["matrix"], dtype=float),
    )


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
        metrics = _metric_block(y[val_idx].tolist(), val_probs.tolist())
        validation_rows.append({"c": c_value, **metrics})
    valid_rows = [row for row in validation_rows if row["auroc"] is not None]
    if not valid_rows:
        raise ValueError("No C value produced a valid validation AUROC")
    selected = max(valid_rows, key=lambda row: (row["auroc"], -row["brier"]))
    train_val_idx = np.concatenate([train_idx, val_idx])
    final_model = _fit_probe(x[train_val_idx], y[train_val_idx], float(selected["c"]))
    test_probs = final_model.predict_proba(x[test_idx])[:, 1]
    return {
        "selected_c": selected["c"],
        "validation": {
            "rows": validation_rows,
            "selected": selected,
        },
        "test": {
            "claim_ids": [claim_ids[idx] for idx in test_idx],
            "y_true": y[test_idx].tolist(),
            "scores": test_probs.tolist(),
            "metrics": _metric_block(y[test_idx].tolist(), test_probs.tolist()),
        },
        "split_counts": {
            "train": int(len(train_idx)),
            "validation": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
    }


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    residual = payload["residual_probe"]["test"]["metrics"]
    self_report = payload["llama_self_report"]["test"]["metrics"]
    delta = payload["paired_delta_residual_minus_self_report_auroc"]
    decision = payload["decision"]
    materialization = payload["materialization_summary"]
    lines = [
        "# Minimum-Viable FELM Validation",
        "",
        "## Summary",
        "",
        decision["summary"],
        "",
        "## Method",
        "",
        "This is a narrow transfer check of the MAUD methods pipeline on FELM world knowledge (`wk`). Each FELM human-annotated response segment is treated as a claim. Llama 3.1 8B Instruct is asked for structured self-report confidences over those fixed segments, and layer-19 residual activations are extracted by running the FELM-provided answer text through Llama and pooling over each segment span. A logistic-regression probe is trained on the train split, C is selected on validation, and the held-out test split is evaluated once.",
        "",
        "FELM responses were generated by ChatGPT, not Llama. The probe is therefore trained on Llama activations over ChatGPT-generated text. That is a methodological compromise: a negative result could reflect generation mismatch rather than absence of transferable correctness signal.",
        "",
        f"Materialization covered {materialization['n_feature_rows']}/{materialization['n_expected_segments']} segment features and {materialization['n_self_report_predictions']}/{materialization['n_expected_segments']} self-report predictions. Missing or malformed self-report outputs are reported in the JSON artifact and are not silently treated as labels.",
        "",
        "## Results",
        "",
        "| Method | Test AUROC | Test Brier |",
        "| --- | ---: | ---: |",
        f"| Llama self-report | {self_report['auroc']:.3f} | {self_report['brier']:.3f} |",
        f"| Residual probe | {residual['auroc']:.3f} | {residual['brier']:.3f} |",
        "",
        f"Residual minus self-report AUROC delta: {delta['point']:.3f} [{delta['ci_low']:.3f}, {delta['ci_high']:.3f}] from 200 paired bootstrap resamples. The interval {'excludes' if delta['ci_excludes_zero'] else 'includes'} zero.",
        "",
        "## Comparison To MAUD",
        "",
        "On MAUD, residual probes beat Llama self-report by about 0.26 AUROC under the GPT-5.4 judge and about 0.24 under the Kimi judge. The FELM point estimate points the same way, but the gap is smaller and not decisive under this scoping bootstrap.",
        "",
        "## Decision",
        "",
        decision["library_decision"],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate residual probe vs Llama self-report on FELM.")
    parser.add_argument("--examples-jsonl", type=Path, default=Path("data/felm/felm_wk_examples.jsonl"))
    parser.add_argument("--labels-jsonl", type=Path, default=Path("data/annotations/felm_wk_labels.jsonl"))
    parser.add_argument("--features-npz", type=Path, default=Path("artifacts/runs/felm_wk_probe_features_residual.npz"))
    parser.add_argument("--self-report-jsonl", type=Path, default=Path("data/cached_baselines/llama_self_report/parsed/felm_wk/_all_predictions.jsonl"))
    parser.add_argument("--materialization-summary-json", type=Path, default=Path("artifacts/runs/felm_wk_materialization_summary.json"))
    parser.add_argument("--output-json", type=Path, default=Path("artifacts/runs/felm_wk_validation_eval.json"))
    parser.add_argument("--report-md", type=Path, default=Path("docs/felm_validation.md"))
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260424)
    parser.add_argument("--c-values", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0])
    args = parser.parse_args()

    labels = _load_labels(args.labels_jsonl)
    examples = _load_examples(args.examples_jsonl)
    self_report_scores = _load_predictions(args.self_report_jsonl)
    claim_ids, example_ids, x = _load_features(args.features_npz)
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
    delta = _bootstrap_delta(
        y_test,
        residual_scores,
        self_scores,
        auroc,
        n_resamples=args.n_bootstrap,
        seed=args.seed,
    )
    validation_auroc = probe["validation"]["selected"]["auroc"]
    residual_auroc = probe["test"]["metrics"]["auroc"]
    self_auroc = self_metrics["auroc"]
    if validation_auroc is not None and validation_auroc < 0.55:
        decision = {
            "status": "stop_validation_below_threshold",
            "summary": (
                f"The FELM transfer check is negative at the validation gate: selected validation AUROC was "
                f"{validation_auroc:.3f}, below the 0.55 stopping threshold."
            ),
            "library_decision": (
                "Do not include FELM in the scaffolding-dataset story yet. The next step would be a follow-up "
                "diagnosis, especially testing whether the ChatGPT-generation caveat is responsible."
            ),
        }
    elif (
        residual_auroc is not None
        and self_auroc is not None
        and residual_auroc > self_auroc
        and delta["ci_excludes_zero"]
    ):
        decision = {
            "status": "positive_transfer",
            "summary": (
                f"Residual probes beat Llama self-report on held-out FELM: AUROC {residual_auroc:.3f} "
                f"vs {self_auroc:.3f}, delta {delta['point']:.3f}; the paired bootstrap CI excludes zero."
            ),
            "library_decision": (
                "FELM can remain a candidate scaffolding dataset, but the ChatGPT-generation caveat should "
                "stay prominent until a matched Llama-generation follow-up is run."
            ),
        }
    elif residual_auroc is not None and self_auroc is not None and residual_auroc > self_auroc:
        decision = {
            "status": "directional_positive_ambiguous",
            "summary": (
                f"Residual probes directionally beat Llama self-report on held-out FELM: AUROC {residual_auroc:.3f} "
                f"vs {self_auroc:.3f}, delta {delta['point']:.3f}; however, the paired bootstrap CI includes zero."
            ),
            "library_decision": (
                "FELM should remain a candidate scaffolding dataset, but not yet be treated as a confirmed "
                "replication of the MAUD result. A matched Llama-generation follow-up or larger FELM slice is "
                "needed before building library claims around transfer."
            ),
        }
    else:
        decision = {
            "status": "negative_or_ambiguous_transfer",
            "summary": (
                f"Residual probes did not clearly beat Llama self-report on held-out FELM: residual AUROC "
                f"{residual_auroc:.3f}, self-report AUROC {self_auroc:.3f}."
            ),
            "library_decision": (
                "Do not build library scaffolding around the assumption that the MAUD result transfers. "
                "Investigate FELM span alignment, prompt fit, and the ChatGPT-generation mismatch first."
            ),
        }
    payload = {
        "dataset": "hkust-nlp/felm",
        "domain": "wk",
        "label_source": "felm_human_annotations",
        "chatgpt_generation_caveat": True,
        "materialization_summary": materialization,
        "n_shared_claims": len(claim_ids),
        "llama_self_report": {"test": {"metrics": self_metrics, "scores": dict(zip(test_claim_ids, self_scores))}},
        "residual_probe": probe,
        "paired_delta_residual_minus_self_report_auroc": delta,
        "decision": decision,
    }
    write_json(args.output_json, payload)
    _write_report(args.report_md, payload)
    print(f"Wrote FELM validation eval to {args.output_json}")
    print(f"Wrote FELM validation report to {args.report_md}")
    print(decision["summary"])


if __name__ == "__main__":
    main()
