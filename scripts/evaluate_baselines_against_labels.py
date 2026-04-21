from __future__ import annotations

import argparse
from pathlib import Path

from interp_experiment.evaluation.metrics import auroc, brier_score, paired_bootstrap_metric_delta
from interp_experiment.io import read_jsonl, write_json
from interp_experiment.schemas import BaselinePrediction
from interp_experiment.utils import ensure_parent


def _load_predictions(path: Path) -> dict[str, BaselinePrediction]:
    rows: list[BaselinePrediction] = []
    if path.is_dir():
        for file_path in sorted(path.glob("*.jsonl")):
            rows.extend(BaselinePrediction.from_dict(row) for row in read_jsonl(file_path))
    else:
        rows.extend(BaselinePrediction.from_dict(row) for row in read_jsonl(path))
    return {row.claim_id: row for row in rows}


def _score_method(name: str, predictions: dict[str, BaselinePrediction], labels: list[dict[str, str]]) -> dict[str, object]:
    shared = [row for row in labels if row["claim_id"] in predictions]
    if not shared:
        raise ValueError(f"No overlapping claim ids for method {name}")
    correctness_true = [1 if row["correctness_label"] == "true" else 0 for row in shared]
    load_true = [1 if row["load_bearing_label"] == "yes" else 0 for row in shared]
    correctness_scores = [predictions[row["claim_id"]].correctness_confidence for row in shared]
    load_scores = [predictions[row["claim_id"]].load_bearing_confidence for row in shared]

    def _safe_auroc(y_true, y_score):
        try:
            return auroc(y_true, y_score)
        except ValueError:
            return None

    return {
        "n_claims": len(shared),
        "correctness": {
            "auroc": _safe_auroc(correctness_true, correctness_scores),
            "brier": brier_score(correctness_true, correctness_scores),
        },
        "load_bearing": {
            "auroc": _safe_auroc(load_true, load_scores),
            "brier": brier_score(load_true, load_scores),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate precomputed baselines against a label file (human or proxy).")
    parser.add_argument("--labels-jsonl", type=Path, required=True)
    parser.add_argument("--llama-predictions", type=Path, required=True)
    parser.add_argument("--gpt54-predictions", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--label-source", default="judge_llm_proxy")
    args = parser.parse_args()

    labels = [row for row in read_jsonl(args.labels_jsonl)]
    llama_predictions = _load_predictions(args.llama_predictions)
    gpt_predictions = _load_predictions(args.gpt54_predictions)

    payload = {
        "label_source": args.label_source,
        "llama_self_report": _score_method("llama_self_report", llama_predictions, labels),
        "gpt54_cached": _score_method("gpt54_cached", gpt_predictions, labels),
    }

    shared_claims = [
        row
        for row in labels
        if row["claim_id"] in llama_predictions and row["claim_id"] in gpt_predictions
    ]
    correctness_true = [1 if row["correctness_label"] == "true" else 0 for row in shared_claims]
    load_true = [1 if row["load_bearing_label"] == "yes" else 0 for row in shared_claims]
    if shared_claims:
        payload["paired_deltas"] = {
            "correctness_brier": paired_bootstrap_metric_delta(
                correctness_true,
                [gpt_predictions[row["claim_id"]].correctness_confidence for row in shared_claims],
                [llama_predictions[row["claim_id"]].correctness_confidence for row in shared_claims],
                metric=brier_score,
                n_resamples=1000,
            ),
            "load_bearing_brier": paired_bootstrap_metric_delta(
                load_true,
                [gpt_predictions[row["claim_id"]].load_bearing_confidence for row in shared_claims],
                [llama_predictions[row["claim_id"]].load_bearing_confidence for row in shared_claims],
                metric=brier_score,
                n_resamples=1000,
            ),
        }
    ensure_parent(args.output_json)
    write_json(args.output_json, payload)
    print(f"Wrote baseline-vs-label summary to {args.output_json}")


if __name__ == "__main__":
    main()
