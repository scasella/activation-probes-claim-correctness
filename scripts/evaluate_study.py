from __future__ import annotations

import argparse
from pathlib import Path

from interp_experiment.evaluation.metrics import auroc, brier_score, calibration_bin_stats, paired_bootstrap_metric_delta
from interp_experiment.io import read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one or two scored methods against binary labels.")
    parser.add_argument("--truth-json", type=Path, required=True, help="JSON with {'y_true': [...], 'scores_a': [...], 'scores_b': [...]} or without scores_b.")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    args = parser.parse_args()

    payload = read_json(args.truth_json)
    y_true = payload["y_true"]
    scores_a = payload["scores_a"]
    result = {
        "method_a": {
            "auroc": auroc(y_true, scores_a),
            "brier": brier_score(y_true, scores_a),
            "calibration_bins": calibration_bin_stats(y_true, scores_a),
        }
    }
    if "scores_b" in payload:
        scores_b = payload["scores_b"]
        result["method_b"] = {
            "auroc": auroc(y_true, scores_b),
            "brier": brier_score(y_true, scores_b),
            "calibration_bins": calibration_bin_stats(y_true, scores_b),
        }
        result["paired_delta_auroc"] = paired_bootstrap_metric_delta(
            y_true,
            scores_a,
            scores_b,
            metric=auroc,
            n_resamples=args.bootstrap_resamples,
        )
        result["paired_delta_brier"] = paired_bootstrap_metric_delta(
            y_true,
            scores_b,
            scores_a,
            metric=brier_score,
            n_resamples=args.bootstrap_resamples,
        )
    write_json(args.output_json, result)
    print(f"Wrote evaluation summary to {args.output_json}")


if __name__ == "__main__":
    main()
