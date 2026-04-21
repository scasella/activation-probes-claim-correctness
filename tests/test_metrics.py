from interp_experiment.evaluation.metrics import auroc, brier_score, calibration_bin_stats, paired_bootstrap_metric_delta


def test_auroc_and_brier_score() -> None:
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.4, 0.6, 0.9]
    assert round(auroc(y_true, y_score), 6) == 1.0
    assert round(brier_score(y_true, y_score), 6) == 0.085


def test_paired_bootstrap_metric_delta_returns_interval() -> None:
    y_true = [0, 0, 1, 1]
    better = [0.1, 0.2, 0.8, 0.9]
    worse = [0.4, 0.5, 0.6, 0.7]
    result = paired_bootstrap_metric_delta(y_true, better, worse, metric=auroc, n_resamples=100)
    assert result["delta"] >= 0.0
    assert result["ci_low"] <= result["ci_high"]


def test_calibration_bin_stats_have_expected_keys() -> None:
    rows = calibration_bin_stats([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8], n_bins=2)
    assert rows
    assert {"bin_left", "bin_right", "mean_confidence", "empirical_accuracy", "count"} <= set(rows[0])
