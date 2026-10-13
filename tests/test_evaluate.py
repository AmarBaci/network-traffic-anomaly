import numpy as np
import pytest

from nta.evaluate import calibrate_threshold, compute_metrics


def test_calibrate_threshold_returns_requested_quantile():
    scores = np.arange(101, dtype=float)  # 0..100

    assert calibrate_threshold(scores, 0.95) == pytest.approx(95.0)


@pytest.mark.parametrize("quantile", [0.0, 1.0, -0.1, 1.5])
def test_calibrate_threshold_rejects_invalid_quantile(quantile):
    with pytest.raises(ValueError):
        calibrate_threshold(np.array([1.0, 2.0]), quantile)


def test_compute_metrics_perfect_separation():
    y_true = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.9, 1.0])

    m = compute_metrics(y_true, scores, threshold=0.5)

    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["pr_auc"] == pytest.approx(1.0)
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(1.0)
    assert m["fpr"] == pytest.approx(0.0)
    assert m["n_samples"] == 5
    assert m["n_anomalies"] == 2


def test_compute_metrics_counts_false_positives_and_misses():
    # One normal sample is flagged (FP) and one anomaly is missed (FN).
    y_true = np.array([0, 0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.4, 0.9])

    m = compute_metrics(y_true, scores, threshold=0.5)

    assert m["fpr"] == pytest.approx(1 / 4)
    assert m["precision"] == pytest.approx(1 / 2)
    assert m["recall"] == pytest.approx(1 / 2)


def test_compute_metrics_uses_explicit_threshold_over_quantile():
    y_true = np.array([0, 1, 1, 1])
    scores = np.array([0.1, 0.6, 0.7, 0.8])

    m = compute_metrics(y_true, scores, threshold_quantile=0.95, threshold=0.5)

    assert m["threshold"] == 0.5
    assert m["recall"] == pytest.approx(1.0)


def test_compute_metrics_single_class_has_undefined_auc():
    m = compute_metrics(np.zeros(4), np.array([0.1, 0.2, 0.3, 0.4]), threshold=0.25)

    assert np.isnan(m["roc_auc"])
    assert np.isnan(m["pr_auc"])
