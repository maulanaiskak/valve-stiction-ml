import numpy as np
import pytest

from valve_stiction_ml.evaluate import (
    compute_metrics,
    find_best_threshold,
    sanity_check_agreement,
)


def test_compute_metrics_perfect_prediction():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.9, 0.8])

    m = compute_metrics(y_true, y_pred, y_proba)

    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["roc_auc"] == 1.0
    assert m["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}
    assert m["n_samples"] == 4
    assert m["positive_rate"] == 0.5


def test_compute_metrics_all_wrong():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])
    y_proba = np.array([0.9, 0.8, 0.1, 0.2])

    m = compute_metrics(y_true, y_pred, y_proba)

    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0
    assert m["confusion_matrix"] == {"tn": 0, "fp": 2, "fn": 2, "tp": 0}


def test_sanity_check_agreement():
    pred = np.array(["yes", "no", "yes", "no"])
    folder = np.array(["yes", "yes", "yes", "no"])

    result = sanity_check_agreement(pred, folder)

    assert result["agreement_with_folder_label"] == 0.75
    assert result["n_samples"] == 4


def test_sanity_check_agreement_excludes_unknown_folder_label():
    # "unknown" folder_label (e.g. SACAC files the original researchers
    # couldn't root-cause either) has nothing to compare against -- must
    # not be counted as a mismatch.
    pred = np.array(["yes", "no", "yes", "no"])
    folder = np.array(["yes", "unknown", "unknown", "no"])

    result = sanity_check_agreement(pred, folder)

    assert result["agreement_with_folder_label"] == 1.0
    assert result["n_samples"] == 2


def test_find_best_threshold_recovers_clean_separation():
    # proba clearly separates at 0.5; a low threshold should recall
    # everything but hurt precision, so f1 should peak near the true gap
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

    threshold = find_best_threshold(y_true, y_proba)

    preds = (y_proba >= threshold).astype(int)
    np.testing.assert_array_equal(preds, y_true)


def test_find_best_threshold_rejects_unsupported_metric():
    with pytest.raises(ValueError):
        find_best_threshold(np.array([0, 1]), np.array([0.2, 0.8]), metric="accuracy")
