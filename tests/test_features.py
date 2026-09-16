import numpy as np
import pandas as pd
import pytest

from valve_stiction_ml.features import extract_features_batch, prune_correlated_features


def test_extract_features_batch_one_row_per_window():
    rng = np.random.default_rng(0)
    pv_windows = [rng.normal(0, 1, 100) for _ in range(5)]
    op_windows = [rng.normal(0, 1, 100) for _ in range(5)]

    features = extract_features_batch(pv_windows, op_windows, window_size=100)

    assert len(features) == 5
    assert features.shape[1] > 0
    assert not features.isna().any().any()


def test_extract_features_batch_empty_input():
    features = extract_features_batch([], [], window_size=100)
    assert len(features) == 0


def test_extract_features_batch_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        extract_features_batch([np.zeros(100)], [np.zeros(100), np.zeros(100)], 100)


def test_extract_features_batch_deterministic():
    rng = np.random.default_rng(1)
    pv_windows = [rng.normal(0, 1, 100) for _ in range(3)]
    op_windows = [rng.normal(0, 1, 100) for _ in range(3)]

    first = extract_features_batch(pv_windows, op_windows, window_size=100)
    second = extract_features_batch(pv_windows, op_windows, window_size=100)

    pd.testing.assert_frame_equal(first, second)


def test_prune_correlated_features_drops_duplicate_column():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 200)
    df = pd.DataFrame({"a": a, "a_copy": a, "independent": rng.normal(0, 1, 200)})

    pruned = prune_correlated_features(df, threshold=0.9)

    assert pruned.shape[1] == 2
    assert "independent" in pruned.columns
    assert ("a" in pruned.columns) != ("a_copy" in pruned.columns)


def test_prune_correlated_features_keeps_independent_columns():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {f"col_{i}": rng.normal(0, 1, 200) for i in range(5)}
    )

    pruned = prune_correlated_features(df, threshold=0.9)

    assert pruned.shape[1] == 5
