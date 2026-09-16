import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from valve_stiction_ml.features import extract_features_batch
from valve_stiction_ml.inference import predict_window


def make_toy_artifact(window_size=100, n_features=5, seed=0):
    """A real (small) trained model + a subset of real tsfel feature names,
    matching the shape of what train.py actually produces -- not a mock."""
    rng = np.random.default_rng(seed)
    pv_windows = [rng.normal(0, 1, window_size) for _ in range(30)]
    op_windows = [rng.normal(0, 1, window_size) for _ in range(30)]
    all_features = extract_features_batch(pv_windows, op_windows, window_size)

    feature_names = list(all_features.columns[:n_features])
    X = all_features[feature_names].to_numpy()
    y = (X[:, 0] > np.median(X[:, 0])).astype(int)

    model = RandomForestClassifier(n_estimators=10, random_state=seed).fit(X, y)

    return {
        "model": model,
        "feature_names": feature_names,
        "window_size": window_size,
        "predict_threshold": 0.5,
    }


def test_predict_window_returns_valid_prediction():
    artifact = make_toy_artifact()
    rng = np.random.default_rng(1)
    pv = rng.normal(0, 1, 100)
    op = rng.normal(0, 1, 100)

    result = predict_window(artifact, pv, op)

    assert result["label"] in ("yes", "no")
    assert 0.0 <= result["probability"] <= 1.0


def test_predict_window_raises_on_wrong_length():
    artifact = make_toy_artifact(window_size=100)
    pv = np.zeros(50)
    op = np.zeros(50)

    with pytest.raises(ValueError):
        predict_window(artifact, pv, op)


def test_predict_window_only_uses_declared_feature_names():
    # If this used all raw tsfel columns instead of artifact["feature_names"],
    # it would still "work" by accident here since predict_proba would get
    # the wrong number of columns and raise -- so the real assertion is that
    # a model trained on a *specific* small subset still gets scored
    # correctly, not that some error appears.
    artifact = make_toy_artifact(n_features=3)
    assert len(artifact["feature_names"]) == 3
    rng = np.random.default_rng(2)
    pv, op = rng.normal(0, 1, 100), rng.normal(0, 1, 100)

    result = predict_window(artifact, pv, op)
    assert result["label"] in ("yes", "no")
