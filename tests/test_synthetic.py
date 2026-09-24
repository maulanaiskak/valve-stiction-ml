"""Tests for synthetic.py -- see its module docstring for why this exists
(closing the train/serve gap valve-stiction-pipeline's streaming
evaluation found)."""

import numpy as np
from valve_stiction_ml.classic import ellipse_stiction_index, kano_pattern_check
from valve_stiction_ml.synthetic import SyntheticConfig, generate_windows, sample_configs


def zscore(x: np.ndarray) -> np.ndarray:
    std = x.std()
    return (x - x.mean()) / std if std > 0 else x - x.mean()


def test_generate_windows_returns_requested_shape():
    cfg = SyntheticConfig(
        stiction_enabled=True, period_samples=50, amplitude=20.0,
        stick_band=7.0, noise_std=0.4, center=50.0, seed=0,
    )
    windows = generate_windows(cfg, window_size=100, n_windows=3)
    assert len(windows) == 3
    for pv, op in windows:
        assert pv.shape == (100,)
        assert op.shape == (100,)


def test_generate_windows_is_deterministic_given_seed():
    cfg = SyntheticConfig(
        stiction_enabled=False, period_samples=50, amplitude=20.0,
        stick_band=7.0, noise_std=0.4, center=50.0, seed=42,
    )
    windows_a = generate_windows(cfg, window_size=100, n_windows=2)
    windows_b = generate_windows(cfg, window_size=100, n_windows=2)
    for (pv_a, op_a), (pv_b, op_b) in zip(windows_a, windows_b):
        np.testing.assert_array_equal(pv_a, pv_b)
        np.testing.assert_array_equal(op_a, op_b)


def test_stiction_configs_are_detected_by_classic_detector():
    """Regression check, same spirit as valve-stiction-simulator's own
    test: ground truth (stiction_enabled) should actually be detectable
    by the real classic detector, not just a label we assert by fiat."""
    rng = np.random.default_rng(7)
    configs = sample_configs(rng, n_configs=20)

    correct = 0
    for cfg in configs:
        (pv, op), = generate_windows(cfg, window_size=100, n_windows=1)
        pv_z, op_z = zscore(pv), zscore(op)
        ellipse_verdict = ellipse_stiction_index(pv_z, op_z) >= 0.3762
        kano_verdict = kano_pattern_check(pv_z, op_z)
        predicted = ellipse_verdict and kano_verdict
        if predicted == cfg.stiction_enabled:
            correct += 1

    # Not every random config is expected to be unambiguous (some
    # combinations of noise/stick_band are inherently borderline), but the
    # classic detector should agree with ground truth on most of them.
    assert correct / len(configs) >= 0.7


def test_sample_configs_produces_both_classes():
    rng = np.random.default_rng(123)
    configs = sample_configs(rng, n_configs=50)
    labels = {cfg.stiction_enabled for cfg in configs}
    assert labels == {True, False}
