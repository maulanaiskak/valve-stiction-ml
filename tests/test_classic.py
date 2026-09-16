import numpy as np

from valve_stiction_ml.classic import (
    ellipse_stiction_index,
    has_sufficient_activity,
    kano_pattern_check,
    label_window,
)


def zscore(x: np.ndarray) -> np.ndarray:
    std = x.std()
    return (x - x.mean()) / std if std > 0 else x - x.mean()


def make_stick_slip(n=100, stick_band=0.35, seed=0):
    """OP triangle wave; PV lags until deviation exceeds the stick band,
    then slips to catch up -- the standard simplified stiction model."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    op = 2 * np.abs(((t / 25) % 2) - 1) - 1
    pv = np.zeros(n)
    pv[0] = op[0]
    for i in range(1, n):
        pv[i] = op[i] if abs(op[i] - pv[i - 1]) > stick_band else pv[i - 1]
    return pv + rng.normal(0, 0.02, n), op + rng.normal(0, 0.02, n)


def make_lagged_sine(n=100, lag=0.3, seed=0):
    """PV tracks OP sinusoidally with a phase lag -- oscillating but no
    stick-slip (e.g. aggressive tuning or an external disturbance)."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 4 * np.pi, n)
    op = np.sin(x)
    pv = np.sin(x - lag) + rng.normal(0, 0.03, n)
    return pv, op


def make_calm(n=100, seed=0):
    """Valve essentially at rest -- OP barely moves at all."""
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.01, n), rng.normal(0, 0.01, n)


def test_kano_detects_stick_slip_but_not_lagged_sine():
    pv_s, op_s = make_stick_slip()
    pv_h, op_h = make_lagged_sine()

    assert kano_pattern_check(zscore(pv_s), zscore(op_s)) is True
    assert kano_pattern_check(zscore(pv_h), zscore(op_h)) is False


def test_ellipse_index_higher_for_stick_slip_than_lagged_sine():
    pv_s, op_s = make_stick_slip()
    pv_h, op_h = make_lagged_sine()

    idx_stiction = ellipse_stiction_index(zscore(pv_s), zscore(op_s))
    idx_healthy = ellipse_stiction_index(zscore(pv_h), zscore(op_h))

    assert idx_stiction > idx_healthy


def test_ellipse_index_zero_for_flat_pv():
    pv = np.zeros(100)
    op = np.linspace(-1, 1, 100)

    assert ellipse_stiction_index(pv, op) == 0.0


def test_ellipse_index_raises_on_too_short_window():
    pv = np.array([0.0, 0.1])
    op = np.array([0.0, 0.1])

    try:
        ellipse_stiction_index(pv, op, min_band_points=4)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_has_sufficient_activity_flags_inactive_op():
    _, op_calm = make_calm()
    _, op_active = make_stick_slip()

    reference_std = op_active.std()

    assert has_sufficient_activity(op_calm, reference_std) is False
    assert has_sufficient_activity(op_active, reference_std) is True


def test_has_sufficient_activity_no_reference_means_no_filtering():
    _, op_calm = make_calm()
    assert has_sufficient_activity(op_calm, reference_op_std=0.0) is True


def test_label_window_returns_no_for_inactive_op_without_running_detectors():
    pv_calm, op_calm = make_calm()
    _, op_active = make_stick_slip()

    label = label_window(
        pv_calm, op_calm, ellipse_threshold=1.0, reference_op_std=op_active.std()
    )

    assert label == "no"


def test_label_window_normalizes_raw_input_consistently_with_derive_label():
    pv_s, op_s = make_stick_slip()

    # a generous threshold (0.5) should agree with derive_label on
    # already-normalized input, confirming label_window's internal
    # normalization matches what tests elsewhere assume
    from valve_stiction_ml.classic import derive_label

    expected = derive_label(zscore(pv_s), zscore(op_s), ellipse_threshold=0.5)
    actual = label_window(pv_s, op_s, ellipse_threshold=0.5, reference_op_std=None)

    assert actual == expected
