"""Validates classic.py against SACAC's own literature-tagged files.

These labels come from the published benchmark papers (Horch 2003,
Bacci di Capaci & Scali 2018, Thornhill 2002/2003), not from guessing -- see
ML_PLAN.md §14. Skipped if data/raw/ hasn't been populated yet (run
scripts/import_thesis_data.py first); these are opportunistic integration
tests, not run against committed data (the thesis CSVs aren't in this repo).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from valve_stiction_ml.classic import ellipse_stiction_index, kano_pattern_check
from valve_stiction_ml.dataset import window_signal

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw" / "SACAC"

FIXTURES = {
    "easy_positive": RAW_DIR / "yes" / "stiction-F-paper-horch-2003.csv",
    "harder_positive": RAW_DIR / "yes" / "stiction-P-oilgas-DB-1-baccidicapaci-2018.csv",
    "hard_negative_saturation": RAW_DIR / "no" / "saturation-T-oilgas-thornhill-2002.csv",
    "hard_negative_quantization": RAW_DIR / "no" / "quantisation-Q-paper-horch-2003.csv",
}

pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in FIXTURES.values()),
    reason="thesis data not imported -- run scripts/import_thesis_data.py first",
)


def zscore(x: np.ndarray) -> np.ndarray:
    std = x.std()
    return (x - x.mean()) / std if std > 0 else x - x.mean()


def windows_for(path: Path, window_size: int = 100):
    df = pd.read_csv(path)
    return window_signal(df["PV"].to_numpy(float), df["OP"].to_numpy(float), window_size)


def test_easy_positive_detected_in_most_windows():
    # stiction-F-paper: every one of its 11 windows showed the stick-slip
    # signature when this was checked manually -- a strong, consistent
    # positive case (unlike harder_positive below, whose file mostly
    # doesn't).
    windows = windows_for(FIXTURES["easy_positive"])
    kano_hits = sum(kano_pattern_check(zscore(pv), zscore(op)) for pv, op in windows)
    assert kano_hits / len(windows) >= 0.8


def test_harder_positive_is_not_uniformly_positive():
    # stiction-P-oilgas-DB-1: PV sits essentially flat for most of the
    # file and only shows real activity in its last couple of windows --
    # exactly the file-vs-window label mismatch this whole detector exists
    # to fix (ML_PLAN.md §2). This is a regression guard on that finding:
    # if this ever starts reporting "yes" for every window, something
    # about the detector's sensitivity has drifted, since we know from
    # inspection that isn't true of this file.
    windows = windows_for(FIXTURES["harder_positive"])
    kano_hits = sum(kano_pattern_check(zscore(pv), zscore(op)) for pv, op in windows)
    assert 0 < kano_hits < len(windows)


def test_hard_negative_saturation_never_flagged():
    windows = windows_for(FIXTURES["hard_negative_saturation"])
    kano_hits = sum(kano_pattern_check(zscore(pv), zscore(op)) for pv, op in windows)
    assert kano_hits == 0


def test_hard_negative_quantization_never_flagged():
    # Regression guard for the quantization/stick-slip confusion found
    # during milestone-2 validation: without is_likely_quantized(), every
    # window in this file was a false positive (same "flat, then jump"
    # step shape as real stick-slip, for an unrelated reason -- DAC/sensor
    # resolution, not valve friction).
    windows = windows_for(FIXTURES["hard_negative_quantization"])
    kano_hits = sum(kano_pattern_check(zscore(pv), zscore(op)) for pv, op in windows)
    assert kano_hits == 0


def test_ellipse_index_lower_for_saturation_than_stiction():
    stiction_windows = windows_for(FIXTURES["easy_positive"])
    saturation_windows = windows_for(FIXTURES["hard_negative_saturation"])

    stiction_idx = np.mean(
        [ellipse_stiction_index(zscore(pv), zscore(op)) for pv, op in stiction_windows]
    )
    saturation_idx = np.mean(
        [ellipse_stiction_index(zscore(pv), zscore(op)) for pv, op in saturation_windows]
    )

    assert stiction_idx > saturation_idx
