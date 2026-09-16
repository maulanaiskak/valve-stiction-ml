"""Per-window normalization + tsfel feature extraction.

See docs/ML_PLAN.md §6, §8. Uses tsfel instead of hand-rolled feature code
(the thesis's library/utility.py `temporal` class) for speed, correctness,
and maintenance -- see ML_PLAN.md §2 research summary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import tsfel

FEATURE_DOMAINS = ["statistical", "temporal"]


def zscore(x: np.ndarray) -> np.ndarray:
    std = x.std()
    return (x - x.mean()) / std if std > 0 else x - x.mean()


def extract_features_batch(
    pv_windows: list[np.ndarray], op_windows: list[np.ndarray], window_size: int
) -> pd.DataFrame:
    """Normalize each window independently, then extract tsfel features for
    all of them in as few calls as possible.

    Calling tsfel once per window measured at ~1.4s/window (would take
    ~55 minutes for this corpus). The fix: normalize each window on its
    own, then concatenate normalized windows back-to-back and let tsfel do
    its own internal windowing in one batched call -- since every
    concatenated segment is exactly window_size samples, tsfel's internal
    split at the same window_size recovers the exact same boundaries,
    just computed far more efficiently (~11ms/window in one big call vs.
    ~1.4s calling it per window separately).
    """
    if len(pv_windows) != len(op_windows):
        raise ValueError("pv_windows and op_windows must have equal length")
    if not pv_windows:
        return pd.DataFrame()

    pv_concat = np.concatenate([zscore(w) for w in pv_windows])
    op_concat = np.concatenate([zscore(w) for w in op_windows])
    df = pd.DataFrame({"PV": pv_concat, "OP": op_concat})

    cfg = tsfel.get_features_by_domain(FEATURE_DOMAINS)
    features = tsfel.time_series_features_extractor(
        cfg, df, fs=1, window_size=window_size, overlap=0, verbose=0, n_jobs=-1
    )
    features = features.reset_index(drop=True)

    if len(features) != len(pv_windows):
        raise RuntimeError(
            f"expected {len(pv_windows)} feature rows, got {len(features)} -- "
            "a window's length probably wasn't exactly window_size"
        )
    return features


def prune_correlated_features(
    features: pd.DataFrame, threshold: float = 0.9
) -> pd.DataFrame:
    """Drop features so no remaining pair has |correlation| above threshold.

    Repeatedly finds the single most-correlated remaining pair and drops
    whichever of the two has the higher mean absolute correlation to
    everything else (the more redundant one), until no pair exceeds the
    threshold.
    """
    remaining = features.copy()
    while remaining.shape[1] > 1:
        corr = remaining.corr().abs().to_numpy(copy=True)
        np.fill_diagonal(corr, 0.0)
        if corr.max() <= threshold:
            break
        i, j = np.unravel_index(np.argmax(corr), corr.shape)
        columns = remaining.columns
        feat_i, feat_j = columns[i], columns[j]
        mean_corr = corr.mean(axis=1)
        drop = feat_i if mean_corr[i] >= mean_corr[j] else feat_j
        remaining = remaining.drop(columns=[drop])
    return remaining
