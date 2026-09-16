"""Load a trained model artifact and score a single raw PV/OP window.

The artifact from train.py is self-describing (feature_names, window_size,
normalization, predict_threshold all travel with the model) specifically so
a caller doesn't have to know or guess any of this separately -- unlike the
thesis's subscribe.py, where the feature list was hardcoded independently
of the model file and could silently drift out of sync with it.

What this module does NOT do: it doesn't buffer/window a live stream (that's
the PRD's V1 service's job) or run the classic detector (classic.py, used
only to generate training labels). This is purely: one raw window in, one
prediction out.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TypedDict

import joblib
import numpy as np

from valve_stiction_ml.features import extract_features_batch


class Prediction(TypedDict):
    label: Literal["yes", "no"]
    probability: float


def load_artifact(path: Path) -> dict:
    return joblib.load(path)


def predict_window(artifact: dict, pv_raw: np.ndarray, op_raw: np.ndarray) -> Prediction:
    """Score one raw (non-normalized) PV/OP window with a loaded artifact.

    Raises ValueError if the window length doesn't match what the model was
    trained on -- silently truncating or padding would produce a
    prediction that looks valid but isn't, which is worse than failing
    loudly.
    """
    window_size = artifact["window_size"]
    if len(pv_raw) != window_size or len(op_raw) != window_size:
        raise ValueError(
            f"expected windows of length {window_size}, got "
            f"pv={len(pv_raw)}, op={len(op_raw)}"
        )

    raw_features = extract_features_batch([pv_raw], [op_raw], window_size)
    # The model was trained on a pruned subset of tsfel's full feature set
    # (features.py's correlation pruning) -- select exactly those columns,
    # in the order the model expects. Never assume raw tsfel output already
    # matches; that assumption is exactly what a stale/hand-copied feature
    # list (the thesis's failure mode) would get wrong silently.
    X = raw_features[artifact["feature_names"]].to_numpy()

    proba = float(artifact["model"].predict_proba(X)[0, 1])
    label: Literal["yes", "no"] = "yes" if proba >= artifact["predict_threshold"] else "no"
    return {"label": label, "probability": proba}
