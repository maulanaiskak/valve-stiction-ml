"""Metrics + sanity-check reporting. See docs/ML_PLAN.md §10.

Never report accuracy alone on this imbalanced data (~19% positive) --
precision/recall/F1/PR-AUC/ROC-AUC/confusion matrix, per ML_PLAN.md §10.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def find_best_threshold(
    y_true: np.ndarray, y_proba: np.ndarray, metric: str = "f1"
) -> float:
    """Search a fine grid of thresholds for the one maximizing `metric` on
    the given predictions.

    Only ever call this with ISDB out-of-fold predictions -- SACAC must
    stay untouched by any tuning, per ML_PLAN.md §5, same discipline as
    ellipse_threshold in classic.py.
    """
    if metric != "f1":
        raise ValueError("only 'f1' is implemented")

    candidates = np.linspace(0.01, 0.99, 197)
    best_threshold, best_score = 0.5, -1.0
    for t in candidates:
        score = f1_score(y_true, (y_proba >= t).astype(int), zero_division=0)
        if score > best_score:
            best_threshold, best_score = float(t), float(score)
    return best_threshold


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    """y_pred is the thresholded (0.5) prediction; y_proba is P(positive)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_samples": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
    }


def sanity_check_agreement(y_pred_label: np.ndarray, folder_label: np.ndarray) -> dict:
    """Agreement between a "yes"/"no" prediction and the old thesis folder
    labels. Reported only -- never used to pick a model or hyperparameter,
    per ML_PLAN.md §2, §4.

    Excludes folder_label == "unknown" (e.g. SACAC's own files where the
    original researchers couldn't determine a root cause either) -- there's
    nothing to compare against there, and counting it as a mismatch would
    understate the agreement rate for no real reason.
    """
    known = folder_label != "unknown"
    agree = (y_pred_label[known] == folder_label[known]).mean()
    return {"agreement_with_folder_label": float(agree), "n_samples": int(known.sum())}
