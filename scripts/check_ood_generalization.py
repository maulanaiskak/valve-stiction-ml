"""Out-of-distribution stress test: scores a trained model against
synthetic configs deliberately sampled OUTSIDE sample_configs()'s
trained ranges, to check whether it learned a transferable concept of
stiction or just the specific trained regime. See ML_PLAN.md §15's
"Checked directly, not assumed: is this overfitting?" for why this
exists and what it found.

Not wired into pytest/CI: models/ is gitignored (regenerate, don't
expect it checked in -- same convention as everywhere else in this repo),
so there's no committed artifact for CI to load. Run manually after
training, same as build_synthetic_features.py.

Usage:
    python scripts/check_ood_generalization.py [path/to/model.joblib]
    # defaults to the most recently modified model under models/
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

from valve_stiction_ml.inference import predict_window
from valve_stiction_ml.synthetic import SyntheticConfig, generate_windows

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
WINDOW_SIZE = 100
N_CONFIGS = 40
SEED = 9999  # well outside build_synthetic_features.py's TRAIN_SEED/HELDOUT_SEED ranges


def latest_model_path() -> Path:
    candidates = sorted(MODELS_DIR.glob("*/model.joblib"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No model.joblib found under {MODELS_DIR} -- train one first.")
    return candidates[-1]


def sample_ood_configs(rng: np.random.Generator, n_configs: int) -> list[SyntheticConfig]:
    """Push every axis sample_configs() constrains at once, well past its
    range: bigger amplitude, longer period (exceeding window size),
    a stick_band/noise regime outside the trained fractions, and a
    different center range."""
    configs = []
    for _ in range(n_configs):
        amplitude = float(rng.uniform(40.0, 90.0))  # trained: 10-40
        configs.append(
            SyntheticConfig(
                stiction_enabled=bool(rng.integers(0, 2)),
                period_samples=int(rng.integers(85, 100)),  # trained: capped at 80
                amplitude=amplitude,
                stick_band=float(amplitude * rng.uniform(0.05, 0.15)),  # trained: 0.25-0.45
                noise_std=float(amplitude * rng.uniform(0.08, 0.20)),  # trained: 0.01-0.05
                center=float(rng.uniform(100.0, 200.0)),  # trained: 20-80
                seed=int(rng.integers(0, 2**31 - 1)),
            )
        )
    return configs


def main() -> None:
    model_path = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_model_path()
    print(f"Loading {model_path}")
    artifact = joblib.load(model_path)

    rng = np.random.default_rng(SEED)
    configs = sample_ood_configs(rng, N_CONFIGS)

    y_true, y_pred, y_prob = [], [], []
    for cfg in configs:
        (pv, op), = generate_windows(cfg, WINDOW_SIZE, n_windows=1)
        pred = predict_window(artifact, pv, op)
        y_true.append(1 if cfg.stiction_enabled else 0)
        y_pred.append(1 if pred["label"] == "yes" else 0)
        y_prob.append(pred["probability"])

    print(f"N={len(y_true)}  predict_threshold={artifact['predict_threshold']}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_true, y_pred))
    print(
        f"Accuracy={accuracy_score(y_true, y_pred):.3f} "
        f"F1={f1_score(y_true, y_pred, zero_division=0):.3f} "
        f"AUC={roc_auc_score(y_true, y_prob):.3f}"
    )
    print(
        "\nExpected: this scores far worse than the trained-regime held-out set -- "
        "that's the point. It shows where the model's generalization actually ends, "
        "rather than assuming domain-randomization training makes it unbounded."
    )


if __name__ == "__main__":
    main()
