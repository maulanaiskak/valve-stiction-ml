"""Build synthetic training-augmentation data (see synthetic.py's module
docstring for why this exists). Two disjoint corpora, different random
ranges of the master seed so they never overlap:

  - synthetic_train_features.csv:   fed into train.py alongside ISDB
  - synthetic_heldout_features.csv: reported on but never trained on,
    exactly like SACAC -- the honest check of whether augmentation
    actually generalizes vs. just memorizing the training configs

Usage:
    python scripts/build_synthetic_features.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from valve_stiction_ml.features import extract_features_batch
from valve_stiction_ml.synthetic import generate_windows, sample_configs

REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = REPO_ROOT / "data" / "processed" / "features.csv"
TRAIN_OUT = REPO_ROOT / "data" / "processed" / "synthetic_train_features.csv"
HELDOUT_OUT = REPO_ROOT / "data" / "processed" / "synthetic_heldout_features.csv"

WINDOW_SIZE = 100
WINDOWS_PER_CONFIG = 5
N_TRAIN_CONFIGS = 250
N_HELDOUT_CONFIGS = 20
TRAIN_SEED = 1000
HELDOUT_SEED = 2000  # disjoint range from TRAIN_SEED -- see module docstring


def build_corpus(feature_names: list[str], seed: int, n_configs: int, prefix: str) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    configs = sample_configs(rng, n_configs)

    pv_windows, op_windows, rows = [], [], []
    for i, cfg in enumerate(configs):
        windows = generate_windows(cfg, WINDOW_SIZE, WINDOWS_PER_CONFIG)
        for j, (pv, op) in enumerate(windows):
            pv_windows.append(pv)
            op_windows.append(op)
            rows.append(
                {
                    "source_file": f"{prefix}-{i:04d}.csv",
                    "loop_id": f"{prefix}-{i:04d}",
                    "origin_dataset": "SYNTHETIC",
                    "folder_label": "yes" if cfg.stiction_enabled else "no",
                    "derived_label": "yes" if cfg.stiction_enabled else "no",
                }
            )

    print(f"{prefix}: {len(configs)} configs x {WINDOWS_PER_CONFIG} windows = {len(rows)} windows")
    raw_features = extract_features_batch(pv_windows, op_windows, WINDOW_SIZE)
    # Reuse the exact pruned feature set the real-data pipeline already
    # settled on (extract_features.py's correlation pruning) -- adding
    # more labeled examples in a fixed feature space, not re-deciding
    # which features to use based on synthetic data too.
    selected = raw_features[feature_names].reset_index(drop=True)
    return pd.concat([pd.DataFrame(rows), selected], axis=1)


def main() -> None:
    features_df = pd.read_csv(FEATURES_PATH)
    meta_columns = ["source_file", "loop_id", "origin_dataset", "folder_label", "derived_label"]
    feature_names = [c for c in features_df.columns if c not in meta_columns]

    train_df = build_corpus(feature_names, TRAIN_SEED, N_TRAIN_CONFIGS, "synth-train")
    train_df.to_csv(TRAIN_OUT, index=False)
    print(f"Wrote {train_df.shape[0]} rows to {TRAIN_OUT}")

    heldout_df = build_corpus(feature_names, HELDOUT_SEED, N_HELDOUT_CONFIGS, "synth-heldout")
    heldout_df.to_csv(HELDOUT_OUT, index=False)
    print(f"Wrote {heldout_df.shape[0]} rows to {HELDOUT_OUT}")


if __name__ == "__main__":
    main()
