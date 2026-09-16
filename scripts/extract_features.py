"""Milestone 4: extract tsfel features for every confident window.

Reads data/processed/window_labels.csv (from label_windows.py), drops
"uncertain" windows, reconstructs each confident window's raw PV/OP from
data/raw/, extracts tsfel features (normalized per-window), prunes
highly-correlated features, and writes data/processed/features.csv.

Usage:
    python scripts/extract_features.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from valve_stiction_ml.dataset import load_manifest, load_signal, window_signal
from valve_stiction_ml.features import extract_features_batch, prune_correlated_features

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
LABELS_PATH = REPO_ROOT / "data" / "processed" / "window_labels.csv"
MANIFEST_PATH = REPO_ROOT / "data" / "processed" / "manifest.csv"
OUT_PATH = REPO_ROOT / "data" / "processed" / "features.csv"
WINDOW_SIZE = 100
CORRELATION_THRESHOLD = 0.9


def main() -> None:
    labels = load_manifest(LABELS_PATH)  # same CSV loader, name is generic
    manifest = load_manifest(MANIFEST_PATH)
    relative_path_by_loop = dict(zip(manifest["loop_id"], manifest["relative_path"]))

    confident = labels[labels["derived_label"] != "uncertain"].reset_index(drop=True)
    print(f"{len(confident)}/{len(labels)} windows are confident (not 'uncertain')")

    # cache each file's windows so we don't re-load/re-slice per row
    windows_by_loop: dict[str, list] = {}
    pv_windows, op_windows = [], []
    for _, row in confident.iterrows():
        loop_id = row["loop_id"]
        if loop_id not in windows_by_loop:
            df = load_signal(RAW_DIR / relative_path_by_loop[loop_id])
            windows_by_loop[loop_id] = window_signal(
                df["PV"].to_numpy(float), df["OP"].to_numpy(float), WINDOW_SIZE
            )
        pv_w, op_w = windows_by_loop[loop_id][row["window_index"]]
        pv_windows.append(pv_w)
        op_windows.append(op_w)

    print(f"Extracting tsfel features for {len(pv_windows)} windows...")
    features = extract_features_batch(pv_windows, op_windows, WINDOW_SIZE)
    print(f"Raw feature count: {features.shape[1]}")

    n_nan_rows = features.isna().any(axis=1).sum()
    if n_nan_rows:
        print(f"Warning: {n_nan_rows} windows produced NaN features -- dropping them")
        keep = ~features.isna().any(axis=1)
        features = features[keep].reset_index(drop=True)
        confident = confident[keep.to_numpy()].reset_index(drop=True)

    pruned = prune_correlated_features(features, threshold=CORRELATION_THRESHOLD)
    print(f"Pruned feature count (|corr| <= {CORRELATION_THRESHOLD}): {pruned.shape[1]}")

    result = pd.concat(
        [
            confident[
                ["source_file", "loop_id", "origin_dataset", "folder_label", "derived_label"]
            ].reset_index(drop=True),
            pruned.reset_index(drop=True),
        ],
        axis=1,
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_PATH, index=False)
    print(f"Wrote {result.shape[0]} rows x {pruned.shape[1]} features to {OUT_PATH}")

    print("\n=== Rows by dataset / label ===")
    print(result.groupby(["origin_dataset", "derived_label"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
