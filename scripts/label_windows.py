"""Milestone 3: run the classic detector across ISDB + SACAC.

Tunes ellipse_threshold on ISDB only (maximizing agreement with
kano_pattern_check, which has no free parameter -- see below), then labels
every window in both datasets and reports the label distribution, the
"uncertain" rate, and agreement with the old folder labels as a sanity
check (ML_PLAN.md §4, §10).

Usage:
    python scripts/label_windows.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from valve_stiction_ml.classic import (
    ellipse_stiction_index,
    has_sufficient_activity,
    kano_pattern_check,
    label_window,
)
from valve_stiction_ml.dataset import iter_windows, load_manifest, load_signal

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
MANIFEST_PATH = REPO_ROOT / "data" / "processed" / "manifest.csv"
MIN_RELATIVE_ACTIVITY = 0.15


def zscore(x: np.ndarray) -> np.ndarray:
    std = x.std()
    return (x - x.mean()) / std if std > 0 else x - x.mean()


def compute_reference_op_std(manifest: pd.DataFrame) -> dict[str, float]:
    """Full-file OP std per loop_id, used as the activity-guard reference."""
    out = {}
    for _, row in manifest.iterrows():
        df = load_signal(RAW_DIR / row["relative_path"])
        out[row["loop_id"]] = float(df["OP"].std())
    return out


def tune_ellipse_threshold(
    windows, reference_std: dict[str, float], percentile: float = 10.0
) -> float:
    """Set ellipse_threshold as a low percentile of the ellipse_index
    distribution on ISDB's "active" windows (activity guard passed).

    Tried first: searching for the threshold that maximizes agreement with
    kano_pattern_check, or F1 against it. Both failed for the same reason --
    kano's positive rate is only ~15%, so "always predict False" already
    scores ~85% raw agreement, and the search degenerately converged on a
    threshold near the 99th percentile that made ellipse_verdict almost
    always False (observed: 1 "yes" out of 3029 ISDB windows). F1 against
    kano as the reference didn't fare better (best F1 ~0.26, precision
    ~0.15) -- ellipse_stiction_index and kano_pattern_check are *supposed*
    to disagree often (ML_PLAN.md §3: they catch different failure modes),
    so optimizing one against the other doesn't make sense.

    What ellipse_stiction_index should actually do here is act as a
    fine-grained backstop beyond has_sufficient_activity's per-file check:
    drop windows with negligible width even though their file passed the
    file-level activity guard. A low, round percentile is a defensible,
    conservative choice for that -- and deliberately NOT chosen by
    searching for whatever value maximizes agreement with the old folder
    labels either, which would quietly reintroduce exactly the thing this
    whole redesign exists to avoid trusting (ML_PLAN.md §2, §4). The
    resulting agreement with folder labels is reported below as a sanity
    check only, computed after this choice, not used to make it.
    """
    idx = []
    for w in windows:
        if w.origin_dataset != "ISDB":
            continue
        ref_std = reference_std[w.loop_id]
        if not has_sufficient_activity(w.op, ref_std, MIN_RELATIVE_ACTIVITY):
            continue
        pv_z, op_z = zscore(w.pv), zscore(w.op)
        idx.append(ellipse_stiction_index(pv_z, op_z))

    threshold = float(np.percentile(idx, percentile))
    print(
        f"ellipse_threshold={threshold:.4f} "
        f"({percentile:.0f}th percentile of {len(idx)} active ISDB windows' ellipse_index)"
    )
    return threshold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "window_labels.csv",
    )
    args = parser.parse_args()
    out_path = args.out

    manifest = load_manifest(MANIFEST_PATH)
    print(f"window_size={args.window_size}")
    print(f"Computing per-file reference OP std for {len(manifest)} files...")
    reference_std = compute_reference_op_std(manifest)

    windows = iter_windows(manifest, RAW_DIR, args.window_size)
    print(f"{len(windows)} total windows")

    ellipse_threshold = tune_ellipse_threshold(windows, reference_std)

    rows = []
    for w in windows:
        ref_std = reference_std[w.loop_id]
        active = has_sufficient_activity(w.op, ref_std, MIN_RELATIVE_ACTIVITY)
        derived = label_window(
            w.pv, w.op, ellipse_threshold, reference_op_std=ref_std,
            min_relative_activity=MIN_RELATIVE_ACTIVITY,
        )
        pv_z, op_z = zscore(w.pv), zscore(w.op)
        rows.append(
            {
                "source_file": w.source_file,
                "loop_id": w.loop_id,
                "origin_dataset": w.origin_dataset,
                "folder_label": w.folder_label,
                "window_index": w.window_index,
                "ellipse_index": ellipse_stiction_index(pv_z, op_z),
                "kano_verdict": kano_pattern_check(pv_z, op_z),
                "has_activity": active,
                "derived_label": derived,
            }
        )

    result = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"Wrote {len(result)} rows to {out_path}")

    print("\n=== Label distribution by dataset ===")
    print(
        result.groupby(["origin_dataset", "derived_label"]).size().unstack(fill_value=0)
    )

    print("\n=== Uncertain rate by dataset ===")
    print(result.groupby("origin_dataset")["derived_label"].apply(
        lambda s: (s == "uncertain").mean()
    ))

    print("\n=== Sanity check: derived_label vs old folder_label ===")
    print("(agreement computed only on non-uncertain windows with a known folder_label --")
    print(" 'unknown' folder_label windows, e.g. SACAC's own undetermined-root-cause files,")
    print(" have nothing to compare against and are excluded, not counted as disagreement)")
    confident = result[
        (result["derived_label"] != "uncertain") & (result["folder_label"] != "unknown")
    ]
    agree = (confident["derived_label"] == confident["folder_label"]).mean()
    print(f"Overall agreement: {agree:.3f} ({len(confident)} confident windows)")
    for dataset, group in confident.groupby("origin_dataset"):
        a = (group["derived_label"] == group["folder_label"]).mean()
        print(f"  {dataset}: {a:.3f} ({len(group)} confident windows)")


if __name__ == "__main__":
    main()
