"""Loading, windowing, and manifest handling.

See docs/ML_PLAN.md §5-6. `folder_label` here is the thesis-era file-level
yes/no label — kept only as a sanity-check signal (ML_PLAN.md §2, §4), never
as a training target. The training target is derived later by the classic
detector (classic.py), per-window.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["PV", "OP"]


@dataclass(frozen=True)
class Window:
    pv: np.ndarray
    op: np.ndarray
    source_file: str
    loop_id: str
    origin_dataset: str
    folder_label: str
    window_index: int


def load_signal(csv_path: Path) -> pd.DataFrame:
    """Load a thesis-data CSV and return just the PV/OP columns.

    Column casing/order/extra columns (Time, SP, Error, ts...) vary across
    files but PV/OP are consistently named that way across the whole
    ISDB + SACAC corpus (verified against all source files).
    """
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing required columns {missing}")
    return df[REQUIRED_COLUMNS].astype(float).reset_index(drop=True)


def window_signal(
    pv: np.ndarray, op: np.ndarray, window_size: int, stride: int | None = None
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Slice PV/OP arrays into fixed-size windows.

    Defaults to non-overlapping windows (stride == window_size), matching
    the thesis's training-time windowing (not its inference-time sliding
    window, which used stride 1 — see ML_PLAN.md §13 for why we don't
    inherit that here: overlapping windows from the same file would leak
    across GroupKFold folds even with grouping, since adjacent windows are
    near-duplicates).
    """
    if stride is None:
        stride = window_size
    n = len(pv)
    windows = []
    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        windows.append((pv[start:end], op[start:end]))
    return windows


def iter_windows(
    manifest: pd.DataFrame, raw_dir: Path, window_size: int, stride: int | None = None
) -> list[Window]:
    """Load every file in the manifest and slice it into windows."""
    out: list[Window] = []
    for _, row in manifest.iterrows():
        df = load_signal(raw_dir / row["relative_path"])
        slices = window_signal(
            df["PV"].to_numpy(), df["OP"].to_numpy(), window_size, stride
        )
        for i, (pv_w, op_w) in enumerate(slices):
            out.append(
                Window(
                    pv=pv_w,
                    op=op_w,
                    source_file=row["source_file"],
                    loop_id=row["loop_id"],
                    origin_dataset=row["origin_dataset"],
                    folder_label=row["folder_label"],
                    window_index=i,
                )
            )
    return out


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    return pd.read_csv(manifest_path)


def cap_windows_per_loop(
    df: pd.DataFrame,
    max_per_loop: int,
    loop_column: str = "loop_id",
    random_state: int = 42,
) -> pd.DataFrame:
    """Randomly subsample each loop down to at most max_per_loop rows.

    File lengths in this corpus range from 200 to 42,512 samples, so
    fixed-size non-overlapping windowing produces wildly uneven per-loop
    window counts -- checked on the real ISDB training set and found the
    top 5 of 60 loops contribute 72% of all windows (top 3 alone, all from
    the same "BAS" source, contribute 62%). GroupKFold/StratifiedGroupKFold
    already prevent this from leaking across train/test, but it's still a
    real training-data representativeness problem: the model could
    disproportionately learn whatever those specific loops look like
    rather than a generalizable pattern. This caps that influence without
    dropping any loop entirely.
    """
    rng = np.random.default_rng(random_state)
    keep_indices: list = []
    for _, group in df.groupby(loop_column):
        n = min(len(group), max_per_loop)
        keep_indices.extend(rng.choice(group.index.to_numpy(), size=n, replace=False))
    return df.loc[keep_indices].reset_index(drop=True)
