"""Download SACAC files present on the official portal but missing from the
thesis repo's copy, and append them to the manifest.

Checked the official portal (sacac.org.za/resources) against what
scripts/import_thesis_data.py already imported: 10 files were missing. Of
those, only the 6 "unknown-*" ones (root cause undetermined by the original
researchers -- folder_label="unknown", never forced into yes/no) turned out
to be usable: real PV/OP columns, just semicolon-delimited instead of the
comma-delimited convention the thesis repo's copies use. The other 4 aren't
compatible with this pipeline at all:
  - 3 "plantwide-*" files: multi-tag plant-wide files (TI1;TI2;TI3;...,
    temperature/level indicators across a whole plant), not a single
    loop's PV/OP pair, and no clear way to pick which two tags form "the"
    valve loop without domain documentation we don't have.
  - "quantisation-T-chemicals-Thornhill-2003.csv": PV only, no OP column
    at all -- can't compute anything without both.

Usage:
    python scripts/download_additional_sacac.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
MANIFEST_PATH = REPO_ROOT / "data" / "processed" / "manifest.csv"
PORTAL_BASE = "https://sacac.org.za/wp-content/uploads/2023/07"

# folder_label="unknown": these were never confidently root-caused by the
# original researchers, so we don't force them into yes/no either --
# derive_label (classic.py) still evaluates each window on its own signal,
# same as everything else, with no dependency on this field.
UNKNOWN_FILES = [
    "unknown-F-chemicals-thornhill-2003",
    "unknown-F-minerals-bauer-2017",
    "unknown-F-paper-horch-2003",
    "unknown-L-oilgas-thornhill-2002",
    "unknown-P-oilgas-thornhill-2007-1",
    "unknown-P-oilgas-thornhill-2007-2",
]


def main() -> None:
    dest_dir = RAW_DIR / "SACAC" / "unknown"
    dest_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(MANIFEST_PATH)
    new_rows = []

    for name in UNKNOWN_FILES:
        url = f"{PORTAL_BASE}/{name}.csv"
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        resp.raise_for_status()

        # Portal serves these semicolon-delimited; normalize to the same
        # comma-delimited convention as the rest of data/raw/ so
        # dataset.load_signal doesn't need to guess a delimiter per file.
        from io import StringIO

        df = pd.read_csv(StringIO(resp.text), sep=";")
        missing = [c for c in ["PV", "OP"] if c not in df.columns]
        if missing:
            raise ValueError(f"{name} missing columns {missing} -- portal format changed?")

        dest_path = dest_dir / f"{name}.csv"
        df.to_csv(dest_path, index=False)

        relative_path = f"SACAC/unknown/{name}.csv"
        loop_id = f"SACAC__unknown__{name}"
        new_rows.append(
            {
                "source_file": f"{name}.csv",
                "relative_path": relative_path,
                "origin_dataset": "SACAC",
                "folder_label": "unknown",
                "loop_id": loop_id,
            }
        )
        print(f"Imported {name} ({len(df)} rows)")

    manifest = pd.concat([manifest, pd.DataFrame(new_rows)], ignore_index=True)
    manifest.to_csv(MANIFEST_PATH, index=False)
    print(f"\nAppended {len(new_rows)} rows to {MANIFEST_PATH}")
    print(f"Manifest now has {len(manifest)} total files")


if __name__ == "__main__":
    main()
