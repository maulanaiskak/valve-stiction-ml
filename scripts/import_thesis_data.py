"""One-off: copy thesis-repo CSVs into data/raw/ and build data/processed/manifest.csv.

folder_label here is the thesis-era file-level yes/no label — imported only
as a sanity-check signal, never as a training target. See ML_PLAN.md §2, §5.

Usage:
    python scripts/import_thesis_data.py [--source PATH]
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SOURCE = Path(
    "/Users/maulanaiskak/Documents/Kuliah/Tugas Akhir/File TA/Github Tugas Akhir"
    "/Maulana Iskak/Coding/Data"
)

# (subfolder name under DEFAULT_SOURCE, origin_dataset tag)
DATASETS = [
    ("Data Latih ISDB", "ISDB"),
    ("Data Validasi SACAC", "SACAC"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--raw-dir", type=Path, default=REPO_ROOT / "data" / "raw"
    )
    parser.add_argument(
        "--manifest", type=Path, default=REPO_ROOT / "data" / "processed" / "manifest.csv"
    )
    args = parser.parse_args()

    if not args.source.exists():
        raise SystemExit(f"Source data not found at {args.source}")

    rows = []
    for subfolder, origin_dataset in DATASETS:
        dataset_root = args.source / subfolder
        for folder_label in ("yes", "no"):
            label_root = dataset_root / folder_label
            if not label_root.exists():
                continue
            for csv_path in sorted(label_root.rglob("*.csv")):
                rel_within_label = csv_path.relative_to(label_root)
                relative_path = Path(origin_dataset) / folder_label / rel_within_label
                dest = args.raw_dir / relative_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(csv_path, dest)

                loop_id = str(relative_path.with_suffix("")).replace("/", "__")
                rows.append(
                    {
                        "source_file": csv_path.name,
                        "relative_path": str(relative_path),
                        "origin_dataset": origin_dataset,
                        "folder_label": folder_label,
                        "loop_id": loop_id,
                    }
                )

    manifest = pd.DataFrame(rows)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.manifest, index=False)

    print(f"Imported {len(manifest)} files into {args.raw_dir}")
    print(manifest.groupby(["origin_dataset", "folder_label"]).size())
    print(f"Manifest written to {args.manifest}")


if __name__ == "__main__":
    main()
