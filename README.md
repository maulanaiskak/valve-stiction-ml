# valve-stiction-ml

Training pipeline for a valve stiction detection classifier — the "bounded ML comparison" component of a larger [distributed IoT fault-detection pipeline](docs/ML_PLAN.md) project. Rebuilds the ML side of an undergraduate thesis (a custom Random Forest on hand-rolled time-series features, trained on file-level labels) as a reproducible, tested, evidence-driven training project.

The full design rationale — including several real bugs found and fixed along the way, and honest documentation of what didn't work — lives in **[docs/ML_PLAN.md](docs/ML_PLAN.md)**. This README is the practical "how to run it" companion.

## Why this exists

The original thesis assigned stiction labels **per file**, then propagated that label to every 100-sample window cut from it — a file labeled "stiction" doesn't necessarily stick for its entire recording, so many windows were mislabeled. Retraining against those labels would just teach a model to reproduce that mistake.

Instead: a training-free, unsupervised classic detector (ellipse-fit + Kano pattern check, `src/valve_stiction_ml/classic.py`) labels every window on its own signal shape, using process-control theory instead of learned parameters. The RF is trained to cheaply approximate *that*, not the old file-level labels — which are kept around only as a sanity check, never as a training target.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Requires the original thesis repo's `Data/` folder (ISDB + SACAC CSVs) — not included here. Point `scripts/import_thesis_data.py --source <path>` at it, or edit `DEFAULT_SOURCE` in that script.

## Reproducing the full pipeline

Each step writes to `data/processed/` or `models/`/`reports/` (all gitignored — regenerate, don't expect them checked in).

```bash
# 1. Copy thesis CSVs into data/raw/, build data/processed/manifest.csv
python scripts/import_thesis_data.py

# 1b. Optional: pull in SACAC files present on the official portal
#     (sacac.org.za) but missing from the thesis repo's copy -- see the
#     script's docstring for which ones are actually usable and why
python scripts/download_additional_sacac.py

# 2. Run the classic detector (ellipse-fit + Kano) over every window,
#    tune ellipse_threshold on ISDB only -> data/processed/window_labels.csv
python scripts/label_windows.py

# 3. Extract tsfel features for confident (non-"uncertain") windows,
#    prune correlated features -> data/processed/features.csv
python scripts/extract_features.py

# 4. Train RF (RandomizedSearchCV + StratifiedGroupKFold on ISDB),
#    evaluate on held-out SACAC -> models/<date>_<git-hash>/model.joblib
python -m valve_stiction_ml.train

# 5. Optional: sanity-check the feature pipeline via unsupervised
#    clustering -> reports/cluster_sanity_check.png
python scripts/cluster_sanity_check.py
```

Run tests with `pytest`. A handful (`test_classic_real_fixtures.py`) are skipped until step 1 has been run, since they validate against real SACAC files.

## Using a trained model

```python
from pathlib import Path
from valve_stiction_ml.inference import load_artifact, predict_window

artifact = load_artifact(Path("models/<date>_<hash>/model.joblib"))
result = predict_window(artifact, pv_window, op_window)  # raw, non-normalized arrays
# {"label": "yes" | "no", "probability": 0.52}
```

`pv_window`/`op_window` must be raw (non-normalized) arrays of exactly `artifact["window_size"]` samples — normalization and feature extraction happen inside `predict_window`, matching exactly what training did. The artifact is self-describing (feature list, window size, normalization, decision threshold, git commit, full metrics all travel with the model file), specifically so a caller never has to hardcode or guess any of that separately — which is what caused a real bug in the original thesis code (`subscribe.py`'s feature list could silently drift out of sync with the model it loaded).

## Results (current baseline)

|  | ISDB (StratifiedGroupKFold CV) | SACAC (held-out test) |
|---|---|---|
| Precision | 0.733 | 0.866 |
| Recall | 0.839 | 0.522 |
| F1 | 0.782 | 0.651 |
| ROC-AUC | 0.944 | 0.865 |
| PR-AUC | 0.809 | 0.788 |

Read as: *does a cheap RF approximate the classic detector well* — not *does this detect real-world stiction with X% accuracy*. Precision/recall trade off differently between ISDB and SACAC (SACAC: fewer false positives, more misses) — an honest generalization gap between two independently-sourced benchmark corpora, not hidden or averaged away.

## Known limitations (see ML_PLAN.md for full detail)

- **~45-49% of windows are "uncertain"** and excluded from training — the classic detector's two components (ellipse-fit, Kano) are conservative by design and don't always agree. Investigated and kept this way deliberately (§13).
- **Clustering sanity check came back negative** (§10.1) — unsupervised structure in the feature space doesn't cleanly separate stiction from non-stiction; the RF's supervised result isn't independently corroborated by it.
- **Small dataset**: 2232 confident windows total (427 positive) after all filtering, from 115 source files.
- **Window size (100 samples) was investigated, not assumed** — tried larger windows expecting better results; they were empirically worse given how the Kano heuristic aggregates strokes (§13).
- Neither the classic detector nor the RF is claimed as ground truth — even literature methods this compares against top out around 85-90% on the ISDB benchmark.

## Project structure

```
src/valve_stiction_ml/
  dataset.py       loading, windowing, manifest
  classic.py        ellipse-fit + Kano detector (training-free label source)
  features.py         per-window normalization + tsfel extraction
  models.py             RF training (RandomizedSearchCV + StratifiedGroupKFold)
  evaluate.py              metrics, sanity-check reporting
  train.py                   CLI entrypoint, writes the model artifact
  inference.py                  load a trained artifact, score one window
scripts/            one-off pipeline steps (see "Reproducing the full pipeline")
configs/default.yaml  window size, classic-detector threshold, RF search space
docs/ML_PLAN.md     full design rationale, findings, and decision log
```
