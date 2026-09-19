"""Milestone 5: train RF against classic-detector labels.

StratifiedGroupKFold CV (group=loop_id) on ISDB for model selection and CV metrics;
final, untouched evaluation on SACAC. See ML_PLAN.md §9-10.

Usage:
    python -m valve_stiction_ml.train
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

from valve_stiction_ml.dataset import cap_windows_per_loop
from valve_stiction_ml.evaluate import compute_metrics, sanity_check_agreement
from valve_stiction_ml.models import train_random_forest

REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = REPO_ROOT / "data" / "processed" / "features.csv"
CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"
MODELS_DIR = REPO_ROOT / "models"
REPORTS_DIR = REPO_ROOT / "reports"

METADATA_COLUMNS = ["source_file", "loop_id", "origin_dataset", "folder_label", "derived_label"]


def git_short_hash() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    df = pd.read_csv(FEATURES_PATH)
    feature_names = [c for c in df.columns if c not in METADATA_COLUMNS]

    isdb_df = df[df["origin_dataset"] == "ISDB"].reset_index(drop=True)
    sacac_df = df[df["origin_dataset"] != "ISDB"].reset_index(drop=True)

    max_per_loop = config.get("training", {}).get("max_windows_per_loop")
    if max_per_loop:
        before = len(isdb_df)
        isdb_df = cap_windows_per_loop(isdb_df, max_per_loop)
        print(
            f"Capped ISDB windows at {max_per_loop}/loop: {before} -> {len(isdb_df)} windows "
            "(training-data representativeness fix, see configs/default.yaml)"
        )

    X_train = isdb_df[feature_names].to_numpy()
    y_train = (isdb_df["derived_label"] == "yes").astype(int).to_numpy()
    groups_train = isdb_df["loop_id"].to_numpy()
    folder_train = isdb_df["folder_label"].to_numpy()

    X_test = sacac_df[feature_names].to_numpy()
    y_test = (sacac_df["derived_label"] == "yes").astype(int).to_numpy()
    folder_test = sacac_df["folder_label"].to_numpy()

    print(f"Training on {len(X_train)} ISDB windows, testing on {len(X_test)} SACAC windows")
    print(f"ISDB positive rate: {y_train.mean():.3f}  SACAC positive rate: {y_test.mean():.3f}")

    n_splits = config["cross_validation"]["n_splits"]
    print(f"\nRunning RandomizedSearchCV (StratifiedGroupKFold, n_splits={n_splits})...")
    search = train_random_forest(X_train, y_train, groups_train, n_splits=n_splits)
    print(f"Best params: {search.best_params_}")
    print(f"Best CV average_precision: {search.best_score_:.3f}")

    # Proper out-of-fold predictions with the *chosen* hyperparameters, for
    # the full metric suite (search.best_score_ above is only the tuning
    # objective, average_precision).
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_proba = cross_val_predict(
        search.best_estimator_, X_train, y_train, groups=groups_train,
        cv=cv, method="predict_proba", n_jobs=-1,
    )[:, 1]
    oof_pred = (oof_proba >= 0.5).astype(int)

    cv_metrics = compute_metrics(y_train, oof_pred, oof_proba)
    cv_agreement = sanity_check_agreement(
        np.where(oof_pred == 1, "yes", "no"), folder_train
    )
    print(f"\nISDB StratifiedGroupKFold CV metrics: {cv_metrics}")
    print(f"ISDB sanity-check agreement with folder_label: {cv_agreement}")

    # search.best_estimator_ was refit on all of X_train/y_train (refit=True)
    final_model = search.best_estimator_
    test_proba = final_model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= 0.5).astype(int)

    test_metrics = compute_metrics(y_test, test_pred, test_proba)
    test_agreement = sanity_check_agreement(
        np.where(test_pred == 1, "yes", "no"), folder_test
    )
    print(f"\nSACAC held-out test metrics: {test_metrics}")
    print(f"SACAC sanity-check agreement with folder_label: {test_agreement}")

    trained_at = datetime.now(timezone.utc).isoformat()
    git_hash = git_short_hash()
    artifact = {
        "model": final_model,
        "feature_names": feature_names,
        "window_size": config["window"]["size"],
        "normalization": config["normalization"],
        "label_source": "classic_detector_v1",
        "max_windows_per_loop": max_per_loop,
        "classic_detector_threshold": config["classic_detector"]["ellipse_stiction_threshold"],
        "predict_threshold": 0.5,
        "sklearn_version": sklearn.__version__,
        "git_commit_hash": git_hash,
        "best_params": search.best_params_,
        "cv_metrics": cv_metrics,
        "cv_sanity_check_agreement": cv_agreement,
        "test_metrics": test_metrics,
        "test_sanity_check_agreement": test_agreement,
        "trained_at": trained_at,
    }

    run_dir = MODELS_DIR / f"{datetime.now().strftime('%Y%m%d')}_{git_hash}"
    run_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, run_dir / "model.joblib")
    print(f"\nSaved model artifact to {run_dir / 'model.joblib'}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {k: v for k, v in artifact.items() if k not in ("model",)}
    report_path = REPORTS_DIR / f"{datetime.now().strftime('%Y%m%d')}_{git_hash}.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
