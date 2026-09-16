"""RF training via RandomizedSearchCV + StratifiedGroupKFold, scored on ISDB only.

See docs/ML_PLAN.md §9. class_weight='balanced' handles the ~19% positive
rate; scoring is average_precision (PR-AUC) rather than accuracy or a
0.5-threshold metric, since it's threshold-independent and appropriate for
imbalanced data -- matches ML_PLAN.md §10's "never accuracy alone".
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold

RF_PARAM_DISTRIBUTIONS = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [None, 5, 10, 20],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": ["sqrt", "log2", 0.5],
}


def train_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    n_iter: int = 30,
    random_state: int = 42,
) -> RandomizedSearchCV:
    """Fit a RandomForestClassifier via RandomizedSearchCV, scored with
    StratifiedGroupKFold (group = loop_id) so no window from the same
    source file appears in both a fold's train and validation split, while
    still balancing the positive rate across folds.

    Plain GroupKFold doesn't consider class labels at all -- checked on
    the real ISDB data and found one fold with *zero* positive windows and
    two others with only 3-5, while two folds had 113+ each (positives are
    concentrated in a handful of loops). That makes per-fold precision/
    recall/PR-AUC meaningless for the empty-or-near-empty folds, and
    biases model selection toward whatever the two positive-heavy folds
    say. StratifiedGroupKFold fixes this: 45-65 positives per fold on the
    same data, while still respecting grouping.
    """
    base = RandomForestClassifier(class_weight="balanced", random_state=random_state)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        base,
        RF_PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        scoring="average_precision",
        cv=cv,
        random_state=random_state,
        refit=True,
        n_jobs=-1,
    )
    search.fit(X, y, groups=groups)
    return search
