"""Milestone 7 (ML_PLAN.md §10.1): clustering as a feature-pipeline sanity
check, not a modeling method.

Runs KMeans and GMM (k=2) on the same standardized tsfel features used to
train the RF, independent of any label, then checks whether the classic
detector's yes/no windows fall into separate clusters. Good separation is
evidence the features encode stiction-relevant structure; poor separation
is a warning sign about the feature pipeline or the classic detector's
threshold -- diagnostic either way, never a reason to relabel anything.

Usage:
    python scripts/cluster_sanity_check.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = REPO_ROOT / "data" / "processed" / "features.csv"
REPORTS_DIR = REPO_ROOT / "reports"
METADATA_COLUMNS = ["source_file", "loop_id", "origin_dataset", "folder_label", "derived_label"]


def main() -> None:
    df = pd.read_csv(FEATURES_PATH)
    feature_names = [c for c in df.columns if c not in METADATA_COLUMNS]
    labels = (df["derived_label"] == "yes").astype(int).to_numpy()

    X = StandardScaler().fit_transform(df[feature_names])

    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42).fit(X)
    gmm = GaussianMixture(n_components=2, random_state=42).fit(X)
    gmm_clusters = gmm.predict(X)

    for name, clusters in [("KMeans", kmeans.labels_), ("GMM", gmm_clusters)]:
        ari = adjusted_rand_score(labels, clusters)
        nmi = normalized_mutual_info_score(labels, clusters)
        print(f"\n=== {name} vs classic-detector label ===")
        print(f"Adjusted Rand Index: {ari:.3f}  Normalized Mutual Info: {nmi:.3f}")
        ct = pd.crosstab(
            pd.Series(clusters, name="cluster"),
            pd.Series(df["derived_label"].to_numpy(), name="derived_label"),
        )
        print(ct)

    # PCA projection for visual inspection
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)
    print(f"\nPCA: {pca.explained_variance_ratio_.sum():.1%} variance in 2 components")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (title, color_by) in zip(
        axes,
        [
            ("Colored by classic-detector label", labels),
            ("Colored by KMeans cluster", kmeans.labels_),
            ("Colored by GMM cluster", gmm_clusters),
        ],
    ):
        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=color_by, cmap="coolwarm", alpha=0.5, s=10)
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    plt.tight_layout()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "cluster_sanity_check.png"
    plt.savefig(out_path, dpi=100)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
