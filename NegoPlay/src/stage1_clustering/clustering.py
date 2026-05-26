"""
src/stage1_clustering/clustering.py
=====================================
Cluster players into decision-making profiles using K-Means and HDBSCAN.

Usage:
    from src.stage1_clustering.clustering import cluster_players
    labels, scores = cluster_players(features_df)
"""

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Features used for clustering (must match features.py output columns)
FEATURE_COLS: list[str] = [
    "slam_rate",
    "success_rate",
    "double_rate",
    "avg_level",
    "risk_score",
]

RANDOM_STATE: int = 42


def _scale_features(df: pd.DataFrame) -> np.ndarray:
    """Standardize feature columns to zero mean, unit variance."""
    scaler = StandardScaler()
    return scaler.fit_transform(df[FEATURE_COLS].fillna(0))


def find_best_k(
    features: pd.DataFrame,
    k_range: tuple[int, int] = (2, 6),
) -> dict[int, float]:
    """Run K-Means for each k in range and return silhouette scores.

    Args:
        features: DataFrame from compute_player_features().
        k_range: (min_k, max_k) inclusive.

    Returns:
        Dict mapping k → silhouette score.
    """
    X = _scale_features(features)
    scores: dict[int, float] = {}

    for k in range(k_range[0], k_range[1] + 1):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        scores[k] = round(float(score), 4)
        logger.info("  k=%d  silhouette=%.4f", k, score)

    best_k = max(scores, key=scores.__getitem__)
    logger.info("Best k=%d (silhouette=%.4f)", best_k, scores[best_k])
    return scores


def cluster_players(
    features: pd.DataFrame,
    k: int | None = None,
    k_range: tuple[int, int] = (2, 6),
) -> tuple[pd.DataFrame, dict]:
    """Cluster players into decision-making profiles.

    Runs K-Means (choosing best k by silhouette if k is None).
    Optionally runs HDBSCAN as a validation check.

    Args:
        features: DataFrame from compute_player_features().
        k: Number of clusters. If None, best k is chosen automatically.
        k_range: Range to search if k is None.

    Returns:
        Tuple of:
          - features DataFrame with added 'cluster' column
          - results dict with keys: best_k, silhouette_scores,
            kmeans_silhouette, hdbscan_n_clusters (if available)
    """
    logger.info("Starting clustering on %d players ...", len(features))
    X = _scale_features(features)

    # ── Step 1: Choose best k ───────────────────────────────────────────────
    silhouette_scores = find_best_k(features, k_range)
    best_k = k if k is not None else max(silhouette_scores, key=silhouette_scores.__getitem__)

    # ── Step 2: Final K-Means with best k ───────────────────────────────────
    km = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=20)
    kmeans_labels = km.fit_predict(X)
    kmeans_sil = silhouette_score(X, kmeans_labels)

    logger.info(
        "K-Means (k=%d): silhouette=%.4f", best_k, kmeans_sil
    )

    # ── Step 3: HDBSCAN validation ──────────────────────────────────────────
    hdbscan_n = None
    try:
        from sklearn.cluster import HDBSCAN  # available sklearn >= 1.3

        hdb = HDBSCAN(min_cluster_size=max(5, len(features) // 20))
        hdbscan_labels = hdb.fit_predict(X)
        hdbscan_n = len(set(hdbscan_labels) - {-1})  # exclude noise (-1)
        logger.info(
            "HDBSCAN found %d clusters (noise points: %d)",
            hdbscan_n,
            (hdbscan_labels == -1).sum(),
        )
    except ImportError:
        logger.warning("HDBSCAN not available (needs scikit-learn >= 1.3). Skipping.")
    except Exception as exc:
        logger.warning("HDBSCAN failed: %s. Skipping.", exc)

    # ── Step 4: Attach labels to features DataFrame ─────────────────────────
    result = features.copy()
    result["cluster"] = kmeans_labels

    # ── Step 5: Log cluster summary ─────────────────────────────────────────
    summary = (
        result.groupby("cluster")[FEATURE_COLS]
        .mean()
        .round(4)
    )
    logger.info("Cluster centroids:\n%s", summary.to_string())

    results_dict = {
        "best_k": best_k,
        "silhouette_scores": silhouette_scores,
        "kmeans_silhouette": round(kmeans_sil, 4),
        "hdbscan_n_clusters": hdbscan_n,
        "cluster_sizes": result["cluster"].value_counts().to_dict(),
    }

    return result, results_dict
