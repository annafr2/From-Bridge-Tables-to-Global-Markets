"""
src/stage1_clustering/clustering.py
=====================================
Cluster players into decision-making profiles using K-Means and HDBSCAN.

Pipeline (updated May 2026 per Dr. Rami's review):
    Raw features
        → StandardScaler          (normalize to mean=0, std=1)
        → PCA (3 components)      (remove correlated noise, reduce dimensions)
        → K-Means / HDBSCAN       (cluster in clean PCA space)
        → t-SNE                   (2D visualization only — NOT for clustering)

Outputs:
    • cluster labels per player
    • silhouette scores
    • Excel file with 4 sheets: scaled features, PCA components,
      PCA loadings, t-SNE coordinates

Usage:
    from src.stage1_clustering.clustering import cluster_players, export_preprocessing
    result_df, info = cluster_players(features_df)
    export_preprocessing(features_df, path="results/preprocessing.xlsx")
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Features used for clustering — 10 behavioural axes
# (outcome features from contract column + process features from bidding column)
FEATURE_COLS: list[str] = [
    # — Outcome features (declarer only) —
    "slam_rate",            # Slam Hunter signal           cv=0.40 ✓
    "double_rate",          # how often opponents doubled  cv=0.41 ✓
    "nt_rate",              # NT Specialist signal         cv=0.16 ✓
    "partscore_rate",       # Insurance Player signal      cv=0.09 ✓
    # — Process features (any seat, from full bidding) —
    "opening_rate",         # how often they open          cv=0.13 ✓
    "preempt_rate",         # weak preemptive openings     cv=0.39 ✓
    "intervention_rate",    # bidding into opp's auction   cv=0.16 ✓
    "penalty_double_rate",  # Fighter signal               cv=0.26 ✓
    # REMOVED (cv < 0.09 — near-zero variance, add noise without signal):
    #   avg_level          cv=0.04  (everyone bids ~level 3.0-3.7)
    #   avg_bids_per_board cv=0.08  (everyone makes ~1 bid/board)
]

RANDOM_STATE: int = 42
N_PCA_COMPONENTS: int = 3       # top PCA components to keep
TSNE_PERPLEXITY: int = 30       # t-SNE perplexity (typical: 5–50)


# ── Step 1 — Scale ─────────────────────────────────────────────────────────────

def scale_features(df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """Standardize feature columns to mean=0, std=1.

    Args:
        df: DataFrame with FEATURE_COLS columns.

    Returns:
        Tuple of (scaled array shape [n_players, n_features], fitted scaler).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURE_COLS].fillna(0))
    return X_scaled, scaler


# ── Step 2 — PCA ───────────────────────────────────────────────────────────────

def run_pca(
    X_scaled: np.ndarray,
    n_components: int = N_PCA_COMPONENTS,
) -> tuple[np.ndarray, PCA]:
    """Reduce scaled features to top PCA components.

    Why PCA before K-Means:
    - Some features are correlated (e.g. slam_rate and avg_level both increase
      together). PCA merges them into one independent axis.
    - K-Means in 10 dimensions suffers from the 'curse of dimensionality' —
      distances become less meaningful. 3 clean components = better clusters.
    - Low-variance components (noise) are discarded.

    Args:
        X_scaled: Output of scale_features().
        n_components: Number of PCA axes to keep (default 3).

    Returns:
        Tuple of (PCA-reduced array shape [n_players, n_components], fitted PCA).
    """
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    for i, (var, cum) in enumerate(zip(explained, cumulative), 1):
        logger.info("  PC%d: %.1f%% variance  (cumulative: %.1f%%)", i, var * 100, cum * 100)

    logger.info(
        "PCA: %d components explain %.1f%% of total variance",
        n_components, cumulative[-1] * 100,
    )
    return X_pca, pca


# ── Step 3 — t-SNE (visualization only) ───────────────────────────────────────

def run_tsne(
    X_scaled: np.ndarray,
    perplexity: int = TSNE_PERPLEXITY,
) -> np.ndarray:
    """Compute 2D t-SNE coordinates for visualization.

    ⚠️  IMPORTANT: t-SNE is for VISUALIZATION ONLY.
        It always creates visual blobs regardless of whether true clusters exist.
        Never use t-SNE output as evidence of clustering.
        Always run K-Means or HDBSCAN for the statistical claim.

    Args:
        X_scaled: Scaled feature array (use scaled, not PCA, for t-SNE —
                  t-SNE works better in higher dimensions than K-Means does).
        perplexity: t-SNE perplexity parameter (typical range 5–50).

    Returns:
        Array shape [n_players, 2] with (tsne_x, tsne_y) coordinates.
    """
    logger.info("Running t-SNE (perplexity=%d) — visualization only ...", perplexity)
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=RANDOM_STATE,
        max_iter=1000,
    )
    return tsne.fit_transform(X_scaled)


# ── Step 4 — K-Means on PCA space ─────────────────────────────────────────────

def find_best_k(
    X_pca: np.ndarray,
    k_range: tuple[int, int] = (2, 6),
) -> dict[int, float]:
    """Run K-Means for each k on PCA-reduced data, return silhouette scores.

    Args:
        X_pca: PCA-reduced array from run_pca().
        k_range: (min_k, max_k) inclusive.

    Returns:
        Dict mapping k → silhouette score.
    """
    scores: dict[int, float] = {}
    for k in range(k_range[0], k_range[1] + 1):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labels = km.fit_predict(X_pca)
        score = silhouette_score(X_pca, labels)
        scores[k] = round(float(score), 4)
        logger.info("  K-Means (PCA space) k=%d  silhouette=%.4f", k, score)

    best_k = max(scores, key=scores.__getitem__)
    logger.info("Best k=%d (silhouette=%.4f)", best_k, scores[best_k])
    return scores


def cluster_players(
    features: pd.DataFrame,
    k: int | None = None,
    k_range: tuple[int, int] = (2, 6),
    n_pca_components: int = N_PCA_COMPONENTS,
) -> tuple[pd.DataFrame, dict]:
    """Full clustering pipeline: Scale → PCA → K-Means → HDBSCAN validation.

    Args:
        features:          DataFrame from compute_player_features().
        k:                 Number of clusters (None = auto-select by silhouette).
        k_range:           Range to search if k is None.
        n_pca_components:  Number of PCA components before K-Means.

    Returns:
        Tuple of:
          - features DataFrame with added columns:
              'cluster', 'pc1', 'pc2', 'pc3', 'tsne_x', 'tsne_y'
          - results dict with keys:
              best_k, silhouette_scores, kmeans_silhouette,
              hdbscan_n_clusters, pca_explained_variance
    """
    logger.info("=== Clustering pipeline: %d players, %d features ===",
                len(features), len(FEATURE_COLS))

    # ── Step 1: Scale ─────────────────────────────────────────────────────────
    logger.info("Step 1/4: StandardScaler ...")
    X_scaled, scaler = scale_features(features)

    # ── Step 2: PCA ───────────────────────────────────────────────────────────
    logger.info("Step 2/4: PCA (%d components) ...", n_pca_components)
    X_pca, pca_model = run_pca(X_scaled, n_components=n_pca_components)

    # ── Step 3: K-Means on PCA space ─────────────────────────────────────────
    logger.info("Step 3/4: K-Means on PCA space ...")
    silhouette_scores = find_best_k(X_pca, k_range)
    best_k = k if k is not None else max(silhouette_scores, key=silhouette_scores.__getitem__)

    km = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=20)
    kmeans_labels = km.fit_predict(X_pca)
    kmeans_sil = silhouette_score(X_pca, kmeans_labels)
    logger.info("K-Means (k=%d, PCA space): silhouette=%.4f", best_k, kmeans_sil)

    # ── Step 4: HDBSCAN validation ────────────────────────────────────────────
    hdbscan_n = None
    try:
        from sklearn.cluster import HDBSCAN
        hdb = HDBSCAN(min_cluster_size=max(5, len(features) // 20))
        hdbscan_labels = hdb.fit_predict(X_pca)
        hdbscan_n = len(set(hdbscan_labels) - {-1})
        logger.info(
            "HDBSCAN (PCA space): %d clusters, %d noise points",
            hdbscan_n, (hdbscan_labels == -1).sum(),
        )
    except ImportError:
        logger.warning("HDBSCAN not available (needs scikit-learn >= 1.3). Skipping.")
    except Exception as exc:
        logger.warning("HDBSCAN failed: %s. Skipping.", exc)

    # ── Step 5: t-SNE (visualization only) ───────────────────────────────────
    logger.info("Step 4/4: t-SNE (visualization) ...")
    X_tsne = run_tsne(X_scaled)   # run on scaled (not PCA) — standard practice

    # ── Step 6: Attach all coordinates to result DataFrame ───────────────────
    result = features.copy()
    result["cluster"] = kmeans_labels
    for i in range(n_pca_components):
        result[f"pc{i + 1}"] = X_pca[:, i].round(4)
    result["tsne_x"] = X_tsne[:, 0].round(4)
    result["tsne_y"] = X_tsne[:, 1].round(4)

    # Cluster summary log
    summary = result.groupby("cluster")[FEATURE_COLS].mean().round(4)
    logger.info("Cluster centroids:\n%s", summary.to_string())

    results_dict = {
        "best_k": best_k,
        "silhouette_scores": silhouette_scores,
        "kmeans_silhouette": round(kmeans_sil, 4),
        "hdbscan_n_clusters": hdbscan_n,
        "pca_explained_variance": list(np.round(pca_model.explained_variance_ratio_, 4)),
        "pca_cumulative_variance": float(np.round(
            np.sum(pca_model.explained_variance_ratio_), 4
        )),
        "cluster_sizes": result["cluster"].value_counts().to_dict(),
    }

    return result, results_dict


# ── Excel export ───────────────────────────────────────────────────────────────

def export_preprocessing(
    features: pd.DataFrame,
    path: str | Path = "results/preprocessing.xlsx",
    n_pca_components: int = N_PCA_COMPONENTS,
) -> Path:
    """Export normalized data, PCA components, loadings, and t-SNE to Excel.

    Creates an Excel file with 4 sheets so you can inspect the preprocessing
    visually (e.g., in Excel or Google Sheets) without writing any code.

    Sheets:
        1. scaled_features  — values after StandardScaler (mean=0, std=1)
        2. pca_components   — PC1/PC2/PC3 coordinates per player
        3. pca_loadings     — how much each original feature contributes to each PC
        4. tsne_coords      — 2D t-SNE layout for visualization

    Args:
        features: DataFrame from compute_player_features().
        path:     Output path for the .xlsx file.
        n_pca_components: Number of PCA components.

    Returns:
        Path to the saved file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Building preprocessing export for %d players ...", len(features))

    # ── Scale ─────────────────────────────────────────────────────────────────
    X_scaled, scaler = scale_features(features)
    X_pca, pca_model = run_pca(X_scaled, n_components=n_pca_components)
    X_tsne = run_tsne(X_scaled)

    player_names = features["player_name"].values

    # ── Sheet 1: Scaled features ──────────────────────────────────────────────
    scaled_df = pd.DataFrame(X_scaled, columns=FEATURE_COLS)
    scaled_df.insert(0, "player_name", player_names)
    # Add original profile assignment if present
    if "profile" in features.columns:
        scaled_df.insert(1, "profile", features["profile"].values)

    # ── Sheet 2: PCA components ───────────────────────────────────────────────
    pc_cols = [f"PC{i+1}" for i in range(n_pca_components)]
    pca_df = pd.DataFrame(X_pca, columns=pc_cols)
    pca_df.insert(0, "player_name", player_names)
    if "profile" in features.columns:
        pca_df.insert(1, "profile", features["profile"].values)

    # Append explained variance summary rows
    variance_rows = pd.DataFrame(
        [[f"% variance explained (PC{i+1})", "", round(v * 100, 2)] + [""] * (n_pca_components - 1)
         for i, v in enumerate(pca_model.explained_variance_ratio_)],
        columns=["player_name"] + (["profile"] if "profile" in features.columns else []) + pc_cols[:n_pca_components]
    )

    # ── Sheet 3: PCA loadings ─────────────────────────────────────────────────
    # Loadings = how much each original feature contributes to each PC
    # High absolute value = that feature is important for this component
    loadings_df = pd.DataFrame(
        pca_model.components_.T,
        index=FEATURE_COLS,
        columns=pc_cols,
    ).round(4)
    loadings_df.index.name = "original_feature"
    loadings_df.reset_index(inplace=True)

    # Add explained variance row at bottom
    ev_row = pd.DataFrame(
        [["explained_variance_%"] + [round(v * 100, 2) for v in pca_model.explained_variance_ratio_]],
        columns=["original_feature"] + pc_cols,
    )
    loadings_df = pd.concat([loadings_df, ev_row], ignore_index=True)

    # ── Sheet 4: t-SNE coordinates ────────────────────────────────────────────
    tsne_df = pd.DataFrame({"tsne_x": X_tsne[:, 0].round(4), "tsne_y": X_tsne[:, 1].round(4)})
    tsne_df.insert(0, "player_name", player_names)
    if "profile" in features.columns:
        tsne_df.insert(1, "profile", features["profile"].values)

    # ── Write to Excel ────────────────────────────────────────────────────────
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        scaled_df.to_excel(writer, sheet_name="scaled_features", index=False)
        pca_df.to_excel(writer, sheet_name="pca_components", index=False)
        loadings_df.to_excel(writer, sheet_name="pca_loadings", index=False)
        tsne_df.to_excel(writer, sheet_name="tsne_coords", index=False)

    logger.info("Preprocessing export saved → %s", path)
    print(f"✅  Saved: {path}")
    print(f"    Sheet 1 — scaled_features : {len(scaled_df)} players × {len(FEATURE_COLS)} features")
    print(f"    Sheet 2 — pca_components  : {len(pca_df)} players × {n_pca_components} PCs")
    print(f"    Sheet 3 — pca_loadings    : which features drive each PC")
    print(f"    Sheet 4 — tsne_coords     : 2D layout for visualization")
    print(f"    PCA variance explained    : {[round(v*100,1) for v in pca_model.explained_variance_ratio_]}%")

    return path
