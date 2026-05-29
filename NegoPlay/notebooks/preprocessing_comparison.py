"""
notebooks/preprocessing_comparison.py
======================================
Full preprocessing + clustering comparison after Dr. Rami's review.

Runs 4 clustering algorithms (K-Means, GMM, HDBSCAN, plus k-range scan)
on the cleaned data and compares against the baseline.
"""

from __future__ import annotations

import sys
import os
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

from src.shared.data_loader import load_matches
from src.stage1_clustering.features import compute_player_features
from src.stage1_clustering.preprocessing import preprocess_full

# ── Original 10 features (before any filtering) ───────────────────────────────

ALL_FEATURES = [
    "slam_rate", "double_rate", "avg_level", "nt_rate", "partscore_rate",
    "opening_rate", "preempt_rate", "intervention_rate",
    "penalty_double_rate", "avg_bids_per_board",
]

DATA = (
    r"C:\Users\annaf\OneDrive\Desktop\דוקטורט\דוקטורט אנה"
    r"\דוקטורט - שילוב בינה מלאכותית\collectBridgeData"
    r"\data\processed\all_matches_full.csv"
)

# ── Run pipeline ──────────────────────────────────────────────────────────────

print("Loading data...")
df = load_matches(DATA)
features = compute_player_features(df, min_bidding_boards=20)
print(f"Players loaded: {len(features)}")
print()

X_scaled, X_pca, clean_df, report = preprocess_full(
    features=features,
    feature_cols=ALL_FEATURES,
    min_cv=0.10,
    max_corr=0.70,
    target_variance=0.80,
    outlier_alpha=0.01,
    scaler_type="robust",
    remove_outliers=True,
)

print(report.summary())
print()

# ── Run 3 clustering algorithms on PCA-reduced data ───────────────────────────

print("=" * 60)
print("CLUSTERING RESULTS ON CLEAN PCA SPACE")
print("=" * 60)

# K-Means: scan k=2..8
print()
print("K-Means (PCA space):")
kmeans_scores: dict[int, float] = {}
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=30)
    labels = km.fit_predict(X_pca)
    score = silhouette_score(X_pca, labels)
    kmeans_scores[k] = score
    flag = ""
    print(f"  k={k}  silhouette={score:.4f}{flag}")

best_k = max(kmeans_scores, key=kmeans_scores.__getitem__)
print(f"  -> Best k = {best_k}  (silhouette = {kmeans_scores[best_k]:.4f})")

# GMM: scan k=2..8 with BIC
print()
print("GMM (Gaussian Mixture Model, PCA space):")
gmm_bic: dict[int, float] = {}
gmm_sil: dict[int, float] = {}
for k in range(2, 9):
    gmm = GaussianMixture(n_components=k, random_state=42, n_init=10)
    gmm.fit(X_pca)
    labels = gmm.predict(X_pca)
    sil = silhouette_score(X_pca, labels) if len(set(labels)) > 1 else -1
    bic = gmm.bic(X_pca)
    gmm_bic[k] = bic
    gmm_sil[k] = sil
    print(f"  k={k}  silhouette={sil:.4f}  BIC={bic:.0f}")

best_gmm = min(gmm_bic, key=gmm_bic.__getitem__)
print(f"  -> Best k by BIC = {best_gmm}  (silhouette = {gmm_sil[best_gmm]:.4f})")

# HDBSCAN
print()
print("HDBSCAN (PCA space):")
hdb = HDBSCAN(min_cluster_size=max(10, len(clean_df) // 25))
hdb_labels = hdb.fit_predict(X_pca)
hdb_n = len(set(hdb_labels) - {-1})
hdb_noise = (hdb_labels == -1).sum()
print(f"  clusters found: {hdb_n}")
print(f"  noise points:   {hdb_noise} / {len(clean_df)}")
if hdb_n >= 2:
    mask = hdb_labels != -1
    if mask.sum() > 1:
        hdb_sil = silhouette_score(X_pca[mask], hdb_labels[mask])
        print(f"  silhouette (non-noise): {hdb_sil:.4f}")

# ── Comparison table ──────────────────────────────────────────────────────────

print()
print("=" * 60)
print("COMPARISON TABLE")
print("=" * 60)
print()
print(f"  {'Pipeline':<45} {'Best k':>7} {'Silhouette':>12}")
print(f"  {'-' * 45} {'-' * 7} {'-' * 12}")
print(f"  {'V1: 10 feat, no PCA, StandardScaler':<45} "
      f"{'4':>7} {'~0.15':>12}")
print(f"  {'V2: 8 feat + PCA(3), StandardScaler':<45} "
      f"{'4':>7} {'0.24':>12}")
print(f"  {'V3: full preprocessing + K-Means':<45} "
      f"{best_k:>7} {kmeans_scores[best_k]:>12.4f}")
print(f"  {'V3: full preprocessing + GMM':<45} "
      f"{best_gmm:>7} {gmm_sil[best_gmm]:>12.4f}")
print(f"  {'V3: full preprocessing + HDBSCAN':<45} "
      f"{str(hdb_n):>7} "
      f"{'n/a' if hdb_n < 2 else f'{hdb_sil:.4f}':>12}")
print()

# ── Verdict ──────────────────────────────────────────────────────────────────

print("=" * 60)
print("VERDICT")
print("=" * 60)
best_score = max(kmeans_scores[best_k], gmm_sil[best_gmm])
print()
if best_score >= 0.50:
    print(f"  STRONG CLUSTERS  (silhouette = {best_score:.3f} >= 0.50)")
    print(f"  Dr. Rami was right — clusters exist but were hidden by noise.")
elif best_score >= 0.35:
    print(f"  MODERATE STRUCTURE  (silhouette = {best_score:.3f} >= 0.35)")
    print(f"  Some grouping exists — neither pure continuum nor strong clusters.")
    print(f"  Preprocessing helped substantially; consider profile blending.")
elif best_score >= 0.25:
    print(f"  WEAK STRUCTURE  (silhouette = {best_score:.3f} >= 0.25)")
    print(f"  Mostly continuum, with subtle structure that K-Means can find.")
    print(f"  Extreme-percentile method (current approach) still appropriate.")
else:
    print(f"  CONTINUUM CONFIRMED  (silhouette = {best_score:.3f} < 0.25)")
    print(f"  Even with full preprocessing, no clusters emerge.")
    print(f"  This is a robust empirical finding — defensible for the paper.")

print()
print("Done.")
