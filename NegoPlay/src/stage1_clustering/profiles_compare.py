"""
src/stage1_clustering/profiles_compare.py
==========================================
Compare three approaches to building player profiles:

    1. PCA visualization  — see structure in 2D
    2. GMM + BIC          — let the data choose k
    3. Extreme Profiles   — top 10% on each axis = sharp profile

Run as:  python -m src.stage1_clustering.profiles_compare
"""

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.shared.data_loader import load_matches
from src.stage1_clustering.clustering import FEATURE_COLS
from src.stage1_clustering.features import compute_player_features

logger = logging.getLogger(__name__)

# Where to write outputs
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# Features that define the 4 "extreme corners" of style space
EXTREME_AXES: dict[str, str] = {
    "Slam Hunter":       "slam_rate",
    "Insurance Player":  "partscore_rate",
    "Fighter":           "penalty_double_rate",
    "NT Specialist":     "nt_rate",
}
EXTREME_PCT: float = 0.10   # top 10% on each axis


# ── Approach 1: PCA visualization ───────────────────────────────────────────

def run_pca(features: pd.DataFrame) -> dict:
    """Project features to 2D via PCA. Print explained variance + extremes."""
    X = StandardScaler().fit_transform(features[FEATURE_COLS].fillna(0))
    pca = PCA(n_components=3, random_state=42)
    coords = pca.fit_transform(X)

    print("\n── PCA ──────────────────────────────────────────────────────────")
    print(f"PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}% of variance")
    print(f"PC2 explains {pca.explained_variance_ratio_[1]*100:.1f}% of variance")
    print(f"PC3 explains {pca.explained_variance_ratio_[2]*100:.1f}% of variance")
    print(f"Total (3 PCs): {pca.explained_variance_ratio_.sum()*100:.1f}%")

    # Top features driving each component
    for i in range(2):
        loadings = pd.Series(pca.components_[i], index=FEATURE_COLS)
        top = loadings.abs().sort_values(ascending=False).head(3)
        print(f"\nPC{i+1} top drivers:")
        for feat in top.index:
            print(f"  {feat:25s}  loading={loadings[feat]:+.3f}")

    # Density check: are points spread or clumped?
    pc1_std = coords[:, 0].std()
    pc1_range = coords[:, 0].max() - coords[:, 0].min()
    print(f"\nPC1 std = {pc1_std:.3f}, range = {pc1_range:.3f}")
    print(f"PC1 range / std = {pc1_range/pc1_std:.2f}  "
          f"({'blob-like' if pc1_range/pc1_std < 5 else 'has spread'})")

    out = features.copy()
    out["pc1"] = coords[:, 0]
    out["pc2"] = coords[:, 1]
    return {"coords": coords, "explained": pca.explained_variance_ratio_,
            "features": out}


# ── Approach 2: GMM + BIC ───────────────────────────────────────────────────

def run_gmm(features: pd.DataFrame) -> dict:
    """Fit GMM for k=1..8, pick best by BIC. Soft assignment."""
    X = StandardScaler().fit_transform(features[FEATURE_COLS].fillna(0))

    print("\n── GMM + BIC ────────────────────────────────────────────────────")
    bics: dict[int, float] = {}
    for k in range(1, 9):
        gmm = GaussianMixture(n_components=k, covariance_type="full",
                              random_state=42, n_init=5)
        gmm.fit(X)
        bic = gmm.bic(X)
        bics[k] = bic
        print(f"  k={k}  BIC={bic:.1f}")

    best_k = min(bics, key=bics.get)
    print(f"\nBIC selects k = {best_k}")

    # Fit final GMM
    gmm = GaussianMixture(n_components=best_k, covariance_type="full",
                          random_state=42, n_init=10)
    gmm.fit(X)
    labels = gmm.predict(X)
    probs = gmm.predict_proba(X)

    # Sharpness of assignment: how confident is GMM on average?
    max_probs = probs.max(axis=1)
    print(f"Mean assignment confidence: {max_probs.mean():.3f}")
    print(f"Players assigned with >70% confidence: "
          f"{(max_probs > 0.70).sum()} / {len(features)}")

    out = features.copy()
    out["gmm_cluster"] = labels
    for j in range(best_k):
        out[f"gmm_p{j}"] = probs[:, j]

    # Print centroids
    summary = (out.groupby("gmm_cluster")[FEATURE_COLS]
                  .mean().round(3))
    print(f"\nGMM centroids:")
    print(summary.to_string())

    return {"best_k": best_k, "bics": bics, "features": out}


# ── Approach 3: Extreme Profiles ────────────────────────────────────────────

def run_extreme_profiles(features: pd.DataFrame) -> dict:
    """Tag the top-X% on each axis as that profile. Rest = Generalist.

    Players in MULTIPLE top-tiers go to their *strongest* axis (highest z-score).
    """
    print("\n── Extreme Profiles (top 10% per axis) ──────────────────────────")
    df = features.copy()

    # Compute z-score for each profile-defining feature
    z_scores = pd.DataFrame(index=df.index)
    for profile, col in EXTREME_AXES.items():
        mean = df[col].mean()
        std = df[col].std()
        z_scores[profile] = (df[col] - mean) / std

    # Compute the cutoff z-score for "top 10%"
    cutoff_z = {}
    for profile, col in EXTREME_AXES.items():
        threshold = df[col].quantile(1 - EXTREME_PCT)
        cutoff_z[profile] = (threshold - df[col].mean()) / df[col].std()
        n_above = (df[col] >= threshold).sum()
        print(f"  {profile:18s} top 10% on {col:20s}  "
              f"threshold={threshold:.4f}  n={n_above}")

    # Assign: a player is a profile only if their z-score on that axis
    # is BOTH above the 90th percentile cutoff AND is their max z-score
    # across all 4 axes (so they belong most strongly to that profile).
    def assign(row_z: pd.Series) -> str:
        top_profile = row_z.idxmax()
        if row_z[top_profile] >= cutoff_z[top_profile]:
            return top_profile
        return "Generalist"

    df["profile"] = z_scores.apply(assign, axis=1)
    df["z_max"]   = z_scores.max(axis=1)
    df["z_top_profile"] = z_scores.idxmax(axis=1)

    counts = df["profile"].value_counts()
    print(f"\nProfile sizes:")
    for p, n in counts.items():
        print(f"  {p:18s}  n={n:4d}  ({100*n/len(df):.1f}%)")

    # Centroids per profile
    summary_cols = ["slam_rate", "partscore_rate", "nt_rate",
                    "penalty_double_rate", "opening_rate", "preempt_rate",
                    "intervention_rate", "avg_level"]
    summary = (df.groupby("profile")[summary_cols].mean().round(3))
    print(f"\nProfile centroids:")
    print(summary.to_string())

    return {"features": df, "summary": summary, "counts": counts.to_dict()}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    data_path = os.getenv("DATA_PATH") or (
        r"C:\Users\annaf\OneDrive\Desktop\דוקטורט\דוקטורט אנה"
        r"\דוקטורט - שילוב בינה מלאכותית\collectBridgeData"
        r"\data\processed\all_matches_full.csv"
    )
    print(f"Loading: {data_path}")
    df = load_matches(data_path)
    features = compute_player_features(df, min_bidding_boards=20)
    print(f"\nPlayers for analysis: {len(features)}  "
          f"({len(FEATURE_COLS)} features)")

    pca_out = run_pca(features)
    gmm_out = run_gmm(features)
    extreme_out = run_extreme_profiles(features)

    # Save outputs
    pca_out["features"].to_csv(RESULTS_DIR / "players_pca.csv", index=False)
    gmm_out["features"].to_csv(RESULTS_DIR / "players_gmm.csv", index=False)
    extreme_out["features"].to_csv(RESULTS_DIR / "players_extreme_profiles.csv",
                                   index=False)

    print(f"\n✅ All 3 approaches saved to {RESULTS_DIR}")
    print(f"   - players_pca.csv  (PC1, PC2 coords)")
    print(f"   - players_gmm.csv  (GMM cluster + probabilities)")
    print(f"   - players_extreme_profiles.csv  (4 profiles + Generalist)")


if __name__ == "__main__":
    main()
