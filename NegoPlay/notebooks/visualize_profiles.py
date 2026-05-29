"""
notebooks/visualize_profiles.py
================================
Generate visualizations showing how players cluster into profiles.

Produces 5 images saved to docs/images/:
  1. pca_scatter.png    — PCA 2D scatter, coloured by profile
  2. radar_profiles.png — Radar chart comparing the 5 profiles
  3. feature_bars.png   — Side-by-side bar chart of key features
  4. tsne_scatter.png   — t-SNE 2D scatter (visualization only — NOT evidence of clusters)
  5. pca_variance.png   — PCA scree plot: how much variance each component explains
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")   # non-interactive backend (saves to file)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from src.shared.data_loader import load_matches
from src.stage1_clustering.features import compute_player_features
from src.stage1_clustering.extreme_profiles import (
    assign_extreme_profiles,
    PROFILE_NAMES,
)
from src.stage1_clustering.clustering import FEATURE_COLS

# ── Config ───────────────────────────────────────────────────────────────────

DATA_PATH = (
    r"C:\Users\annaf\OneDrive\Desktop\דוקטורט\דוקטורט אנה"
    r"\דוקטורט - שילוב בינה מלאכותית\collectBridgeData"
    r"\data\processed\all_matches_full.csv"
)
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "images")
os.makedirs(OUT_DIR, exist_ok=True)

PROFILE_COLORS = {
    "Slam Hunter":      "#E74C3C",   # red
    "Insurance Player": "#27AE60",   # green
    "Fighter":          "#E67E22",   # orange
    "NT Specialist":    "#2980B9",   # blue
    "Generalist":       "#BDC3C7",   # light grey
}

PROFILE_MARKERS = {
    "Slam Hunter":      "^",
    "Insurance Player": "s",
    "Fighter":          "D",
    "NT Specialist":    "P",
    "Generalist":       "o",
}

# ── Load data ─────────────────────────────────────────────────────────────────

print("Loading data...")
df = load_matches(DATA_PATH)
features = compute_player_features(df, min_bidding_boards=20)
profiles = assign_extreme_profiles(features)
print(f"Players: {len(profiles)}")


# ── 1. PCA scatter ────────────────────────────────────────────────────────────

print("Creating PCA scatter...")
X = StandardScaler().fit_transform(profiles[FEATURE_COLS].fillna(0))
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X)

fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor("#FAFAFA")
ax.set_facecolor("#FAFAFA")

# Generalist first (behind), then extremes on top
order = ["Generalist", "Insurance Player", "NT Specialist", "Fighter", "Slam Hunter"]
for profile in order:
    mask = profiles["profile"] == profile
    size = 20 if profile == "Generalist" else 70
    alpha = 0.35 if profile == "Generalist" else 0.85
    ax.scatter(
        coords[mask, 0], coords[mask, 1],
        c=PROFILE_COLORS[profile],
        marker=PROFILE_MARKERS[profile],
        s=size, alpha=alpha, zorder=3 if profile != "Generalist" else 2,
        label=f"{profile} (n={mask.sum()})",
    )

ax.set_xlabel(
    f"PC1 — Bidding Height  ({pca.explained_variance_ratio_[0]*100:.0f}% variance)",
    fontsize=11,
)
ax.set_ylabel(
    f"PC2 — Bidding Activity ({pca.explained_variance_ratio_[1]*100:.0f}% variance)",
    fontsize=11,
)
ax.set_title(
    "Elite Bridge Player Profiles in Behaviour Space\n"
    "(PCA of 10 behavioural features, 807 players)",
    fontsize=13, fontweight="bold", pad=12,
)
ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
ax.grid(alpha=0.3, linestyle="--")
plt.tight_layout()
path1 = os.path.join(OUT_DIR, "pca_scatter.png")
plt.savefig(path1, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {path1}")


# ── 2. Radar chart ────────────────────────────────────────────────────────────

print("Creating radar chart...")

radar_features = [
    "slam_rate", "partscore_rate", "nt_rate",
    "penalty_double_rate", "opening_rate", "preempt_rate",
]
radar_labels = [
    "Slam Rate", "Partscore Rate", "NT Rate",
    "Penalty Double", "Opening Rate", "Preempt Rate",
]

# Normalize each feature to [0, 1] based on min/max across ALL players
norms = {}
for col in radar_features:
    vmin, vmax = profiles[col].min(), profiles[col].max()
    norms[col] = (profiles[col] - vmin) / (vmax - vmin + 1e-9)

# Compute mean normalised values per profile (exclude Generalist from extremes)
means = {}
for profile in PROFILE_NAMES:
    mask = profiles["profile"] == profile
    means[profile] = [norms[col][mask].mean() for col in radar_features]

N = len(radar_features)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]   # close the polygon

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
fig.patch.set_facecolor("#FAFAFA")
ax.set_facecolor("#FAFAFA")

for profile in PROFILE_NAMES:
    values = means[profile] + means[profile][:1]
    lw = 1.5 if profile == "Generalist" else 2.5
    ls = "--" if profile == "Generalist" else "-"
    ax.plot(angles, values, color=PROFILE_COLORS[profile],
            linewidth=lw, linestyle=ls, label=profile)
    ax.fill(angles, values, color=PROFILE_COLORS[profile],
            alpha=0.08 if profile == "Generalist" else 0.15)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_labels, fontsize=10)
ax.set_yticklabels([])
ax.set_title(
    "Behavioural Fingerprints of 5 Player Profiles\n"
    "(values normalised to [0,1])",
    fontsize=12, fontweight="bold", pad=20,
)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
plt.tight_layout()
path2 = os.path.join(OUT_DIR, "radar_profiles.png")
plt.savefig(path2, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {path2}")


# ── 3. Feature bar chart ──────────────────────────────────────────────────────

print("Creating feature bar chart...")

bar_features = [
    ("slam_rate",           "Slam Rate\n(% contracts at level 6-7)"),
    ("partscore_rate",      "Partscore Rate\n(% contracts at level 1-3)"),
    ("nt_rate",             "NT Rate\n(% contracts in NoTrump)"),
    ("penalty_double_rate", "Penalty Double Rate\n(% boards with a double)"),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.patch.set_facecolor("#FAFAFA")
fig.suptitle(
    "Key Feature Comparison Across Player Profiles",
    fontsize=14, fontweight="bold", y=1.01,
)

for ax, (col, title) in zip(axes.flatten(), bar_features):
    ax.set_facecolor("#FAFAFA")
    vals = [profiles[profiles["profile"] == p][col].mean() for p in PROFILE_NAMES]
    bars = ax.bar(
        PROFILE_NAMES, vals,
        color=[PROFILE_COLORS[p] for p in PROFILE_NAMES],
        edgecolor="white", linewidth=0.8, alpha=0.9,
    )
    # Value labels on top of bars
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{v:.3f}",
            ha="center", va="bottom", fontsize=8, fontweight="bold",
        )
    ax.set_title(title, fontsize=10, pad=6)
    ax.set_xticklabels(
        [p.replace(" ", "\n") for p in PROFILE_NAMES],
        fontsize=8,
    )
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim(0, max(vals) * 1.2)
    # Highlight the defining profile for each feature
    defining = {
        "slam_rate": "Slam Hunter",
        "partscore_rate": "Insurance Player",
        "nt_rate": "NT Specialist",
        "penalty_double_rate": "Fighter",
    }
    if col in defining:
        idx = PROFILE_NAMES.index(defining[col])
        bars[idx].set_edgecolor("#2C3E50")
        bars[idx].set_linewidth(2.5)

plt.tight_layout()
path3 = os.path.join(OUT_DIR, "feature_bars.png")
plt.savefig(path3, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {path3}")

# ── 4. t-SNE scatter ─────────────────────────────────────────────────────────

print("Creating t-SNE scatter (this takes ~30 seconds)...")

X_tsne = TSNE(
    n_components=2, perplexity=30,
    random_state=42, max_iter=1000,
).fit_transform(X)

fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor("#FAFAFA")
ax.set_facecolor("#FAFAFA")

for profile in order:
    mask = profiles["profile"] == profile
    size  = 20  if profile == "Generalist" else 70
    alpha = 0.35 if profile == "Generalist" else 0.85
    ax.scatter(
        X_tsne[mask.values, 0], X_tsne[mask.values, 1],
        c=PROFILE_COLORS[profile],
        marker=PROFILE_MARKERS[profile],
        s=size, alpha=alpha,
        zorder=3 if profile != "Generalist" else 2,
        label=f"{profile} (n={mask.sum()})",
    )

ax.set_xlabel("t-SNE dimension 1", fontsize=11)
ax.set_ylabel("t-SNE dimension 2", fontsize=11)
ax.set_title(
    "t-SNE Visualization of Player Profiles\n"
    "⚠️  For visualization only — not evidence of clusters",
    fontsize=13, fontweight="bold", pad=12,
)
ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
ax.grid(alpha=0.3, linestyle="--")

# Add disclaimer text at the bottom
fig.text(
    0.5, -0.02,
    "t-SNE always creates visual blobs regardless of whether true clusters exist. "
    "Statistical clustering (K-Means/HDBSCAN) is the valid test.",
    ha="center", fontsize=8, style="italic", color="#7F8C8D",
)

plt.tight_layout()
path4 = os.path.join(OUT_DIR, "tsne_scatter.png")
plt.savefig(path4, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {path4}")


# ── 5. PCA scree plot ─────────────────────────────────────────────────────────

print("Creating PCA scree plot...")

pca_full = PCA(n_components=len(FEATURE_COLS), random_state=42)
pca_full.fit(X)

explained = pca_full.explained_variance_ratio_ * 100
cumulative = np.cumsum(explained)
n_comp = len(explained)

fig, ax1 = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor("#FAFAFA")
ax1.set_facecolor("#FAFAFA")

bars = ax1.bar(
    range(1, n_comp + 1), explained,
    color="#2980B9", alpha=0.75, edgecolor="white", label="Individual variance %",
)
ax1.set_xlabel("Principal Component", fontsize=11)
ax1.set_ylabel("Explained Variance (%)", fontsize=11, color="#2980B9")
ax1.tick_params(axis="y", labelcolor="#2980B9")

ax2 = ax1.twinx()
ax2.plot(range(1, n_comp + 1), cumulative, "o-", color="#E74C3C",
         linewidth=2, markersize=6, label="Cumulative variance %")
ax2.set_ylabel("Cumulative Variance (%)", fontsize=11, color="#E74C3C")
ax2.tick_params(axis="y", labelcolor="#E74C3C")
ax2.axhline(y=80, color="#E74C3C", linestyle="--", alpha=0.4)
ax2.text(n_comp - 0.5, 81, "80%", color="#E74C3C", fontsize=9)
ax2.set_ylim(0, 110)

# Mark the 3 components we use
ax1.axvline(x=3.5, color="#27AE60", linestyle="--", alpha=0.7, linewidth=1.5)
ax1.text(3.6, max(explained) * 0.9, "← 3 components used",
         color="#27AE60", fontsize=9)

ax1.set_title(
    "PCA Scree Plot — How Much Variance Each Component Explains",
    fontsize=12, fontweight="bold", pad=10,
)
ax1.set_xticks(range(1, n_comp + 1))

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=9)

plt.tight_layout()
path5 = os.path.join(OUT_DIR, "pca_variance.png")
plt.savefig(path5, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {path5}")


print("\nDone! All 5 images saved to docs/images/")
print(f"  1. pca_scatter.png   — PCA coloured by profile")
print(f"  2. radar_profiles.png — behavioural fingerprints")
print(f"  3. feature_bars.png  — key feature comparison")
print(f"  4. tsne_scatter.png  — t-SNE layout (visualization only)")
print(f"  5. pca_variance.png  — scree plot: variance per component")
