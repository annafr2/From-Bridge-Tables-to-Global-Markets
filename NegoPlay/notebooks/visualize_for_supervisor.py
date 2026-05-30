"""
notebooks/visualize_for_supervisor.py
=====================================
Supervisor-facing "snapshot" visualizations of the player profiles.

Produces four publication-ready figures from data/processed/player_profiles.csv
(563 players, 5 profiles). No LLM calls, no network — pure matplotlib + pandas.

Figures (saved to docs/images/):
  1. supervisor_box_defining_metrics.png
     2x2 box+strip plots. For each profile's DEFINING metric, shows the
     distribution across all 5 profiles. Visual proof that each profile spikes
     on its own axis (this is the Q7.7 validation, made visual).

  2. supervisor_heatmap_fingerprint.png
     One heatmap: 5 profiles (rows) x 8 metrics (cols). Colour = z-score of the
     profile mean relative to the whole population. One-glance "fingerprint" of
     each behavioural type.

  3. supervisor_population_breakdown.png
     How the 563 players split into profiles (counts) + the mean defining-metric
     value per profile against the population baseline.

  4. supervisor_scatter_risk_axis.png
     Scatter of slam_rate (aggression) vs partscore_rate (caution), one point
     per player, coloured by profile. Shows the continuum and where the extreme
     profiles sit on it.

Run:
    python notebooks/visualize_for_supervisor.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

PROFILES_CSV = Path("data/processed/player_profiles.csv")
OUT_DIR = Path("docs/images")

# Display order: baseline first, then the four extreme profiles
PROFILE_ORDER: list[str] = [
    "Generalist",
    "Slam Hunter",
    "Insurance Player",
    "Fighter",
    "NT Specialist",
]

# Consistent colour per profile across all figures
PROFILE_COLORS: dict[str, str] = {
    "Generalist":       "#9aa0a6",  # grey  = baseline
    "Slam Hunter":      "#d62728",  # red   = aggression
    "Insurance Player": "#1f77b4",  # blue  = caution
    "Fighter":          "#ff7f0e",  # orange= combative
    "NT Specialist":    "#2ca02c",  # green = NT
}

# Each profile is DEFINED by being extreme on this metric (extreme_profiles.py)
DEFINING_METRIC: dict[str, str] = {
    "Slam Hunter":      "slam_rate",
    "Insurance Player": "partscore_rate",
    "Fighter":          "penalty_double_rate",
    "NT Specialist":    "nt_rate",
}

# Human-readable metric labels
METRIC_LABELS: dict[str, str] = {
    "slam_rate":           "Slam rate",
    "partscore_rate":      "Partscore rate",
    "penalty_double_rate": "Penalty-double rate",
    "nt_rate":             "NT rate",
    "double_rate":         "Doubled-contract rate",
    "success_rate":        "Success rate",
    "preempt_rate":        "Preempt rate",
    "opening_rate":        "Opening rate",
}

# Metrics shown in the fingerprint heatmap (behaviourally meaningful, comparable)
HEATMAP_METRICS: list[str] = [
    "slam_rate",
    "partscore_rate",
    "penalty_double_rate",
    "nt_rate",
    "success_rate",
    "opening_rate",
    "preempt_rate",
    "double_rate",
]

DPI = 150


# ── Data loading ──────────────────────────────────────────────────────────────

def load_profiles() -> pd.DataFrame:
    """Load the 563-player profile table and order the profile column."""
    df = pd.read_csv(PROFILES_CSV, encoding="utf-8-sig")
    present = [p for p in PROFILE_ORDER if p in df["profile"].unique()]
    df["profile"] = pd.Categorical(df["profile"], categories=present, ordered=True)
    logger.info("Loaded %d players across %d profiles", len(df), len(present))
    return df


# ── Figure 1: box + strip plots of defining metrics ──────────────────────────

def fig_box_defining_metrics(df: pd.DataFrame) -> Path:
    """2x2 grid: per defining metric, box plots across all profiles."""
    present = list(df["profile"].cat.categories)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(
        "Each profile spikes on its own defining metric\n"
        f"European Championship players (n = {len(df)})",
        fontsize=15, fontweight="bold",
    )

    for ax, (profile, metric) in zip(axes.ravel(), DEFINING_METRIC.items()):
        groups = [df.loc[df["profile"] == p, metric].dropna().values for p in present]

        bp = ax.boxplot(
            groups, patch_artist=True, widths=0.6,
            medianprops=dict(color="black", linewidth=1.5),
            showfliers=False,
        )
        for patch, p in zip(bp["boxes"], present):
            # Fade every profile except the one this panel is about
            patch.set_facecolor(PROFILE_COLORS[p])
            patch.set_alpha(1.0 if p == profile else 0.35)

        # Jittered points on top
        rng = np.random.default_rng(42)
        for i, (p, vals) in enumerate(zip(present, groups), start=1):
            x = rng.normal(i, 0.06, size=len(vals))
            ax.scatter(x, vals, s=10, color=PROFILE_COLORS[p],
                       alpha=0.5, edgecolors="none", zorder=3)

        ax.set_title(f"{profile}  →  {METRIC_LABELS[metric]}", fontweight="bold")
        ax.set_xticks(range(1, len(present) + 1))
        ax.set_xticklabels(present, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = OUT_DIR / "supervisor_box_defining_metrics.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out)
    return out


# ── Figure 2: fingerprint heatmap ────────────────────────────────────────────

def fig_heatmap_fingerprint(df: pd.DataFrame) -> Path:
    """Heatmap of per-profile mean metrics, z-scored against the population."""
    present = list(df["profile"].cat.categories)
    metrics = [m for m in HEATMAP_METRICS if m in df.columns]

    # z-score each metric across the whole population, then average per profile
    z = df[metrics].copy()
    z = (z - z.mean()) / z.std(ddof=0)
    z["profile"] = df["profile"]
    mat = z.groupby("profile", observed=True)[metrics].mean().reindex(present)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    vmax = float(np.nanmax(np.abs(mat.values)))
    im = ax.imshow(mat.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([METRIC_LABELS.get(m, m) for m in metrics],
                       rotation=30, ha="right")
    ax.set_yticks(range(len(present)))
    ax.set_yticklabels(present)

    # Annotate each cell with the z value
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.values[i, j]
            ax.text(j, i, f"{val:+.1f}", ha="center", va="center",
                    fontsize=8, color="black" if abs(val) < vmax * 0.6 else "white")

    ax.set_title(
        "Behavioural fingerprint per profile\n"
        "(z-score of profile mean vs. whole population; red = above average)",
        fontweight="bold",
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Std. deviations from population mean")

    fig.tight_layout()
    out = OUT_DIR / "supervisor_heatmap_fingerprint.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out)
    return out


# ── Figure 3: population breakdown ───────────────────────────────────────────

def fig_population_breakdown(df: pd.DataFrame) -> Path:
    """Left: player count per profile. Right: mean defining metric vs baseline."""
    present = list(df["profile"].cat.categories)
    counts = df["profile"].value_counts().reindex(present)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"How {len(df)} elite players split into profiles",
        fontsize=15, fontweight="bold",
    )

    # ── Left: counts ──
    bars = ax1.bar(present, counts.values,
                   color=[PROFILE_COLORS[p] for p in present], edgecolor="black")
    for bar, c in zip(bars, counts.values):
        pct = 100 * c / len(df)
        ax1.text(bar.get_x() + bar.get_width() / 2, c + 4,
                 f"{c}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)
    ax1.set_ylabel("Number of players")
    ax1.set_title("Population breakdown (continuum → small extreme tails)")
    ax1.set_xticks(range(len(present)))
    ax1.set_xticklabels(present, rotation=20, ha="right")
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_ylim(0, counts.max() * 1.18)

    # ── Right: defining metric mean per extreme profile vs Generalist ──
    extreme = [p for p in present if p != "Generalist"]
    metric_for = DEFINING_METRIC
    prof_means, base_means, labels = [], [], []
    for p in extreme:
        m = metric_for[p]
        prof_means.append(df.loc[df["profile"] == p, m].mean())
        base_means.append(df.loc[df["profile"] == "Generalist", m].mean())
        labels.append(f"{p}\n({METRIC_LABELS[m]})")

    x = np.arange(len(extreme))
    w = 0.38
    ax2.bar(x - w / 2, base_means, w, label="Generalist (baseline)",
            color="#9aa0a6", edgecolor="black")
    ax2.bar(x + w / 2, prof_means, w, label="Profile",
            color=[PROFILE_COLORS[p] for p in extreme], edgecolor="black")
    for i, (b, pm) in enumerate(zip(base_means, prof_means)):
        ratio = pm / b if b else float("nan")
        ax2.text(i + w / 2, pm, f"×{ratio:.2f}", ha="center", va="bottom",
                 fontsize=9, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("Mean rate")
    ax2.set_title("Defining metric: profile vs. baseline")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = OUT_DIR / "supervisor_population_breakdown.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out)
    return out


# ── Figure 4: risk-axis scatter ──────────────────────────────────────────────

def fig_scatter_risk_axis(df: pd.DataFrame) -> Path:
    """Scatter slam_rate (aggression) vs partscore_rate (caution) by profile."""
    present = list(df["profile"].cat.categories)
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot Generalist first (background), extremes on top
    for p in present:
        sub = df[df["profile"] == p]
        ax.scatter(
            sub["slam_rate"], sub["partscore_rate"],
            s=70 if p != "Generalist" else 28,
            color=PROFILE_COLORS[p],
            alpha=0.85 if p != "Generalist" else 0.35,
            edgecolors="black" if p != "Generalist" else "none",
            linewidths=0.5,
            label=f"{p} (n={len(sub)})",
            zorder=3 if p != "Generalist" else 1,
        )

    ax.set_xlabel("Slam rate  →  more aggressive", fontsize=12)
    ax.set_ylabel("Partscore rate  →  more conservative", fontsize=12)
    ax.set_title(
        "The risk continuum: aggression vs. caution\n"
        f"one point = one player (n = {len(df)})",
        fontsize=14, fontweight="bold",
    )
    ax.legend(title="Profile", loc="upper right", framealpha=0.9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = OUT_DIR / "supervisor_scatter_risk_axis.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out)
    return out


# ── Figure 5: combative-vs-NT scatter (separates Fighter & NT Specialist) ─────

def fig_scatter_fighter_nt(df: pd.DataFrame) -> Path:
    """Scatter penalty_double_rate vs nt_rate by profile.

    Companion to Figure 4. The risk-axis scatter (slam vs partscore) only
    separates Slam Hunter and Insurance Player cleanly; Fighter and NT
    Specialist sit central there because their defining axes are not shown.
    This figure puts those two axes on the plot so all four extreme profiles
    become visually distinct (addresses bridge-expert caveat #3).
    """
    present = list(df["profile"].cat.categories)
    fig, ax = plt.subplots(figsize=(10, 8))

    for p in present:
        sub = df[df["profile"] == p]
        ax.scatter(
            sub["penalty_double_rate"], sub["nt_rate"],
            s=70 if p != "Generalist" else 28,
            color=PROFILE_COLORS[p],
            alpha=0.85 if p != "Generalist" else 0.35,
            edgecolors="black" if p != "Generalist" else "none",
            linewidths=0.5,
            label=f"{p} (n={len(sub)})",
            zorder=3 if p != "Generalist" else 1,
        )

    ax.set_xlabel("Penalty-double rate  →  more combative", fontsize=12)
    ax.set_ylabel("NT rate  →  more NT-oriented", fontsize=12)
    ax.set_title(
        "Combative vs. NT style: separating Fighter & NT Specialist\n"
        f"one point = one player (n = {len(df)})",
        fontsize=14, fontweight="bold",
    )
    ax.legend(title="Profile", loc="upper right", framealpha=0.9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = OUT_DIR / "supervisor_scatter_fighter_nt.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out)
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_profiles()

    outputs = [
        fig_box_defining_metrics(df),
        fig_heatmap_fingerprint(df),
        fig_population_breakdown(df),
        fig_scatter_risk_axis(df),
        fig_scatter_fighter_nt(df),
    ]

    print("\nGenerated figures:")
    for o in outputs:
        print(f"  - {o}")
    print(f"\nAll saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
