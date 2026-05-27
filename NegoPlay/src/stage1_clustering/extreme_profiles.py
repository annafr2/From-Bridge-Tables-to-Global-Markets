"""
src/stage1_clustering/extreme_profiles.py
==========================================
Assign each player to ONE of 5 behavioural profiles.

Rationale
---------
K-Means and GMM both struggle on this dataset because elite European
tournament players form a tight statistical continuum (silhouette ≤ 0.15,
GMM clusters with near-identical centroids). The data is not blob-like —
PCA shows 56% variance in 3 components — but the variation is smooth,
not discrete.

The solution: instead of cutting the continuum into fuzzy slices, we
identify the **extreme corners** of style space. A player belongs to
profile X if they are in the top P% on the axis that defines X *and*
that axis is their strongest deviation from the mean.

Profiles
--------
    Slam Hunter       — top P% on slam_rate
    Insurance Player  — top P% on partscore_rate
    Fighter           — top P% on penalty_double_rate
    NT Specialist     — top P% on nt_rate
    Generalist        — everyone else (the "average elite player")

This gives ~7-8% of the population to each extreme profile and ~70% to
Generalist — which matches the empirical reality that most elite players
are well-rounded.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Profile name → feature column that defines it
PROFILE_AXES: dict[str, str] = {
    "Slam Hunter":      "slam_rate",
    "Insurance Player": "partscore_rate",
    "Fighter":          "penalty_double_rate",
    "NT Specialist":    "nt_rate",
}

# Default percentile threshold for "extreme" — top 10%
DEFAULT_EXTREME_PCT: float = 0.10

# Profile labels in canonical order (used for prompts/agents)
PROFILE_NAMES: list[str] = list(PROFILE_AXES.keys()) + ["Generalist"]


def assign_extreme_profiles(
    features: pd.DataFrame,
    extreme_pct: float = DEFAULT_EXTREME_PCT,
) -> pd.DataFrame:
    """Assign each player to one of 5 profiles based on extreme behaviour.

    Args:
        features: DataFrame from compute_player_features() — must contain
            the feature columns referenced in PROFILE_AXES.
        extreme_pct: Fraction defining "extreme" (default 0.10 = top 10%).

    Returns:
        Copy of `features` with three new columns:
            profile         — assigned profile name (one of PROFILE_NAMES)
            profile_axis    — the feature column that drove the assignment
            profile_z       — z-score on that axis (strength of profile)
    """
    missing = [c for c in PROFILE_AXES.values() if c not in features.columns]
    if missing:
        raise ValueError(
            f"Features DataFrame is missing required columns: {missing}"
        )

    df = features.copy()

    # Compute z-scores on each profile-defining axis
    z = pd.DataFrame(index=df.index)
    cutoff_z: dict[str, float] = {}
    for profile, col in PROFILE_AXES.items():
        mean = df[col].mean()
        std = df[col].std()
        if std == 0:
            z[profile] = 0.0
            cutoff_z[profile] = float("inf")
            continue
        z[profile] = (df[col] - mean) / std
        # Cutoff: top X% of THIS axis
        threshold = df[col].quantile(1 - extreme_pct)
        cutoff_z[profile] = (threshold - mean) / std

    # Assign: a player is profile P only if (a) their strongest axis is P
    # AND (b) their z-score on P passes the top-X% cutoff for P.
    def _assign(row_z: pd.Series) -> tuple[str, str, float]:
        top_profile = row_z.idxmax()
        top_z = float(row_z[top_profile])
        if top_z >= cutoff_z[top_profile]:
            return top_profile, PROFILE_AXES[top_profile], top_z
        return "Generalist", "", top_z

    assignments = z.apply(_assign, axis=1, result_type="expand")
    assignments.columns = ["profile", "profile_axis", "profile_z"]
    df = pd.concat([df, assignments], axis=1)

    # Log summary
    counts = df["profile"].value_counts()
    logger.info("Assigned %d players to %d profiles:", len(df), len(counts))
    for profile in PROFILE_NAMES:
        n = int(counts.get(profile, 0))
        pct = 100.0 * n / max(len(df), 1)
        logger.info("  %-18s n=%4d  (%.1f%%)", profile, n, pct)

    return df


def profile_summary(features_with_profile: pd.DataFrame) -> pd.DataFrame:
    """Return mean of every numeric feature, grouped by profile."""
    numeric_cols = features_with_profile.select_dtypes(include="number").columns
    # Drop noisy/uninteresting columns
    drop_cols = {"profile_z", "n_bidding_boards", "n_declared"}
    keep = [c for c in numeric_cols if c not in drop_cols]
    return (
        features_with_profile.groupby("profile")[keep]
        .mean()
        .round(4)
        .reindex(PROFILE_NAMES)
    )
