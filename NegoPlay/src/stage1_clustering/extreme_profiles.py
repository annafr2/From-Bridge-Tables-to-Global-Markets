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

Sample-size robustness (added May 2026)
---------------------------------------
After review by an expert bridge advisor (Nezer), an additional binomial
significance test was added: a player is only assigned to a profile if
their rate on the defining axis is **statistically distinguishable from
the population baseline** at p < 0.05 (one-sided binomial test, "greater").

This prevents false-positive profile assignment for players whose extreme
rate is a small-sample artefact. For example, a player with 20 declared
boards and 2 slams has slam_rate=10%, but the 95% CI is [1.2%, 31.7%] —
not statistically distinguishable from the 4.1% population baseline.
Under the new rule, such a player would NOT be classified as Slam Hunter.

Profiles
--------
    Slam Hunter       — top P% on slam_rate AND binomial p < 0.05
    Insurance Player  — top P% on partscore_rate AND binomial p < 0.05
    Fighter           — top P% on penalty_double_rate AND binomial p < 0.05
    NT Specialist     — top P% on nt_rate AND binomial p < 0.05
    Generalist        — everyone else (the "average elite player")
"""

import logging

import pandas as pd
from scipy.stats import binomtest

logger = logging.getLogger(__name__)

# Profile name → feature column that defines it
PROFILE_AXES: dict[str, str] = {
    "Slam Hunter":      "slam_rate",
    "Insurance Player": "partscore_rate",
    "Fighter":          "penalty_double_rate",
    "NT Specialist":    "nt_rate",
}

# Profile name → column holding the denominator (number of trials)
# Outcome features (slam/partscore/nt) use declared boards.
# Process features (penalty_double) use bidding boards.
PROFILE_DENOMINATORS: dict[str, str] = {
    "Slam Hunter":      "n_declared",
    "Insurance Player": "n_declared",
    "Fighter":          "n_bidding_boards",
    "NT Specialist":    "n_declared",
}

# Default percentile threshold for "extreme" — top 10%
DEFAULT_EXTREME_PCT: float = 0.10

# Significance threshold for the binomial test (one-sided, "greater")
DEFAULT_ALPHA: float = 0.05

# Profile labels in canonical order (used for prompts/agents)
PROFILE_NAMES: list[str] = list(PROFILE_AXES.keys()) + ["Generalist"]


def _binomial_pvalue(successes: int, trials: int, baseline_rate: float) -> float:
    """One-sided binomial test: is observed rate > baseline?

    Returns the p-value. Lower p means stronger evidence the player's rate
    is genuinely higher than the population baseline (not a small-sample
    artefact).
    """
    if trials <= 0:
        return 1.0
    if baseline_rate <= 0 or baseline_rate >= 1:
        return 1.0
    result = binomtest(
        k=int(successes), n=int(trials), p=baseline_rate, alternative="greater"
    )
    return float(result.pvalue)


def assign_extreme_profiles(
    features: pd.DataFrame,
    extreme_pct: float = DEFAULT_EXTREME_PCT,
    alpha: float = DEFAULT_ALPHA,
    require_significance: bool = True,
) -> pd.DataFrame:
    """Assign each player to one of 5 profiles based on extreme behaviour.

    Args:
        features: DataFrame from compute_player_features() — must contain
            the feature columns referenced in PROFILE_AXES and the
            denominator columns referenced in PROFILE_DENOMINATORS.
        extreme_pct: Fraction defining "extreme" (default 0.10 = top 10%).
        alpha: Significance threshold for the binomial test (default 0.05).
        require_significance: If True (default), also require the player's
            rate to be statistically distinguishable from the baseline at
            p < alpha. Set to False to reproduce the pre-May-2026 behaviour.

    Returns:
        Copy of `features` with these new columns:
            profile          — assigned profile name (one of PROFILE_NAMES)
            profile_axis     — the feature column that drove the assignment
            profile_z        — z-score on that axis (strength of profile)
            profile_pvalue   — binomial p-value (lower = stronger evidence)
    """
    missing_axes = [c for c in PROFILE_AXES.values() if c not in features.columns]
    if missing_axes:
        raise ValueError(
            f"Features DataFrame is missing required axis columns: {missing_axes}"
        )
    if require_significance:
        missing_denoms = [
            c for c in PROFILE_DENOMINATORS.values() if c not in features.columns
        ]
        if missing_denoms:
            raise ValueError(
                "Features DataFrame is missing denominator columns required "
                f"for the binomial test: {missing_denoms}. Either provide "
                "them, or call with require_significance=False."
            )

    df = features.copy()

    # ── Step 1: Compute z-scores and cutoff thresholds per axis ─────────────
    z = pd.DataFrame(index=df.index)
    cutoff_z: dict[str, float] = {}
    baseline_rate: dict[str, float] = {}
    for profile, col in PROFILE_AXES.items():
        mean = df[col].mean()
        std = df[col].std()
        baseline_rate[profile] = float(mean)
        if std == 0:
            z[profile] = 0.0
            cutoff_z[profile] = float("inf")
            continue
        z[profile] = (df[col] - mean) / std
        threshold = df[col].quantile(1 - extreme_pct)
        cutoff_z[profile] = (threshold - mean) / std

    # ── Step 2: Assign each player ──────────────────────────────────────────
    def _assign(row: pd.Series) -> tuple[str, str, float, float]:
        # Find the player's strongest axis
        row_z = pd.Series({p: z.loc[row.name, p] for p in PROFILE_AXES})
        top_profile = row_z.idxmax()
        top_z = float(row_z[top_profile])

        # Must pass the top-X% z-score cutoff
        if top_z < cutoff_z[top_profile]:
            return "Generalist", "", top_z, 1.0

        # Compute binomial p-value for the chosen profile
        axis_col = PROFILE_AXES[top_profile]
        denom_col = PROFILE_DENOMINATORS[top_profile]
        rate = float(row[axis_col])
        trials = int(row[denom_col])
        successes = int(round(rate * trials))
        pval = _binomial_pvalue(successes, trials, baseline_rate[top_profile])

        # If significance is required, fail-over to Generalist
        if require_significance and pval >= alpha:
            return "Generalist", "", top_z, pval

        return top_profile, axis_col, top_z, pval

    assignments = df.apply(_assign, axis=1, result_type="expand")
    assignments.columns = ["profile", "profile_axis", "profile_z", "profile_pvalue"]
    df = pd.concat([df, assignments], axis=1)

    # ── Step 3: Log summary ─────────────────────────────────────────────────
    counts = df["profile"].value_counts()
    logger.info(
        "Assigned %d players to %d profiles (alpha=%.3f, sig=%s):",
        len(df), len(counts), alpha, require_significance,
    )
    for profile in PROFILE_NAMES:
        n = int(counts.get(profile, 0))
        pct = 100.0 * n / max(len(df), 1)
        logger.info("  %-18s n=%4d  (%.1f%%)", profile, n, pct)

    return df


def profile_summary(features_with_profile: pd.DataFrame) -> pd.DataFrame:
    """Return mean of every numeric feature, grouped by profile."""
    numeric_cols = features_with_profile.select_dtypes(include="number").columns
    # Drop noisy/uninteresting columns
    drop_cols = {"profile_z", "profile_pvalue", "n_bidding_boards", "n_declared"}
    keep = [c for c in numeric_cols if c not in drop_cols]
    return (
        features_with_profile.groupby("profile")[keep]
        .mean()
        .round(4)
        .reindex(PROFILE_NAMES)
    )
