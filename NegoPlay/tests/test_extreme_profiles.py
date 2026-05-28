"""
tests/test_extreme_profiles.py
Tests for the extreme-profile assignment used in Stage 1.
"""

import numpy as np
import pandas as pd
import pytest

from src.stage1_clustering.extreme_profiles import (
    PROFILE_AXES,
    PROFILE_NAMES,
    assign_extreme_profiles,
    profile_summary,
)


@pytest.fixture
def synthetic_features() -> pd.DataFrame:
    """100 players. Mostly average, with 4 obvious extreme players seeded.

    Each player gets n_declared=100 and n_bidding_boards=100 so the binomial
    test will have enough power to confirm the seeded extreme rates.
    """
    rng = np.random.default_rng(seed=42)
    n = 100
    df = pd.DataFrame({
        "player_name":         [f"P{i:03d}" for i in range(n)],
        "slam_rate":           rng.normal(0.05, 0.01, n).clip(0, 1),
        "partscore_rate":      rng.normal(0.55, 0.02, n).clip(0, 1),
        "nt_rate":             rng.normal(0.28, 0.02, n).clip(0, 1),
        "penalty_double_rate": rng.normal(0.08, 0.01, n).clip(0, 1),
        "n_declared":          np.full(n, 100, dtype=int),
        "n_bidding_boards":    np.full(n, 100, dtype=int),
    })
    # Seed 4 extreme players — one per axis, at index 0..3.
    # The rates here are far enough from the baseline that the binomial
    # test with n=100 will yield p << 0.05.
    df.loc[0, "slam_rate"]           = 0.50  # clear Slam Hunter
    df.loc[1, "partscore_rate"]      = 0.90  # clear Insurance Player
    df.loc[2, "penalty_double_rate"] = 0.40  # clear Fighter
    df.loc[3, "nt_rate"]             = 0.80  # clear NT Specialist
    return df


def test_assignment_returns_all_5_profile_types(synthetic_features):
    out = assign_extreme_profiles(synthetic_features, extreme_pct=0.10)
    present = set(out["profile"].unique())
    # All 4 extremes must appear, plus Generalist
    assert "Slam Hunter" in present
    assert "Insurance Player" in present
    assert "Fighter" in present
    assert "NT Specialist" in present
    assert "Generalist" in present


def test_seeded_extreme_player_assigned_correctly(synthetic_features):
    out = assign_extreme_profiles(synthetic_features, extreme_pct=0.10)
    assert out.loc[0, "profile"] == "Slam Hunter"
    assert out.loc[1, "profile"] == "Insurance Player"
    assert out.loc[2, "profile"] == "Fighter"
    assert out.loc[3, "profile"] == "NT Specialist"


def test_extreme_pct_controls_threshold(synthetic_features):
    # Disable significance to isolate the effect of the percentile cutoff
    # (otherwise the binomial filter dominates and both runs collapse to the
    # 4 seeded extreme players).
    out10 = assign_extreme_profiles(
        synthetic_features, extreme_pct=0.10, require_significance=False,
    )
    out05 = assign_extreme_profiles(
        synthetic_features, extreme_pct=0.05, require_significance=False,
    )
    n_extreme_10 = (out10["profile"] != "Generalist").sum()
    n_extreme_05 = (out05["profile"] != "Generalist").sum()
    assert n_extreme_05 < n_extreme_10


def test_missing_required_column_raises():
    bad_df = pd.DataFrame({"player_name": ["A"], "slam_rate": [0.5]})
    with pytest.raises(ValueError, match="missing required axis columns"):
        assign_extreme_profiles(bad_df)


def test_missing_denominator_column_raises():
    """If significance is required, missing denominator must raise clearly."""
    df = pd.DataFrame({
        "player_name":         ["A"],
        "slam_rate":           [0.5],
        "partscore_rate":      [0.5],
        "nt_rate":             [0.5],
        "penalty_double_rate": [0.5],
        # n_declared / n_bidding_boards missing
    })
    with pytest.raises(ValueError, match="denominator columns"):
        assign_extreme_profiles(df, require_significance=True)


def test_profile_z_score_is_set(synthetic_features):
    out = assign_extreme_profiles(synthetic_features, extreme_pct=0.10)
    slam_hunter = out[out["profile"] == "Slam Hunter"].iloc[0]
    # The seeded extreme player has slam_rate=0.50 vs mean ~0.05 → z >> 2
    assert slam_hunter["profile_z"] > 2.0


def test_profile_pvalue_is_significant_for_seeded_extreme(synthetic_features):
    """The seeded extreme players must pass the p<0.05 binomial test."""
    out = assign_extreme_profiles(synthetic_features, extreme_pct=0.10)
    slam_hunter = out[out["profile"] == "Slam Hunter"].iloc[0]
    fighter = out[out["profile"] == "Fighter"].iloc[0]
    assert slam_hunter["profile_pvalue"] < 0.05
    assert fighter["profile_pvalue"] < 0.05


def test_small_sample_extreme_player_demoted_to_generalist():
    """A player with high rate but tiny sample should NOT become a profile.

    Player A: slam_rate=0.15 with only 10 declared boards (≈1.5 slams).
    Even though 15% > baseline 5%, with n=10 the binomial test is not
    significant (p ≈ 0.10). The player must be assigned to Generalist.
    """
    rng = np.random.default_rng(seed=42)
    n = 50
    df = pd.DataFrame({
        "player_name":         [f"P{i:03d}" for i in range(n)],
        "slam_rate":           rng.normal(0.05, 0.005, n).clip(0, 1),
        "partscore_rate":      rng.normal(0.55, 0.02, n).clip(0, 1),
        "nt_rate":             rng.normal(0.28, 0.02, n).clip(0, 1),
        "penalty_double_rate": rng.normal(0.08, 0.01, n).clip(0, 1),
        "n_declared":          np.full(n, 100, dtype=int),
        "n_bidding_boards":    np.full(n, 100, dtype=int),
    })
    # Player 0: extreme rate but tiny sample → should be demoted to Generalist
    df.loc[0, "slam_rate"] = 0.15
    df.loc[0, "n_declared"] = 10

    out = assign_extreme_profiles(df, extreme_pct=0.10, require_significance=True)
    assert out.loc[0, "profile"] == "Generalist"


def test_require_significance_false_reproduces_old_behaviour():
    """When require_significance=False, the small-sample player can still
    be classified — matching the pre-May-2026 behaviour."""
    rng = np.random.default_rng(seed=42)
    n = 50
    df = pd.DataFrame({
        "player_name":         [f"P{i:03d}" for i in range(n)],
        "slam_rate":           rng.normal(0.05, 0.005, n).clip(0, 1),
        "partscore_rate":      rng.normal(0.55, 0.02, n).clip(0, 1),
        "nt_rate":             rng.normal(0.28, 0.02, n).clip(0, 1),
        "penalty_double_rate": rng.normal(0.08, 0.01, n).clip(0, 1),
        "n_declared":          np.full(n, 100, dtype=int),
        "n_bidding_boards":    np.full(n, 100, dtype=int),
    })
    df.loc[0, "slam_rate"] = 0.15
    df.loc[0, "n_declared"] = 10

    out = assign_extreme_profiles(
        df, extreme_pct=0.10, require_significance=False,
    )
    # With significance disabled, the extreme rate (top 10%) is sufficient
    assert out.loc[0, "profile"] == "Slam Hunter"


def test_profile_summary_returns_all_profiles_in_order(synthetic_features):
    out = assign_extreme_profiles(synthetic_features, extreme_pct=0.10)
    summary = profile_summary(out)
    # All profiles in canonical order (even if empty)
    assert list(summary.index) == PROFILE_NAMES


def test_profile_axes_map_to_real_columns():
    # Sanity: the axes named in PROFILE_AXES are real feature names
    expected = {"slam_rate", "partscore_rate", "nt_rate", "penalty_double_rate"}
    assert set(PROFILE_AXES.values()) == expected
