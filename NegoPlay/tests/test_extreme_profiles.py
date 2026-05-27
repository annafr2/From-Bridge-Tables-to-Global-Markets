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
    """100 players. Mostly average, with 4 obvious extreme players seeded."""
    rng = np.random.default_rng(seed=42)
    n = 100
    df = pd.DataFrame({
        "player_name":         [f"P{i:03d}" for i in range(n)],
        "slam_rate":           rng.normal(0.05, 0.01, n).clip(0, 1),
        "partscore_rate":      rng.normal(0.55, 0.02, n).clip(0, 1),
        "nt_rate":             rng.normal(0.28, 0.02, n).clip(0, 1),
        "penalty_double_rate": rng.normal(0.08, 0.01, n).clip(0, 1),
    })
    # Seed 4 extreme players — one per axis, at index 0..3
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
    # With 10% cutoff → ~40 extreme players (top 10% × 4 axes)
    out10 = assign_extreme_profiles(synthetic_features, extreme_pct=0.10)
    # With 5% cutoff → fewer extreme players
    out05 = assign_extreme_profiles(synthetic_features, extreme_pct=0.05)
    n_extreme_10 = (out10["profile"] != "Generalist").sum()
    n_extreme_05 = (out05["profile"] != "Generalist").sum()
    assert n_extreme_05 < n_extreme_10


def test_missing_required_column_raises():
    bad_df = pd.DataFrame({"player_name": ["A"], "slam_rate": [0.5]})
    with pytest.raises(ValueError, match="missing required columns"):
        assign_extreme_profiles(bad_df)


def test_profile_z_score_is_set(synthetic_features):
    out = assign_extreme_profiles(synthetic_features, extreme_pct=0.10)
    slam_hunter = out[out["profile"] == "Slam Hunter"].iloc[0]
    # The seeded extreme player has slam_rate=0.50 vs mean ~0.05 → z >> 2
    assert slam_hunter["profile_z"] > 2.0


def test_profile_summary_returns_all_profiles_in_order(synthetic_features):
    out = assign_extreme_profiles(synthetic_features, extreme_pct=0.10)
    summary = profile_summary(out)
    # All profiles in canonical order (even if empty)
    assert list(summary.index) == PROFILE_NAMES


def test_profile_axes_map_to_real_columns():
    # Sanity: the axes named in PROFILE_AXES are real feature names
    expected = {"slam_rate", "partscore_rate", "nt_rate", "penalty_double_rate"}
    assert set(PROFILE_AXES.values()) == expected
