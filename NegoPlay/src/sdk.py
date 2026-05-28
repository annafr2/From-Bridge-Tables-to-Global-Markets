"""
NegoPlay SDK — Single Entry Point
==================================
All public operations go through this module (Dr. Segal methodology).

Stage 1 — Profile discovery
    build_profiles(data_path) → DataFrame with 5 player profiles
"""

from pathlib import Path

import pandas as pd

from src.shared.data_loader import load_matches
from src.stage1_clustering.extreme_profiles import (
    DEFAULT_EXTREME_PCT,
    PROFILE_NAMES,
    assign_extreme_profiles,
    profile_summary,
)
from src.stage1_clustering.features import compute_player_features


def build_profiles(
    data_path: str | Path,
    min_boards: int = 50,
    min_bidding_boards: int = 50,
    extreme_pct: float = DEFAULT_EXTREME_PCT,
    alpha: float = 0.05,
    require_significance: bool = True,
) -> pd.DataFrame:
    """End-to-end Stage 1: raw CSV → per-player profile assignments.

    Args:
        data_path: path to all_matches_full.csv (149K rows).
        min_boards: minimum declared boards required per player (default 50,
            raised from 20 in May 2026 after sample-size review).
        min_bidding_boards: minimum boards with full bidding required (default 50).
        extreme_pct: top-percentile threshold defining "extreme" (0.10 = top 10%).
        alpha: significance threshold for the binomial test (default 0.05).
        require_significance: if True (default), a player must be both in the
            top extreme_pct AND statistically distinguishable from the
            baseline at p < alpha to be assigned to a profile.

    Returns:
        DataFrame, one row per player, with all features plus:
            profile         — one of {Slam Hunter, Insurance Player,
                              Fighter, NT Specialist, Generalist}
            profile_axis    — the feature that defined the profile
            profile_z       — z-score on that axis
            profile_pvalue  — binomial p-value (lower = stronger evidence)
    """
    df = load_matches(data_path)
    features = compute_player_features(
        df,
        min_boards=min_boards,
        min_bidding_boards=min_bidding_boards,
    )
    profiles = assign_extreme_profiles(
        features,
        extreme_pct=extreme_pct,
        alpha=alpha,
        require_significance=require_significance,
    )
    return profiles


class NegoPlaySDK:
    """Main SDK class — single contract for all NegoPlay operations.

    Usage:
        sdk = NegoPlaySDK(data_path="data/processed/all_matches_full.csv")
        profiles = sdk.build_profiles()
        summary = sdk.profile_summary(profiles)
    """

    PROFILE_NAMES = PROFILE_NAMES

    def __init__(self, data_path: str | Path | None = None) -> None:
        self.data_path = data_path

    def build_profiles(self, **kwargs) -> pd.DataFrame:
        """Run Stage 1 end-to-end. See module-level build_profiles()."""
        if self.data_path is None:
            raise ValueError("data_path was not provided to NegoPlaySDK(...)")
        return build_profiles(self.data_path, **kwargs)

    @staticmethod
    def profile_summary(profiles: pd.DataFrame) -> pd.DataFrame:
        """Mean of every feature, grouped by profile."""
        return profile_summary(profiles)
