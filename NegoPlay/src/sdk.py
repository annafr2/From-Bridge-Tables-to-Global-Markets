"""
NegoPlay SDK — Single Entry Point
==================================
All public operations go through this module (Dr. Segal methodology).

Stage 1 — Profile discovery
    build_profiles(data_path) → DataFrame with 5 player profiles

Stage 3 — Agent construction
    build_bridge_agent(profile)  → one BridgeAgent
    build_bridge_agents()        → dict of all 5 BridgeAgents
    build_nego_agent(profile)    → one NegotiationAgent
    build_nego_agents()          → dict of all 5 NegotiationAgents
"""

from pathlib import Path

import pandas as pd

from src.shared.data_loader import load_matches
from src.shared.llm_client import LLMClient
from src.shared.prompts import (
    PROFILE_NAMES as AGENT_PROFILE_NAMES,
    load_profile_signatures,
)
from src.stage1_clustering.extreme_profiles import (
    DEFAULT_EXTREME_PCT,
    PROFILE_NAMES,
    assign_extreme_profiles,
    profile_summary,
)
from src.stage1_clustering.features import compute_player_features
from src.stage3_agents.bridge_agent import BridgeAgent
from src.stage3_agents.nego_agent import NegotiationAgent


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


# ── Stage 3: Agent construction ───────────────────────────────────────────────

def build_bridge_agent(
    profile: str,
    signatures_path: str | Path | None = None,
    client: LLMClient | None = None,
    temperature: float = 0.0,
) -> BridgeAgent:
    """Build ONE bridge-bidding agent for the given profile.

    Args:
        profile: One of {Slam Hunter, Insurance Player, Fighter, NT Specialist,
            Generalist}.
        signatures_path: Path to the Stage 2 skill_profiles JSON. Defaults to
            the validated threshold-0.40 output.
        client: Optional shared LLMClient (share one across agents to keep a
            single budget/cost log for a whole game). A new one is created if
            omitted.
        temperature: Sampling temperature (0.0 = reproducible bridge bidding;
            same hand → same bid across runs).

    Returns:
        A ready-to-use BridgeAgent.

    Raises:
        ValueError: if `profile` is not a known profile name.
    """
    if profile not in AGENT_PROFILE_NAMES:
        raise ValueError(
            f"Unknown profile {profile!r}. Must be one of {AGENT_PROFILE_NAMES}."
        )
    sigs = (
        load_profile_signatures(signatures_path)
        if signatures_path is not None
        else load_profile_signatures()
    )
    return BridgeAgent(sigs[profile], client=client, temperature=temperature)


def build_bridge_agents(
    signatures_path: str | Path | None = None,
    client: LLMClient | None = None,
    temperature: float = 0.0,
) -> dict[str, BridgeAgent]:
    """Build ALL five bridge agents, sharing one LLMClient by default.

    Sharing a single client means cost is logged and budget-capped across the
    whole set of agents (important once they play full games).

    Returns:
        Mapping {profile_name: BridgeAgent} for all five profiles.
    """
    sigs = (
        load_profile_signatures(signatures_path)
        if signatures_path is not None
        else load_profile_signatures()
    )
    shared = client or LLMClient()
    return {
        name: BridgeAgent(sigs[name], client=shared, temperature=temperature)
        for name in AGENT_PROFILE_NAMES
    }


def build_nego_agent(
    profile: str,
    signatures_path: str | Path | None = None,
    client: LLMClient | None = None,
    temperature: float = 0.7,
) -> NegotiationAgent:
    """Build ONE business-negotiation agent for the given profile.

    The agent's character card is built from the SAME Stage 2 bridge skills as
    its BridgeAgent twin — only the domain rules differ. This is the mechanism
    that lets us test cross-domain behavioural alignment (the core research
    question) without injecting new personality.

    Args:
        profile: One of the five profile names.
        signatures_path: Path to the Stage 2 skill_profiles JSON (defaults to
            the validated threshold-0.40 output).
        client: Optional shared LLMClient (one budget log across agents).
        temperature: Sampling temperature (0.7 = some bargaining variety;
            reproducibility comes from persisting outputs, not determinism).

    Returns:
        A ready-to-use NegotiationAgent.

    Raises:
        ValueError: if `profile` is not a known profile name.
    """
    if profile not in AGENT_PROFILE_NAMES:
        raise ValueError(
            f"Unknown profile {profile!r}. Must be one of {AGENT_PROFILE_NAMES}."
        )
    sigs = (
        load_profile_signatures(signatures_path)
        if signatures_path is not None
        else load_profile_signatures()
    )
    return NegotiationAgent(sigs[profile], client=client, temperature=temperature)


def build_nego_agents(
    signatures_path: str | Path | None = None,
    client: LLMClient | None = None,
    temperature: float = 0.7,
) -> dict[str, NegotiationAgent]:
    """Build ALL five negotiation agents, sharing one LLMClient by default.

    Returns:
        Mapping {profile_name: NegotiationAgent} for all five profiles.
    """
    sigs = (
        load_profile_signatures(signatures_path)
        if signatures_path is not None
        else load_profile_signatures()
    )
    shared = client or LLMClient()
    return {
        name: NegotiationAgent(sigs[name], client=shared, temperature=temperature)
        for name in AGENT_PROFILE_NAMES
    }


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

    @staticmethod
    def build_bridge_agent(profile: str, **kwargs) -> BridgeAgent:
        """Build one bridge agent. See module-level build_bridge_agent()."""
        return build_bridge_agent(profile, **kwargs)

    @staticmethod
    def build_bridge_agents(**kwargs) -> dict[str, BridgeAgent]:
        """Build all five bridge agents. See module-level build_bridge_agents()."""
        return build_bridge_agents(**kwargs)

    @staticmethod
    def build_nego_agent(profile: str, **kwargs) -> NegotiationAgent:
        """Build one negotiation agent. See module-level build_nego_agent()."""
        return build_nego_agent(profile, **kwargs)

    @staticmethod
    def build_nego_agents(**kwargs) -> dict[str, NegotiationAgent]:
        """Build all five negotiation agents. See module-level build_nego_agents()."""
        return build_nego_agents(**kwargs)
