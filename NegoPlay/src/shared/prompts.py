"""
src/shared/prompts.py
=====================
Centralised prompt library for NegoPlay agents (Stage 3).

This module turns each player profile into a *system prompt* — the "character
card" the LLM receives before every decision. A system prompt has five parts
(see docs/agent_architecture.md, Section 3):

    1. Identity        — "You are a Slam Hunter."
    2. Core skills     — the 5-7 skills extracted from REAL bridge hands in
                         Stage 2 (NOT hand-written personality words).
    3. Decision rules  — domain-specific (bridge bidding laws OR negotiation).
    4. Output format   — strict JSON, enforced by a response schema.
    5. Few-shot example— one worked example so the model knows the shape.

Why skills come from Stage 2 (the anti-tautology rule)
------------------------------------------------------
If we wrote "be aggressive", the LLM would act aggressively in BOTH bridge and
negotiation simply because we told it to — and any measured cross-domain
alignment would be fake. Instead we inject the bridge-specific skills that the
LLM itself extracted from real hands (e.g. "Aggressive Penalty Doubling"). The
business behaviour must then *emerge* from those bridge skills, not from a
personality label. See docs/agent_architecture.md Section 6.

Public API
----------
    load_profile_signatures(path)        -> dict[str, ProfileSignature]
    build_bridge_system_prompt(sig)      -> str
    build_negotiation_system_prompt(sig) -> str
    PROFILE_NAMES                        -> list[str]
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# ── Defaults ──────────────────────────────────────────────────────────────────

# The Stage 2 output we trust (threshold 0.40, semantically aggregated).
DEFAULT_SIGNATURES_PATH = Path(
    "results/stage2_sample_v2_focused_prompt/skill_profiles_reagg_t040.json"
)

# The four extreme profiles plus the Generalist control.
PROFILE_NAMES: list[str] = [
    "Slam Hunter",
    "Insurance Player",
    "Fighter",
    "NT Specialist",
    "Generalist",
]

# One-line essence per profile (used in the identity line). These summarise the
# Stage 1 *defining metric*, not invented personality — each maps to a measured
# behaviour (slam_rate, partscore_rate, penalty_double_rate, nt_rate).
PROFILE_ESSENCE: dict[str, str] = {
    "Slam Hunter": (
        "an elite bridge player who bids slams far more often than average — "
        "you pursue the biggest contracts when the values justify it"
    ),
    "Insurance Player": (
        "an elite bridge player who stops in safe partscore contracts more "
        "often than average — you prefer a sure small gain over a risky large one"
    ),
    "Fighter": (
        "an elite bridge player who penalty-doubles opponents more often than "
        "average — you punish opponents who overreach"
    ),
    "NT Specialist": (
        "an elite bridge player who plays NoTrump contracts more often than "
        "average — you favour balanced, calculated NT play"
    ),
    "Generalist": (
        "an elite bridge player with no extreme tendency in any direction — "
        "you play balanced, textbook-standard bridge (the baseline style)"
    ),
}


# ── Data container ────────────────────────────────────────────────────────────

@dataclass
class ProfileSignature:
    """A profile and its Stage-2 skills, ready to be turned into a prompt."""

    profile: str
    skills: list[dict] = field(default_factory=list)  # each: name, description, ...
    n_players: int = 0

    def skill_lines(self, max_skills: int = 7) -> str:
        """Render skills as a bulleted block for the system prompt."""
        if not self.skills:
            return "  (no specific skills on record — play balanced, standard bridge)"
        lines = []
        for s in self.skills[:max_skills]:
            name = s.get("name", "").strip()
            desc = s.get("description", "").strip()
            lines.append(f"  - {name}: {desc}")
        return "\n".join(lines)


# ── Loading ───────────────────────────────────────────────────────────────────

def load_profile_signatures(
    path: str | Path = DEFAULT_SIGNATURES_PATH,
) -> dict[str, ProfileSignature]:
    """Load Stage 2 profile signatures from JSON.

    The Generalist profile has no extracted skills (it is the baseline), so we
    synthesise an empty signature for it. All five PROFILE_NAMES are guaranteed
    to be present in the returned dict.

    Args:
        path: Path to the Stage 2 skill_profiles JSON.

    Returns:
        Mapping {profile_name: ProfileSignature} for all five profiles.

    Raises:
        FileNotFoundError: if the signatures file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Stage 2 signatures not found at {p}. Run Stage 2 first "
            "(notebooks/reaggregate_v2.py) or pass an explicit path."
        )

    with p.open(encoding="utf-8") as f:
        data = json.load(f)

    sigs: dict[str, ProfileSignature] = {}
    for sig in data.get("profile_signatures", []):
        name = sig["profile"]
        sigs[name] = ProfileSignature(
            profile=name,
            skills=sig.get("skills", []),
            n_players=sig.get("n_players", 0),
        )

    # Ensure Generalist (and any missing profile) always exists as a baseline.
    for name in PROFILE_NAMES:
        sigs.setdefault(name, ProfileSignature(profile=name, skills=[], n_players=0))

    return sigs


# ── Shared blocks ─────────────────────────────────────────────────────────────

def _identity_block(sig: ProfileSignature) -> str:
    essence = PROFILE_ESSENCE.get(sig.profile, "an elite bridge player")
    return (
        f"You are a \"{sig.profile}\" — {essence}.\n\n"
        "The following skills were observed in your ACTUAL play of real "
        "tournament hands. They define how you think. Stay true to them:\n"
        f"{sig.skill_lines()}"
    )


# ── Bridge system prompt ──────────────────────────────────────────────────────

_BRIDGE_RULES = """\
BRIDGE BIDDING RULES (you must obey these):
- A bid names a level (1-7) and a strain (C, D, H, S, or NT), e.g. "1H", "3NT", "6S".
- Strain rank: C < D < H < S < NT. Level rank: 1 < 2 < ... < 7.
- Each new contract bid must be HIGHER than the last contract bid (higher level,
  or same level and higher strain).
- You may always "Pass". You may "X" (double) the opponents' last contract bid,
  or "XX" (redouble) if they doubled you.
- The auction ends after three consecutive passes following a bid.
- Bid according to your style and the hand you hold — do not invent cards."""

_BRIDGE_EXAMPLE = """\
EXAMPLE (for format only):
Hand: S:AKQ72 H:K4 D:A83 C:Q92  (17 HCP, 5 spades)
Auction so far: [Partner: 1S, Opponent: Pass]
A Slam Hunter, holding strong values and a fit, might explore higher:
{"bid": "3S", "reasoning": "Strong raise with 4-card support and slam interest."}"""

_BRIDGE_SCHEMA_HINT = """\
OUTPUT: return ONLY valid JSON with exactly these keys:
  "bid"       — your call as a string: a contract bid ("4H"), "Pass", "X", or "XX"
  "reasoning" — one short sentence explaining the call in your style"""


def build_bridge_system_prompt(sig: ProfileSignature) -> str:
    """Assemble the full bridge bidding system prompt for one profile."""
    return (
        f"{_identity_block(sig)}\n\n"
        f"{_BRIDGE_RULES}\n\n"
        f"{_BRIDGE_EXAMPLE}\n\n"
        f"{_BRIDGE_SCHEMA_HINT}"
    )


# ── Negotiation system prompt ─────────────────────────────────────────────────

_NEGO_RULES = """\
BUSINESS NEGOTIATION RULES:
- You represent ONE side of a deal. You receive the scenario, the history of
  offers, and the opponent's current offer.
- Each turn you either propose a counter-offer, accept, or walk away.
- Your negotiation behaviour must FLOW FROM your bridge style above — do not
  adopt a new personality. A risk-seeking bridge style implies bold offers;
  a safety-first style implies quick, secure deals; a combative style implies
  hard counter-offers; an analytical style implies data-driven justification.
- Stay within the numeric ranges defined by the scenario. Do not invent terms."""

_NEGO_EXAMPLE = """\
EXAMPLE (for format only):
Scenario: acquiring a startup; your max budget is $10M; their ask is $12M.
A Slam Hunter (bold, big-swing style) might:
{"action": "counter", "offer": {"price_musd": 9.5}, "willing_to_close": false,
 "reasoning": "Open bold and anchor low; I back myself to close a big deal."}"""

_NEGO_SCHEMA_HINT = """\
OUTPUT: return ONLY valid JSON with exactly these keys:
  "action"           — one of "counter", "accept", "walk_away"
  "offer"            — object with the numeric terms you propose (or {} if accept/walk_away)
  "willing_to_close" — boolean: are you ready to close on these terms?
  "reasoning"        — one short sentence explaining the move in your style"""


def build_negotiation_system_prompt(sig: ProfileSignature) -> str:
    """Assemble the full negotiation system prompt for one profile."""
    return (
        f"{_identity_block(sig)}\n\n"
        f"{_NEGO_RULES}\n\n"
        f"{_NEGO_EXAMPLE}\n\n"
        f"{_NEGO_SCHEMA_HINT}"
    )


__all__ = [
    "ProfileSignature",
    "PROFILE_NAMES",
    "PROFILE_ESSENCE",
    "DEFAULT_SIGNATURES_PATH",
    "load_profile_signatures",
    "build_bridge_system_prompt",
    "build_negotiation_system_prompt",
]
