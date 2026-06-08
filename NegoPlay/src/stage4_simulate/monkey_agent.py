"""
src/stage4_simulate/monkey_agent.py
===================================
The "random monkey" baseline agent (Task T-A, post-supervisor meeting).

Rationale (Rami, statistician): to prove that a profile's skill is *real* we
need a zero-skill FLOOR to measure against. The monkey has **no strategy and no
knowledge** — it makes uniformly random *legal* actions. A profiled agent that
cannot beat the monkey, in many deals, has not demonstrated skill.

Design:
  - **No LLM, no cost.** All decisions are random draws from a seeded RNG, so the
    monkey is free to run and fully reproducible (same seed -> same sequence).
  - **Duck-typed**, not a BaseAgent subclass: BaseAgent requires a ProfileSignature
    and an LLM. The monkey deliberately has neither. It exposes the SAME public
    methods the runners call, so it drops straight into bridge_runner /
    negotiation without changing them:
        make_bid(hand, auction)              -> like BridgeAgent
        respond_to_offer(scenario, offer, …) -> like NegotiationAgent
  - **Legal-only.** Bridge calls are filtered through the real `is_legal_call`
    (no illegal/absurd auctions). Negotiation offers are drawn inside the
    scenario's legal price range.

The monkey is the floor for the normalized skill scale
    skill% = (agent - monkey) / (perfect - monkey)
computed by the baseline test (Task T-B) and the double-dummy ceiling (T-C).
"""

from __future__ import annotations

import random
from typing import Any

from src.stage3_agents.bridge_agent import is_legal_call

# All 35 contract bids, lowest to highest, plus the non-contract calls.
_STRAINS = ["C", "D", "H", "S", "NT"]
_ALL_CONTRACT_BIDS = [f"{lvl}{strain}" for lvl in range(1, 8) for strain in _STRAINS]
_ALL_CALLS = ["Pass", "X", "XX", *_ALL_CONTRACT_BIDS]

# Negotiation action space (same vocabulary the NegotiationAgent uses).
_NEGO_ACTIONS = ["counter", "accept", "walk_away"]
# Weights: the monkey mostly engages (counter) but sometimes caves or quits at
# random. This is still strategy-free — it just keeps the floor from trivially
# walking away every time (which would make EVERY profile look good by default).
_NEGO_ACTION_WEIGHTS = [0.70, 0.20, 0.10]

MONKEY_PROFILE = "Monkey (random)"


class MonkeyAgent:
    """A zero-skill agent that makes uniformly random *legal* actions.

    Args:
        seed: RNG seed for reproducibility. The same seed reproduces the exact
              sequence of random decisions across the whole experiment.
    """

    def __init__(self, seed: int = 42):
        self.profile = MONKEY_PROFILE
        self._rng = random.Random(seed)

    # ── Bridge: random legal call ─────────────────────────────────────────────

    def make_bid(
        self,
        hand: dict[str, str] | list[str] | None = None,
        auction: list[str] | None = None,
        partner_note: str | None = None,
    ) -> dict[str, Any]:
        """Return a uniformly random LEGAL call for the current auction.

        The hand and partner_note are ignored on purpose — the monkey has no idea
        what its cards mean. Mirrors BridgeAgent.make_bid's return shape so the
        bridge runner (and the multi-round auction runner) can use it unchanged.
        """
        auction = auction or []
        legal = [c for c in _ALL_CALLS if is_legal_call(c, auction)]
        # "Pass" is always legal, so `legal` is never empty.
        bid = self._rng.choice(legal)
        return {
            "bid": bid,
            "reasoning": "(random monkey: no strategy)",
            "legal": True,
            "raw_bid": bid,
        }

    # ── Negotiation: random action + random in-range offer ────────────────────

    def respond_to_offer(
        self,
        scenario: dict,
        current_offer: dict | None = None,
        history: list[dict] | None = None,
    ) -> dict[str, Any]:
        """Return a strategy-free response: random action, random legal price.

        Mirrors NegotiationAgent.respond_to_offer's return shape so the
        negotiation runner can use it unchanged.
        """
        action = self._rng.choices(_NEGO_ACTIONS, weights=_NEGO_ACTION_WEIGHTS, k=1)[0]

        offer: dict | None = None
        if action == "counter":
            lo, hi = self._price_range(scenario)
            offer = {"price_musd": round(self._rng.uniform(lo, hi), 3)}

        return {
            "action": action,
            "offer": offer,
            "reasoning": "(random monkey: no strategy)",
        }

    @staticmethod
    def _price_range(scenario: dict) -> tuple[float, float]:
        """Extract the legal [min, max] price from the scenario terms.

        Falls back to a wide default if the scenario shape is unexpected, so the
        monkey never crashes a run.
        """
        terms = (scenario or {}).get("terms", {})
        price = terms.get("price_musd", {}) if isinstance(terms, dict) else {}
        lo = price.get("min", 0.0)
        hi = price.get("max", lo + 1.0)
        if hi < lo:
            lo, hi = hi, lo
        return float(lo), float(hi)

    # Convenience so the monkey can stand in wherever `.act(...)` is expected.
    def act(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        if "scenario" in kwargs or (args and isinstance(args[0], dict) and "terms" in args[0]):
            return self.respond_to_offer(*args, **kwargs)
        return self.make_bid(*args, **kwargs)

    def __repr__(self) -> str:
        return f"<MonkeyAgent profile={self.profile!r} (random, no LLM)>"


__all__ = ["MonkeyAgent", "MONKEY_PROFILE"]
