"""
src/stage3_agents/nego_agent.py
===============================
A profile-conditioned business-negotiation agent.

This is the cross-domain twin of BridgeAgent. The SAME profile (built from the
SAME Stage 2 bridge skills) now negotiates a business deal. The research
question is whether a profile that behaves a certain way in bridge behaves
analogously here — so it is critical that the *only* thing carried across is the
bridge-derived character card, NOT new personality instructions.

The agent is thin (personality lives in the system prompt, LLM plumbing lives in
BaseAgent). This class only:
  1. formats the scenario + history + current offer into a user prompt,
  2. calls _decide() with a strict JSON schema,
  3. validates/clamps the returned offer to the scenario's legal numeric range.

Temperature note
----------------
Negotiation uses a higher default temperature (0.7) than bridge (0.0): bargaining
benefits from some variety, and unlike bridge we are not chasing call-level
reproducibility — Stage 4 reports distributions over many sessions. Reproducible
analysis comes from persisting every raw output, not from determinism.
"""

from __future__ import annotations

import logging

from src.shared.llm_client import LLMClient
from src.shared.prompts import ProfileSignature, build_negotiation_system_prompt
from src.stage3_agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

# Valid actions the agent may take.
VALID_ACTIONS = {"counter", "accept", "walk_away"}

# JSON schema the LLM must fill.
#
# NOTE: `offer` is modelled as a flat object whose numeric terms are declared
# explicitly. Gemini's structured-output mode returns an EMPTY object for a
# bare {"type": "object"} with no declared properties — so we must name the
# term we expect back. This schema covers the single-term scenarios used in
# Stage 3/4 (price_musd); extend `offer.properties` when richer scenarios are
# added. A free-form fallback (`offer_terms` as a JSON string) is also provided
# so multi-term offers are never silently dropped.
NEGO_RESPONSE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["counter", "accept", "walk_away"]},
        "offer": {
            "type": "object",
            "properties": {
                "price_musd": {"type": "number"},
            },
        },
        "willing_to_close": {"type": "boolean"},
        "reasoning": {"type": "string"},
    },
    "required": ["action", "offer", "willing_to_close", "reasoning"],
}


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp a numeric value into [lo, hi]."""
    return max(lo, min(hi, value))


class NegotiationAgent(BaseAgent):
    """Business-negotiation agent for one profile."""

    def __init__(
        self,
        signature: ProfileSignature,
        client: LLMClient | None = None,
        temperature: float = 0.7,
    ):
        super().__init__(signature, client=client, temperature=temperature)
        # Build the character card once, at construction. NOTE: this is the
        # negotiation card, which explicitly tells the agent its behaviour must
        # FLOW FROM its bridge style (anti-tautology — no new personality).
        self.system_prompt = build_negotiation_system_prompt(signature)

    def respond_to_offer(
        self,
        scenario: dict,
        current_offer: dict | None = None,
        history: list[dict] | None = None,
    ) -> dict:
        """Decide a response to the current offer, in character.

        Args:
            scenario: The deal definition. Expected keys (all optional but
                recommended):
                  "title"        — short scenario name
                  "role"         — which side this agent plays (e.g. "buyer")
                  "description"  — plain-English context
                  "terms"        — dict of negotiable terms, each:
                       {"name": {"min": float, "max": float, "unit": str}}
                  "your_target"  — this side's ideal value(s)
                  "your_limit"   — this side's walk-away threshold
            current_offer: The opponent's current offer (dict of term→value),
                or None on the opening move.
            history: Prior offers/counters (list of dicts), most recent last.

        Returns:
            dict with:
              "action"           — "counter" | "accept" | "walk_away"
              "offer"            — proposed terms (clamped to legal range)
              "willing_to_close" — bool
              "reasoning"        — one-line explanation
              "valid"            — bool, whether the action was recognised
        """
        history = history or []

        user_prompt = self._build_user_prompt(scenario, current_offer, history)

        result = self._decide(
            user_prompt=user_prompt,
            response_schema=NEGO_RESPONSE_SCHEMA,
            purpose="negotiation_turn",
        )

        action = str(result.get("action", "")).strip().lower()
        reasoning = str(result.get("reasoning", "")).strip()
        offer = result.get("offer") or {}
        willing = bool(result.get("willing_to_close", False))

        valid = action in VALID_ACTIONS
        if not valid:
            logger.info(
                "%s returned invalid action %r — defaulting to walk_away",
                self.profile, action,
            )
            return {
                "action": "walk_away",
                "offer": {},
                "willing_to_close": False,
                "reasoning": reasoning or "(fallback: unrecognised action)",
                "valid": False,
            }

        # Clamp any numeric offer terms to the scenario's declared ranges.
        if action == "counter" and isinstance(offer, dict):
            offer = self._clamp_offer(offer, scenario)

        return {
            "action": action,
            "offer": offer if action == "counter" else {},
            "willing_to_close": willing,
            "reasoning": reasoning,
            "valid": True,
        }

    # BaseAgent abstract method.
    def act(self, scenario, current_offer=None, history=None) -> dict:  # type: ignore[override]
        return self.respond_to_offer(scenario, current_offer, history)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _build_user_prompt(
        scenario: dict,
        current_offer: dict | None,
        history: list[dict],
    ) -> str:
        lines: list[str] = []
        lines.append(f"SCENARIO: {scenario.get('title', '(untitled)')}")
        if scenario.get("role"):
            lines.append(f"YOUR ROLE: {scenario['role']}")
        if scenario.get("description"):
            lines.append(f"CONTEXT: {scenario['description']}")

        terms = scenario.get("terms")
        if isinstance(terms, dict) and terms:
            lines.append("NEGOTIABLE TERMS (stay within these ranges):")
            for name, spec in terms.items():
                lo = spec.get("min")
                hi = spec.get("max")
                unit = spec.get("unit", "")
                lines.append(f"  - {name}: {lo}–{hi} {unit}".rstrip())

        if scenario.get("your_target") is not None:
            lines.append(f"YOUR TARGET: {scenario['your_target']}")
        if scenario.get("your_limit") is not None:
            lines.append(f"YOUR WALK-AWAY LIMIT: {scenario['your_limit']}")

        if history:
            lines.append("\nHISTORY (oldest to newest):")
            for h in history:
                lines.append(f"  - {h}")

        if current_offer:
            lines.append(f"\nOPPONENT'S CURRENT OFFER: {current_offer}")
        else:
            lines.append("\nThis is the OPENING move — there is no offer yet.")

        lines.append("\nDecide your response, in character. Respond with JSON only.")
        return "\n".join(lines)

    @staticmethod
    def _clamp_offer(offer: dict, scenario: dict) -> dict:
        """Clamp numeric offer terms into the scenario's declared ranges."""
        terms = scenario.get("terms")
        if not isinstance(terms, dict):
            return offer
        clamped = dict(offer)
        for name, spec in terms.items():
            if name in clamped and isinstance(clamped[name], (int, float)):
                lo = spec.get("min")
                hi = spec.get("max")
                if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
                    clamped[name] = _clamp(float(clamped[name]), lo, hi)
        return clamped
