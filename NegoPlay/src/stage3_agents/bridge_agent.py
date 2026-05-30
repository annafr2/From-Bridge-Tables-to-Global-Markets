"""
src/stage3_agents/bridge_agent.py
=================================
A profile-conditioned bridge-bidding agent.

Given a hand and the auction so far, the agent returns its next call
("1H", "3NT", "Pass", "X", "XX") plus a one-line reasoning, decided in the
character of its profile (Slam Hunter, Insurance Player, Fighter, NT Specialist,
or the Generalist baseline).

The agent is deliberately thin: all the personality lives in the system prompt
(built by src/shared/prompts.py from real Stage 2 skills), and all the LLM
plumbing lives in BaseAgent. This class only:
  1. formats the hand + auction into a user prompt,
  2. calls the shared _decide() choke-point with a strict JSON schema,
  3. validates that the returned call is legal for the current auction.

Legality is checked locally (no LLM) so an illegal bid can be caught and the
caller can re-prompt or fall back to Pass.
"""

from __future__ import annotations

import logging
import re

from src.shared.llm_client import LLMClient
from src.shared.prompts import ProfileSignature, build_bridge_system_prompt
from src.stage3_agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

# Strain rank: clubs lowest, NoTrump highest.
_STRAIN_RANK: dict[str, int] = {"C": 0, "D": 1, "H": 2, "S": 3, "NT": 4}

# A contract bid like "1C", "3NT", "7S".
_BID_RE = re.compile(r"^([1-7])(C|D|H|S|NT)$")

# JSON schema the LLM must fill (provider-agnostic shape).
BRIDGE_BID_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "bid": {"type": "string"},
        "reasoning": {"type": "string"},
    },
    "required": ["bid", "reasoning"],
}


# ── Auction-legality helpers (pure, no LLM) ───────────────────────────────────

def _contract_rank(bid: str) -> int | None:
    """Map a contract bid to a sortable integer; None if not a contract bid.

    Rank = level * 5 + strain_rank, so "1C"=5, "1NT"=9, "2C"=10, "7NT"=39.
    """
    m = _BID_RE.match(bid)
    if not m:
        return None
    level = int(m.group(1))
    strain = m.group(2)
    return level * 5 + _STRAIN_RANK[strain]


def _last_contract_bid(auction: list[str]) -> str | None:
    """Return the most recent actual contract bid (ignoring Pass/X/XX)."""
    for call in reversed(auction):
        if _BID_RE.match(call):
            return call
    return None


def is_legal_call(call: str, auction: list[str]) -> bool:
    """Check whether `call` is legal given the auction so far.

    Rules enforced:
      - "Pass" is always legal.
      - A contract bid is legal only if it outranks the last contract bid.
      - "X" (double) is legal only if the opponents' last action was a contract
        bid that has not already been doubled (simplified: last call is a bid).
      - "XX" (redouble) is legal only if the last call was "X".

    This is a pragmatic subset of the Laws — enough to keep simulated auctions
    valid without modelling seat ownership in full.
    """
    if not isinstance(call, str):
        return False
    call = call.strip()

    if call.lower() == "pass":
        return True

    if call == "X":
        last = auction[-1] if auction else None
        return last is not None and _BID_RE.match(last) is not None

    if call == "XX":
        last = auction[-1] if auction else None
        return last == "X"

    rank = _contract_rank(call)
    if rank is None:
        return False  # not a recognised call
    last_bid = _last_contract_bid(auction)
    if last_bid is None:
        return True  # opening bid — any contract bid is legal
    return rank > _contract_rank(last_bid)


# ── The agent ─────────────────────────────────────────────────────────────────

def _format_hand(hand: dict[str, str] | list[str]) -> str:
    """Render a hand for the prompt.

    Accepts either:
      - dict like {"S": "AKQ72", "H": "K4", "D": "A83", "C": "Q92"}
      - list like ["SA","SK",...] or ["AS","KS",...] (joined as-is)
    """
    if isinstance(hand, dict):
        order = ["S", "H", "D", "C"]
        return "  ".join(f"{s}:{hand.get(s, '-') or '-'}" for s in order)
    return " ".join(hand)


class BridgeAgent(BaseAgent):
    """Bridge-bidding agent for one profile."""

    def __init__(
        self,
        signature: ProfileSignature,
        client: LLMClient | None = None,
        temperature: float = 0.3,
    ):
        super().__init__(signature, client=client, temperature=temperature)
        # Build the character card once, at construction.
        self.system_prompt = build_bridge_system_prompt(signature)

    def make_bid(
        self,
        hand: dict[str, str] | list[str],
        auction: list[str] | None = None,
    ) -> dict:
        """Decide the next call for `hand` given `auction` so far.

        Args:
            hand:    The agent's 13 cards (dict by suit, or list of cards).
            auction: Ordered list of prior calls, e.g. ["1S", "Pass", "2H"].
                     Empty/None means the agent is the opener.

        Returns:
            dict with:
              "bid"       — the chosen call (guaranteed legal; Pass on fallback)
              "reasoning" — one-line explanation
              "legal"     — bool, whether the LLM's first choice was legal
              "raw_bid"   — what the LLM originally returned (for auditing)
        """
        auction = auction or []
        auction_str = " ".join(auction) if auction else "(you are the opener)"

        user_prompt = (
            f"Your hand:\n  {_format_hand(hand)}\n\n"
            f"Auction so far (left to right): {auction_str}\n\n"
            "Decide your next call, in character. Respond with JSON only."
        )

        result = self._decide(
            user_prompt=user_prompt,
            response_schema=BRIDGE_BID_SCHEMA,
            purpose="bridge_bid",
        )

        raw_bid = str(result.get("bid", "")).strip()
        reasoning = str(result.get("reasoning", "")).strip()

        # Normalise common variants (e.g. "pass" -> "Pass", "x" -> "X").
        normalised = _normalise_call(raw_bid)
        legal = is_legal_call(normalised, auction)

        if not legal:
            logger.info(
                "%s returned illegal call %r given auction %s — falling back to Pass",
                self.profile, raw_bid, auction,
            )
            return {
                "bid": "Pass",
                "reasoning": reasoning or "(fallback: illegal call replaced by Pass)",
                "legal": False,
                "raw_bid": raw_bid,
            }

        return {
            "bid": normalised,
            "reasoning": reasoning,
            "legal": True,
            "raw_bid": raw_bid,
        }

    # BaseAgent's abstract method — delegate to make_bid.
    def act(self, hand, auction=None) -> dict:  # type: ignore[override]
        return self.make_bid(hand, auction)


def _normalise_call(call: str) -> str:
    """Canonicalise an LLM call string to NegoPlay convention."""
    c = call.strip()
    low = c.lower()
    if low in ("pass", "p"):
        return "Pass"
    if low in ("x", "dbl", "double"):
        return "X"
    if low in ("xx", "rdbl", "redouble"):
        return "XX"
    # Contract bids: upper-case the strain, keep level.
    m = re.match(r"^([1-7])\s*(c|d|h|s|nt|n)$", low)
    if m:
        strain = m.group(2).upper()
        if strain == "N":
            strain = "NT"
        return f"{m.group(1)}{strain}"
    return c  # leave anything else unchanged (will fail legality)
