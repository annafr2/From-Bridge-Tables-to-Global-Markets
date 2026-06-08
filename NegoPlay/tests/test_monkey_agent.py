"""
tests/test_monkey_agent.py
==========================
Tests for the random "monkey" baseline agent (Task T-A).

The monkey has no LLM, so these run instantly with no API key and no cost.
We verify the three properties the baseline relies on:
  1. Every bridge call it makes is LEGAL for the given auction.
  2. Every negotiation offer is INSIDE the scenario's legal price range.
  3. It is REPRODUCIBLE (same seed -> same sequence; different seed -> differs).
"""

from __future__ import annotations

from src.stage3_agents.bridge_agent import is_legal_call
from src.stage4_simulate.monkey_agent import MONKEY_PROFILE, MonkeyAgent


# ── Bridge: legality ──────────────────────────────────────────────────────────

def test_monkey_bids_are_always_legal_over_many_auctions():
    """Across many random auctions the monkey never returns an illegal call."""
    monkey = MonkeyAgent(seed=1)
    auctions = [
        [],
        ["1C"],
        ["1NT"],
        ["2C", "Pass"],
        ["1S", "X"],
        ["3NT", "X"],
        ["7NT"],          # nothing higher can be bid -> Pass/X must be chosen
        ["1H", "Pass", "2H", "Pass"],
    ]
    for auction in auctions:
        for _ in range(50):
            out = monkey.make_bid({"S": "AK", "H": "Q5", "D": "J9", "C": "T8"}, auction)
            assert out["legal"] is True
            assert is_legal_call(out["bid"], auction), (
                f"illegal call {out['bid']!r} for auction {auction}"
            )


def test_monkey_after_7nt_can_only_pass_or_double():
    """When 7NT is on the table, the only legal calls are Pass (or X)."""
    monkey = MonkeyAgent(seed=7)
    for _ in range(30):
        out = monkey.make_bid(None, ["7NT"])
        assert out["bid"] in {"Pass", "X"}


def test_monkey_bid_shape_matches_bridge_agent():
    """Return dict has the same keys the bridge runner reads."""
    out = MonkeyAgent().make_bid(None, ["1C"])
    assert set(out) >= {"bid", "reasoning", "legal", "raw_bid"}


# ── Negotiation: in-range offers + valid actions ──────────────────────────────

_SCENARIO = {
    "title": "test deal",
    "terms": {"price_musd": {"min": 8.0, "max": 13.0, "unit": "M USD"}},
}


def test_monkey_offers_are_within_legal_range():
    monkey = MonkeyAgent(seed=2)
    for _ in range(200):
        resp = monkey.respond_to_offer(_SCENARIO, current_offer={"price_musd": 13.0})
        assert resp["action"] in {"counter", "accept", "walk_away"}
        if resp["action"] == "counter":
            price = resp["offer"]["price_musd"]
            assert 8.0 <= price <= 13.0, f"offer {price} out of [8, 13]"
        else:
            assert resp["offer"] is None


def test_monkey_handles_missing_terms_without_crashing():
    """A malformed scenario must not crash the monkey (it falls back)."""
    resp = MonkeyAgent().respond_to_offer({"title": "no terms"})
    assert resp["action"] in {"counter", "accept", "walk_away"}


# ── Reproducibility ───────────────────────────────────────────────────────────

def test_same_seed_reproduces_bid_sequence():
    a = MonkeyAgent(seed=123)
    b = MonkeyAgent(seed=123)
    seq_a = [a.make_bid(None, ["1C"])["bid"] for _ in range(40)]
    seq_b = [b.make_bid(None, ["1C"])["bid"] for _ in range(40)]
    assert seq_a == seq_b


def test_different_seeds_differ():
    a = MonkeyAgent(seed=1)
    b = MonkeyAgent(seed=2)
    seq_a = [a.make_bid(None, ["1C"])["bid"] for _ in range(40)]
    seq_b = [b.make_bid(None, ["1C"])["bid"] for _ in range(40)]
    assert seq_a != seq_b


def test_profile_name():
    assert MonkeyAgent().profile == MONKEY_PROFILE
