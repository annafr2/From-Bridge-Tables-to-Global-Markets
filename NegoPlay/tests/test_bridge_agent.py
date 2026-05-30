"""
tests/test_bridge_agent.py
==========================
Tests for the Stage 3 bridge agent.

Two layers:
  1. Pure auction-legality logic (is_legal_call, _normalise_call) — no LLM.
  2. make_bid() with a MOCKED LLMClient — verifies prompt wiring, JSON parsing,
     and the illegal-bid fallback, all without any API call or cost.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.shared.prompts import ProfileSignature
from src.stage3_agents.bridge_agent import (
    BridgeAgent,
    _normalise_call,
    is_legal_call,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def slam_hunter_sig() -> ProfileSignature:
    return ProfileSignature(
        profile="Slam Hunter",
        skills=[{"name": "Aggressive Slam Bidding",
                 "description": "Pushes to slam on strong values."}],
        n_players=5,
    )


def _agent_with_mocked_llm(sig: ProfileSignature, bid: str, reasoning: str = "ok"):
    """Build a BridgeAgent whose LLMClient returns a fixed JSON decision."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.json = {"bid": bid, "reasoning": reasoning}
    mock_response.text = f'{{"bid": "{bid}", "reasoning": "{reasoning}"}}'
    mock_client.generate.return_value = mock_response
    return BridgeAgent(sig, client=mock_client)


# ── Legality: contract bids must ascend ───────────────────────────────────────

def test_opening_bid_is_legal_on_empty_auction():
    assert is_legal_call("1C", [])
    assert is_legal_call("7NT", [])


def test_higher_bid_is_legal():
    assert is_legal_call("2C", ["1NT"])      # 2-level beats 1-level
    assert is_legal_call("1NT", ["1S"])      # same level, NT beats spades


def test_lower_or_equal_bid_is_illegal():
    assert not is_legal_call("1S", ["1NT"])  # spades below NT at level 1
    assert not is_legal_call("1H", ["2C"])   # level 1 below level 2
    assert not is_legal_call("2H", ["2H"])   # equal is not higher


def test_pass_always_legal():
    assert is_legal_call("Pass", [])
    assert is_legal_call("Pass", ["1S", "2D", "3NT"])


# ── Legality: double / redouble ───────────────────────────────────────────────

def test_double_legal_only_after_a_contract_bid():
    assert is_legal_call("X", ["1S"])
    assert not is_legal_call("X", [])          # nothing to double
    assert not is_legal_call("X", ["1S", "Pass"])  # last call was Pass


def test_redouble_legal_only_after_double():
    assert is_legal_call("XX", ["1S", "X"])
    assert not is_legal_call("XX", ["1S"])     # no double to redouble


def test_garbage_call_is_illegal():
    assert not is_legal_call("banana", ["1S"])
    assert not is_legal_call("8C", [])         # level 8 doesn't exist


# ── Normalisation ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("pass", "Pass"),
    ("P", "Pass"),
    ("x", "X"),
    ("double", "X"),
    ("xx", "XX"),
    ("redouble", "XX"),
    ("3nt", "3NT"),
    ("4h", "4H"),
    ("2n", "2NT"),
    ("1S", "1S"),
])
def test_normalise_call(raw, expected):
    assert _normalise_call(raw) == expected


# ── make_bid with mocked LLM ──────────────────────────────────────────────────

def test_make_bid_returns_legal_bid(slam_hunter_sig):
    agent = _agent_with_mocked_llm(slam_hunter_sig, "1NT")
    out = agent.make_bid({"S": "AKQ", "H": "KJ4", "D": "A83", "C": "Q932"}, [])
    assert out["bid"] == "1NT"
    assert out["legal"] is True


def test_make_bid_falls_back_to_pass_on_illegal(slam_hunter_sig):
    # LLM tries to bid 1S over an existing 3NT — illegal, must fall back.
    agent = _agent_with_mocked_llm(slam_hunter_sig, "1S")
    out = agent.make_bid({"S": "AKQ"}, ["3NT"])
    assert out["bid"] == "Pass"
    assert out["legal"] is False
    assert out["raw_bid"] == "1S"


def test_make_bid_normalises_lowercase(slam_hunter_sig):
    agent = _agent_with_mocked_llm(slam_hunter_sig, "3nt")
    out = agent.make_bid({"S": "AKQ"}, ["1S"])
    assert out["bid"] == "3NT"
    assert out["legal"] is True


def test_make_bid_passes_system_prompt(slam_hunter_sig):
    agent = _agent_with_mocked_llm(slam_hunter_sig, "Pass")
    agent.make_bid({"S": "AKQ"}, [])
    # The shared client.generate must have been called with the character card.
    _, kwargs = agent.client.generate.call_args
    assert "Slam Hunter" in kwargs["system"]
    assert kwargs["purpose"] == "bridge_bid"


def test_act_delegates_to_make_bid(slam_hunter_sig):
    agent = _agent_with_mocked_llm(slam_hunter_sig, "Pass")
    out = agent.act({"S": "AKQ"}, [])
    assert out["bid"] == "Pass"
