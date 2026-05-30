"""
tests/test_nego_agent.py
========================
Tests for the Stage 3 negotiation agent.

All LLM calls are MOCKED — no API, no cost. We verify prompt wiring, action
validation, the invalid-action fallback, and offer clamping to scenario ranges.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.shared.prompts import ProfileSignature
from src.stage3_agents.nego_agent import NegotiationAgent


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def slam_hunter_sig() -> ProfileSignature:
    return ProfileSignature(
        profile="Slam Hunter",
        skills=[{"name": "Bold Bidding", "description": "Goes big on strong values."}],
        n_players=5,
    )


@pytest.fixture
def scenario() -> dict:
    return {
        "title": "Startup acquisition",
        "role": "buyer",
        "description": "You are acquiring a startup.",
        "terms": {"price_musd": {"min": 5.0, "max": 15.0, "unit": "M USD"}},
        "your_target": {"price_musd": 8.0},
        "your_limit": {"price_musd": 12.0},
    }


def _agent_with_response(sig: ProfileSignature, payload: dict) -> NegotiationAgent:
    """Build a NegotiationAgent whose LLM returns a fixed JSON payload."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.json = payload
    mock_response.text = "{}"
    mock_client.generate.return_value = mock_response
    return NegotiationAgent(sig, client=mock_client)


# ── action handling ───────────────────────────────────────────────────────────

def test_counter_action_returns_offer(slam_hunter_sig, scenario):
    agent = _agent_with_response(slam_hunter_sig, {
        "action": "counter",
        "offer": {"price_musd": 9.0},
        "willing_to_close": False,
        "reasoning": "Anchor low.",
    })
    out = agent.respond_to_offer(scenario, current_offer={"price_musd": 12.0})
    assert out["action"] == "counter"
    assert out["offer"]["price_musd"] == 9.0
    assert out["valid"] is True


def test_accept_action_has_empty_offer(slam_hunter_sig, scenario):
    agent = _agent_with_response(slam_hunter_sig, {
        "action": "accept",
        "offer": {"price_musd": 10.0},
        "willing_to_close": True,
        "reasoning": "Good enough.",
    })
    out = agent.respond_to_offer(scenario, current_offer={"price_musd": 10.0})
    assert out["action"] == "accept"
    assert out["offer"] == {}            # offer cleared on accept
    assert out["willing_to_close"] is True


def test_invalid_action_falls_back_to_walk_away(slam_hunter_sig, scenario):
    agent = _agent_with_response(slam_hunter_sig, {
        "action": "nonsense",
        "willing_to_close": False,
        "reasoning": "???",
    })
    out = agent.respond_to_offer(scenario, current_offer={"price_musd": 10.0})
    assert out["action"] == "walk_away"
    assert out["valid"] is False


# ── offer clamping ────────────────────────────────────────────────────────────

def test_offer_above_max_is_clamped(slam_hunter_sig, scenario):
    agent = _agent_with_response(slam_hunter_sig, {
        "action": "counter",
        "offer": {"price_musd": 99.0},      # above the 15.0 max
        "willing_to_close": False,
        "reasoning": "Overshoot.",
    })
    out = agent.respond_to_offer(scenario, current_offer={"price_musd": 12.0})
    assert out["offer"]["price_musd"] == 15.0


def test_offer_below_min_is_clamped(slam_hunter_sig, scenario):
    agent = _agent_with_response(slam_hunter_sig, {
        "action": "counter",
        "offer": {"price_musd": 1.0},       # below the 5.0 min
        "willing_to_close": False,
        "reasoning": "Lowball.",
    })
    out = agent.respond_to_offer(scenario, current_offer={"price_musd": 12.0})
    assert out["offer"]["price_musd"] == 5.0


def test_offer_within_range_unchanged(slam_hunter_sig, scenario):
    agent = _agent_with_response(slam_hunter_sig, {
        "action": "counter",
        "offer": {"price_musd": 9.5},
        "willing_to_close": False,
        "reasoning": "Fair.",
    })
    out = agent.respond_to_offer(scenario, current_offer={"price_musd": 12.0})
    assert out["offer"]["price_musd"] == 9.5


# ── prompt wiring (anti-tautology) ────────────────────────────────────────────

def test_system_prompt_is_negotiation_card_linked_to_bridge(slam_hunter_sig, scenario):
    agent = _agent_with_response(slam_hunter_sig, {
        "action": "accept", "willing_to_close": True, "reasoning": "ok",
    })
    agent.respond_to_offer(scenario, current_offer={"price_musd": 10.0})
    _, kwargs = agent.client.generate.call_args
    # Identity present...
    assert "Slam Hunter" in kwargs["system"]
    # ...and the anti-tautology link to bridge style is enforced.
    assert "FLOW FROM your bridge style" in kwargs["system"]
    assert kwargs["purpose"] == "negotiation_turn"


def test_opening_move_handles_no_offer(slam_hunter_sig, scenario):
    agent = _agent_with_response(slam_hunter_sig, {
        "action": "counter", "offer": {"price_musd": 8.0},
        "willing_to_close": False, "reasoning": "Open.",
    })
    out = agent.respond_to_offer(scenario, current_offer=None)
    _, kwargs = agent.client.generate.call_args
    assert "OPENING move" in kwargs["user"]
    assert out["action"] == "counter"


def test_act_delegates_to_respond_to_offer(slam_hunter_sig, scenario):
    agent = _agent_with_response(slam_hunter_sig, {
        "action": "accept", "willing_to_close": True, "reasoning": "ok",
    })
    out = agent.act(scenario, current_offer={"price_musd": 10.0})
    assert out["action"] == "accept"
