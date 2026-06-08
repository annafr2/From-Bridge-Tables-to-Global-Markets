"""
tests/test_double_dummy.py
==========================
Tests for the double-dummy bridge evaluation (Task T-C).

These use deterministic seeded deals (deal_board(b, seed=42)) and the embedded
DDS solver, so they run with no API key and no cost. Board 1 (seed 42) is a
known fixture: N-S can take 12 tricks in NoTrump -> small-slam par, score 990.
"""

from __future__ import annotations

from src.features.double_dummy import (
    dd_bid_score01,
    dd_par_level_class,
    hands_to_pbn,
    ns_makeable_tricks,
)
from src.stage4_simulate.bridge_game import deal_board

_BOARD1 = deal_board(1, seed=42).hands


# ── Conversion ────────────────────────────────────────────────────────────────

def test_pbn_has_four_hands_and_anchor():
    pbn = hands_to_pbn(_BOARD1)
    assert pbn.startswith("N:")
    assert len(pbn[2:].split(" ")) == 4          # four hands
    assert all(hand.count(".") == 3 for hand in pbn[2:].split(" "))  # 4 suits each


# ── True par ──────────────────────────────────────────────────────────────────

def test_board1_ns_makes_small_slam_in_nt():
    tricks = ns_makeable_tricks(_BOARD1)
    assert tricks["NT"] == 12          # known fixture
    assert dd_par_level_class(_BOARD1) == "small_slam"


# ── Finer points-based score ──────────────────────────────────────────────────

def test_optimal_slam_scores_one():
    assert dd_bid_score01("6NT", _BOARD1) == 1.0


def test_overbid_grand_slam_scores_zero():
    # 7NT goes down double-dummy (only 12 tricks) -> negative -> clipped to 0.
    assert dd_bid_score01("7NT", _BOARD1) == 0.0


def test_underbid_game_is_between():
    # 3NT makes but leaves the slam bonus on the table -> partial credit.
    s = dd_bid_score01("3NT", _BOARD1)
    assert 0.0 < s < 1.0


def test_score_is_bounded():
    for bid in ["1C", "2H", "4S", "5D", "6NT", "7NT", "Pass", "X", "garbage"]:
        s = dd_bid_score01(bid, _BOARD1)
        assert 0.0 <= s <= 1.0


def test_skill_beats_random_on_dd_metric():
    """The whole point: a sensible slam bid must outscore a wild overbid."""
    assert dd_bid_score01("6NT", _BOARD1) > dd_bid_score01("7NT", _BOARD1)
