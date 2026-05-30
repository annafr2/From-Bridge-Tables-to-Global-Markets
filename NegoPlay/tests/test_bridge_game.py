"""
tests/test_bridge_game.py
=========================
Tests for the deterministic Stage 4a bridge infrastructure.

Pure tests — no LLM, no network. They verify that deals are valid and
reproducible and that the objective scorer behaves sensibly.
"""

from __future__ import annotations

from src.stage4_simulate.bridge_game import (
    GAME,
    GRAND_SLAM,
    PARTSCORE,
    SMALL_SLAM,
    bid_level_class,
    deal_board,
    hand_hcp,
    par_level_class,
    score_bid,
)


# ── Dealing: validity ─────────────────────────────────────────────────────────

def test_deal_has_four_hands():
    d = deal_board(1)
    assert set(d.hands.keys()) == {"N", "E", "S", "W"}


def test_each_hand_has_13_cards():
    d = deal_board(7)
    for pos in ("N", "E", "S", "W"):
        total = sum(len(d.hands[pos][s]) for s in ("S", "H", "D", "C"))
        assert total == 13, f"{pos} has {total} cards"


def test_full_deck_dealt_no_duplicates():
    d = deal_board(3)
    seen = []
    for pos in ("N", "E", "S", "W"):
        for suit in ("S", "H", "D", "C"):
            for rank in d.hands[pos][suit]:
                seen.append(rank + suit)
    assert len(seen) == 52
    assert len(set(seen)) == 52      # no duplicate cards


def test_total_hcp_is_40():
    """Every full deal has exactly 40 HCP (4*A + ... across the deck)."""
    d = deal_board(5)
    total = sum(d.hcp(p) for p in ("N", "E", "S", "W"))
    assert total == 40


# ── Dealing: reproducibility ──────────────────────────────────────────────────

def test_same_board_same_seed_is_identical():
    a = deal_board(10, seed=42)
    b = deal_board(10, seed=42)
    assert a.hands == b.hands


def test_different_boards_differ():
    a = deal_board(1, seed=42)
    b = deal_board(2, seed=42)
    assert a.hands != b.hands


def test_different_seeds_differ():
    a = deal_board(1, seed=42)
    b = deal_board(1, seed=7)
    assert a.hands != b.hands


# ── HCP ───────────────────────────────────────────────────────────────────────

def test_hand_hcp_known_hand():
    hand = {"S": "AKQ", "H": "J", "D": "", "C": ""}  # 4+3+2+1 = 10
    assert hand_hcp(hand) == 10


def test_hand_hcp_empty():
    assert hand_hcp({"S": "", "H": "", "D": "", "C": ""}) == 0


# ── par_level_class ───────────────────────────────────────────────────────────

def test_par_partscore():
    assert par_level_class(20) == PARTSCORE


def test_par_game():
    assert par_level_class(26) == GAME


def test_par_small_slam():
    assert par_level_class(34) == SMALL_SLAM


def test_par_grand_slam():
    assert par_level_class(38) == GRAND_SLAM


# ── bid_level_class ───────────────────────────────────────────────────────────

def test_bid_class_partscore():
    assert bid_level_class("2H") == PARTSCORE


def test_bid_class_game_3nt():
    assert bid_level_class("3NT") == GAME


def test_bid_class_3_of_suit_is_partscore():
    assert bid_level_class("3S") == PARTSCORE


def test_bid_class_game_major():
    assert bid_level_class("4S") == GAME
    assert bid_level_class("4H") == GAME


def test_bid_class_4_minor_is_partscore():
    assert bid_level_class("4C") == PARTSCORE


def test_bid_class_slam():
    assert bid_level_class("6NT") == SMALL_SLAM
    assert bid_level_class("7S") == GRAND_SLAM


def test_bid_class_pass_is_none():
    assert bid_level_class("Pass") is None


# ── score_bid ─────────────────────────────────────────────────────────────────

def test_perfect_score_right_class():
    # 26 HCP -> game; bidding 4S (game) -> distance 0 -> 1.0
    s = score_bid("4S", 26)
    assert s.score == 1.0
    assert s.distance == 0


def test_one_class_off_half_score():
    # 26 HCP -> game; bidding 2H (partscore) -> distance 1 -> 0.5
    s = score_bid("2H", 26)
    assert s.score == 0.5
    assert s.distance == 1


def test_overbid_slam_on_game_hand():
    # 26 HCP -> game; bidding 6NT (small slam) -> distance 1 -> 0.5
    s = score_bid("6NT", 26)
    assert s.distance == 1


def test_big_overbid_low_score():
    # 20 HCP -> partscore; bidding 7S (grand slam) -> distance 3 -> 0.0
    s = score_bid("7S", 20)
    assert s.score == 0.0


def test_pass_on_partscore_is_acceptable():
    s = score_bid("Pass", 20)
    assert s.score == 0.5
    assert s.notes == "passed"


def test_pass_on_game_is_bad():
    s = score_bid("Pass", 28)
    assert s.score == 0.0
