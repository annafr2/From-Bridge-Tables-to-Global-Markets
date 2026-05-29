"""
tests/test_chunker.py
======================
Tests for src/stage2_skills/chunker.py — no LLM calls.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.stage2_skills.chunker import (
    BoardSummary,
    PlayerChunk,
    PLAYER_COLUMNS,
    build_player_chunks,
    find_player_boards,
)


@pytest.fixture
def minimal_df():
    """A tiny DataFrame with 4 boards across 2 rooms — same player at N seat."""
    rows = []
    for i, (room, board) in enumerate([
        ("Open", 1), ("Closed", 1), ("Open", 2), ("Closed", 2),
    ]):
        rows.append({
            "board": board,
            "room": room,
            "contract": "3NT",
            "declarer": "N",
            "tricks": 9,
            "ns_score": 400,
            "bidding": "W:- N:1NT E:Pass S:3NT | W:Pass N:Pass E:Pass",
            "dealer": "North",
            "vulnerability": "None",
            "has_bidding": True,
            "has_cards": True,
            # 8 player slots
            "open_north":   "TARGET PLAYER" if room == "Open" and i == 0 else "OTHER",
            "open_south":   "P_OS",
            "open_east":    "P_OE",
            "open_west":    "P_OW",
            "closed_north": "TARGET PLAYER" if room == "Closed" and i == 1 else "OTHER",
            "closed_south": "P_CS",
            "closed_east":  "P_CE",
            "closed_west":  "P_CW",
            # Hands
            "north_spades":  "AKQ", "north_hearts": "AKQ", "north_diamonds": "AKQ", "north_clubs": "AKQJ",
            "south_spades":  "JT9", "south_hearts": "JT9", "south_diamonds": "JT9", "south_clubs": "T98",
            "east_spades":   "876", "east_hearts":  "876", "east_diamonds":  "876", "east_clubs":  "765",
            "west_spades":   "543", "west_hearts":  "543", "west_diamonds":  "5432","west_clubs":  "432",
        })
    return pd.DataFrame(rows)


def test_find_player_boards_returns_matching_rows(minimal_df):
    result = find_player_boards(minimal_df, "TARGET PLAYER")
    assert len(result) == 2  # one Open + one Closed match
    assert set(result["player_seat"].unique()) == {"N"}


def test_find_player_boards_empty_when_unknown(minimal_df):
    result = find_player_boards(minimal_df, "WHO IS THIS")
    assert len(result) == 0


def test_find_player_boards_filters_no_bidding(minimal_df):
    minimal_df.loc[0, "has_bidding"] = False
    result = find_player_boards(minimal_df, "TARGET PLAYER", require_bidding=True)
    assert len(result) == 1


def test_build_player_chunks_respects_chunk_size():
    """Synthetic: 60 boards, chunk_size=25 should give 3 chunks (25+25+10)."""
    rows = []
    for i in range(60):
        rows.append({
            "board": i + 1, "room": "Open", "contract": "3NT", "declarer": "N",
            "tricks": 9, "ns_score": 400,
            "bidding": "W:- N:1NT E:Pass S:3NT | W:Pass N:Pass E:Pass",
            "dealer": "North", "vulnerability": "None",
            "has_bidding": True, "has_cards": True,
            "open_north": "TARGET", "open_south": "X", "open_east": "X", "open_west": "X",
            "closed_north": "X", "closed_south": "X", "closed_east": "X", "closed_west": "X",
            "north_spades": "AKQ", "north_hearts": "AKQ", "north_diamonds": "AKQ", "north_clubs": "AKQJ",
            "south_spades": "JT9", "south_hearts": "JT9", "south_diamonds": "JT9", "south_clubs": "T98",
            "east_spades": "876", "east_hearts": "876", "east_diamonds": "876", "east_clubs": "765",
            "west_spades": "543", "west_hearts": "543", "west_diamonds": "5432", "west_clubs": "432",
        })
    df = pd.DataFrame(rows)
    chunks = build_player_chunks(df, "TARGET", chunk_size=25)
    assert len(chunks) == 3
    assert [len(c.boards) for c in chunks] == [25, 25, 10]


def test_build_player_chunks_max_boards_cap():
    """max_boards=10 should cap output to a single 10-board chunk."""
    rows = []
    for i in range(60):
        rows.append({
            "board": i + 1, "room": "Open", "contract": "3NT", "declarer": "N",
            "tricks": 9, "ns_score": 400,
            "bidding": "W:- N:1NT E:Pass S:3NT | W:Pass N:Pass E:Pass",
            "dealer": "North", "vulnerability": "None",
            "has_bidding": True, "has_cards": True,
            "open_north": "TARGET", "open_south": "X", "open_east": "X", "open_west": "X",
            "closed_north": "X", "closed_south": "X", "closed_east": "X", "closed_west": "X",
            "north_spades": "AKQ", "north_hearts": "AKQ", "north_diamonds": "AKQ", "north_clubs": "AKQJ",
            "south_spades": "JT9", "south_hearts": "JT9", "south_diamonds": "JT9", "south_clubs": "T98",
            "east_spades": "876", "east_hearts": "876", "east_diamonds": "876", "east_clubs": "765",
            "west_spades": "543", "west_hearts": "543", "west_diamonds": "5432", "west_clubs": "432",
        })
    df = pd.DataFrame(rows)
    chunks = build_player_chunks(df, "TARGET", chunk_size=25, max_boards=10)
    assert len(chunks) == 1
    assert len(chunks[0].boards) == 10


def test_board_summary_to_text_contains_key_fields():
    b = BoardSummary(
        board_id=42,
        player_seat="N",
        dealer="North",
        vulnerability="NS",
        hand="♠AKQ ♥AKQ ♦AKQ ♣AKQJ",
        bidding="W:- N:1NT E:Pass S:3NT",
        contract="3NT",
        declarer="N",
        tricks=9,
        ns_score=400,
        player_role="Declarer",
    )
    text = b.to_text()
    assert "Board 42" in text
    assert "Declarer" in text
    assert "3NT" in text
    assert "made 9 tricks" in text
    assert "NS +400" in text


def test_player_chunk_to_prompt_includes_header():
    b = BoardSummary(
        board_id=1, player_seat="N", dealer="N", vulnerability="None",
        hand="♠AKQ ♥AKQ ♦AKQ ♣AKQJ",
        bidding="W:Pass", contract="3NT", declarer="N",
        tricks=9, ns_score=400, player_role="Declarer",
    )
    chunk = PlayerChunk(player_name="Alice", profile="Slam Hunter", boards=[b])
    prompt = chunk.to_prompt()
    assert "Player: Alice" in prompt
    assert "Profile: Slam Hunter" in prompt
    assert "Total boards in this chunk: 1" in prompt
    assert "Board 1" in prompt
    # Verify the separator bug fix: the "Player:" header appears exactly once.
    assert prompt.count("Player: Alice") == 1
