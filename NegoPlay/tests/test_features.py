"""
tests/test_features.py
Tests for Stage 1 feature engineering.
Run: pytest tests/test_features.py -v
"""

import pandas as pd
import pytest

from src.stage1_clustering.features import (
    MIN_BOARDS,
    _get_declarer_name,
    _is_doubled,
    _is_game,
    _is_made,
    _is_nt,
    _is_partscore,
    _parse_contract_level,
    compute_player_features,
)


# ── Unit tests for helper functions ─────────────────────────────────────────

class TestParseContractLevel:
    def test_normal_contract(self):
        assert _parse_contract_level("4H") == 4

    def test_slam_contract(self):
        assert _parse_contract_level("6NT") == 6

    def test_grand_slam(self):
        assert _parse_contract_level("7C") == 7

    def test_doubled_contract(self):
        assert _parse_contract_level("3NT*") == 3

    def test_redoubled(self):
        assert _parse_contract_level("4S**") == 4

    def test_passed_out(self):
        assert _parse_contract_level("-") is None

    def test_empty_string(self):
        assert _parse_contract_level("") is None

    def test_none_input(self):
        assert _parse_contract_level(None) is None


class TestIsMade:
    def test_made_4h(self):
        assert _is_made("4H", 10) is True   # needs 10, got 10

    def test_down_4h(self):
        assert _is_made("4H", 9) is False    # needs 10, got 9

    def test_slam_made(self):
        assert _is_made("6NT", 12) is True

    def test_slam_down(self):
        assert _is_made("6NT", 11) is False

    def test_passed_out(self):
        assert _is_made("-", 0) is None


class TestIsDoubled:
    def test_doubled(self):
        assert _is_doubled("4H*") is True

    def test_redoubled(self):
        assert _is_doubled("4H**") is True

    def test_not_doubled(self):
        assert _is_doubled("4H") is False

    def test_none(self):
        assert _is_doubled(None) is False


class TestIsNT:
    def test_nt_contract(self):
        assert _is_nt("3NT") is True

    def test_nt_slam(self):
        assert _is_nt("6NT*") is True

    def test_suit_contract(self):
        assert _is_nt("4H") is False

    def test_none(self):
        assert _is_nt(None) is False


class TestIsPartscore:
    def test_level_1(self):
        assert _is_partscore("1C") is True

    def test_level_2(self):
        assert _is_partscore("2H") is True

    def test_level_3_suit(self):
        assert _is_partscore("3H") is True     # below game

    def test_level_3_nt(self):
        assert _is_partscore("3NT") is True    # level=3, partscore by level def

    def test_level_4(self):
        assert _is_partscore("4S") is False

    def test_slam(self):
        assert _is_partscore("6H") is False


class TestIsGame:
    def test_3nt_is_game(self):
        # level=3 + NT → game (3NT is the classic game contract)
        assert _is_game("3NT") is True

    def test_3h_is_partscore(self):
        assert _is_game("3H") is False

    def test_4h_is_game(self):
        assert _is_game("4H") is True

    def test_4s_is_game(self):
        assert _is_game("4S") is True

    def test_5c_is_game(self):
        assert _is_game("5C") is True

    def test_6h_is_slam_not_game(self):
        assert _is_game("6H") is False

    def test_1c_is_not_game(self):
        assert _is_game("1C") is False


class TestGetDeclarerName:
    def _row(self, declarer, room, **kwargs):
        base = {
            "declarer": declarer, "room": room,
            "open_north": "Alice", "open_south": "Bob",
            "open_east": "Carol", "open_west": "Dave",
            "closed_north": "Eve", "closed_south": "Frank",
            "closed_east": "Grace", "closed_west": "Hank",
        }
        base.update(kwargs)
        return pd.Series(base)

    def test_open_north(self):
        assert _get_declarer_name(self._row("N", "Open")) == "Alice"

    def test_closed_south(self):
        assert _get_declarer_name(self._row("S", "Closed")) == "Frank"

    def test_passed_out(self):
        assert _get_declarer_name(self._row("-", "Open")) is None

    def test_unknown_position(self):
        assert _get_declarer_name(self._row("X", "Open")) is None


# ── Integration test: compute_player_features ───────────────────────────────

@pytest.fixture
def minimal_df():
    """Minimal synthetic DataFrame mimicking the real dataset."""
    rows = []
    # Player "SMITH John" declares 25 boards, mix of slams and partials
    for i in range(25):
        contract = "6H" if i < 5 else "4H"   # 5 slams, 20 partials
        tricks = 12 if i < 5 else (10 if i < 20 else 9)  # some down
        rows.append({
            "match_id": 1, "round": 1, "board": i + 1,
            "room": "Open", "contract": contract,
            "declarer": "N", "tricks": tricks,
            "ns_score": 100, "ew_score": 0,
            "has_bidding": False, "has_cards": False,
            "open_north": "SMITH John",
            "open_south": "JONES Mary",
            "open_east": "BROWN Peter",
            "open_west": "WHITE Anna",
            "closed_north": "A", "closed_south": "B",
            "closed_east": "C", "closed_west": "D",
        })
    # Player "JONES Mary" only 5 boards (below MIN_BOARDS — should be filtered)
    for i in range(5):
        rows.append({
            "match_id": 1, "round": 1, "board": i + 26,
            "room": "Open", "contract": "3NT",
            "declarer": "S", "tricks": 9,
            "ns_score": 400, "ew_score": 0,
            "has_bidding": False, "has_cards": False,
            "open_north": "SMITH John",
            "open_south": "JONES Mary",
            "open_east": "BROWN Peter",
            "open_west": "WHITE Anna",
            "closed_north": "A", "closed_south": "B",
            "closed_east": "C", "closed_west": "D",
        })
    return pd.DataFrame(rows)


def test_compute_player_features_returns_dataframe(minimal_df):
    result = compute_player_features(minimal_df, min_boards=MIN_BOARDS)
    assert isinstance(result, pd.DataFrame)


def test_min_boards_filter_applied(minimal_df):
    result = compute_player_features(minimal_df, min_boards=MIN_BOARDS)
    # JONES Mary has only 5 boards — must be filtered out
    assert "JONES Mary" not in result["player_name"].values
    # SMITH John has 25 boards — must be included
    assert "SMITH John" in result["player_name"].values


def test_slam_rate_correct(minimal_df):
    result = compute_player_features(minimal_df, min_boards=MIN_BOARDS)
    smith = result[result["player_name"] == "SMITH John"].iloc[0]
    # 5 slams out of 25 boards = 0.20
    assert abs(smith["slam_rate"] - 0.20) < 0.01


def test_success_rate_correct(minimal_df):
    result = compute_player_features(minimal_df, min_boards=MIN_BOARDS)
    smith = result[result["player_name"] == "SMITH John"].iloc[0]
    # 5 slams made (tricks=12, need 12) + 15 partials made (tricks=10, need 10) = 20/25 = 0.80
    assert abs(smith["success_rate"] - 0.80) < 0.01


def test_risk_score_in_range(minimal_df):
    result = compute_player_features(minimal_df, min_boards=MIN_BOARDS)
    assert (result["risk_score"] >= 0).all()
    assert (result["risk_score"] <= 10).all()


def test_output_columns(minimal_df):
    result = compute_player_features(minimal_df, min_boards=MIN_BOARDS)
    expected = {"player_name", "n_declared", "slam_rate",
                "success_rate", "double_rate", "avg_level", "risk_score"}
    assert expected.issubset(set(result.columns))
