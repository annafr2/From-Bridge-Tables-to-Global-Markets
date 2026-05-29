"""
tests/test_aggregator.py
=========================
Tests for src/stage2_skills/aggregator.py — no LLM calls.
"""

from __future__ import annotations

import pytest

from src.stage2_skills.aggregator import (
    aggregate_player,
    aggregate_profile,
    _normalize_name,
)
from src.stage2_skills.extractor import ChunkExtraction


def _make_extraction(
    player: str = "Alice",
    profile: str | None = "Slam Hunter",
    n_boards: int = 25,
    skills: list[dict] | None = None,
    cost: float = 0.001,
    summary: str = "A bold player.",
    error: str | None = None,
) -> ChunkExtraction:
    return ChunkExtraction(
        player_name=player,
        profile=profile,
        n_boards=n_boards,
        skills=skills or [],
        summary=summary,
        cost_usd=cost,
        latency_sec=2.0,
        model="gemini-test",
        raw_text="",
        error=error,
    )


def test_normalize_name_lowercases_and_strips_punctuation():
    assert _normalize_name("Aggressive Bidding!") == "aggressive bidding"
    assert _normalize_name("Risk-Taking") == "risk taking"
    assert _normalize_name("  Slam Hunter  ") == "slam hunter"


def test_aggregate_player_groups_similar_skills():
    """Two chunks mentioning the same skill should merge into 1 entry."""
    ext1 = _make_extraction(skills=[
        {"name": "Aggressive Slam Bidding", "description": "Bids slam often.",
         "evidence_boards": [1, 5], "confidence": "high"},
        {"name": "Active Defence", "description": "Defends actively.",
         "evidence_boards": [2], "confidence": "medium"},
    ])
    ext2 = _make_extraction(skills=[
        {"name": "Aggressive Slam Bidding", "description": "Loves slams.",
         "evidence_boards": [10, 15], "confidence": "high"},
    ])
    out = aggregate_player([ext1, ext2])

    assert out.player_name == "Alice"
    assert out.profile == "Slam Hunter"
    assert out.n_chunks == 2
    assert out.n_boards_total == 50

    names = [s.name for s in out.skills]
    assert "Aggressive Slam Bidding" in names
    assert "Active Defence" in names

    slam_skill = [s for s in out.skills if s.name == "Aggressive Slam Bidding"][0]
    assert slam_skill.n_mentions == 2
    assert slam_skill.confidence_avg == pytest.approx(1.0)
    assert set(slam_skill.evidence_boards) == {1, 5, 10, 15}


def test_aggregate_player_sorts_by_mention_count():
    ext1 = _make_extraction(skills=[
        {"name": "X", "description": "x", "evidence_boards": [1], "confidence": "low"},
    ])
    ext2 = _make_extraction(skills=[
        {"name": "Y", "description": "y", "evidence_boards": [2], "confidence": "high"},
        {"name": "Y", "description": "y again", "evidence_boards": [3], "confidence": "high"},
    ])
    out = aggregate_player([ext1, ext2])
    # Y appears 2 times -> first
    assert out.skills[0].name == "Y"
    assert out.skills[1].name == "X"


def test_aggregate_player_handles_errors():
    err = _make_extraction(error="rate limit hit", skills=[])
    ok = _make_extraction(skills=[
        {"name": "Bold Bidding", "description": "Bold.", "evidence_boards": [1], "confidence": "high"},
    ])
    out = aggregate_player([err, ok])
    assert "rate limit hit" in out.errors
    assert len(out.skills) == 1


def test_aggregate_player_rejects_mixed_players():
    e1 = _make_extraction(player="Alice")
    e2 = _make_extraction(player="Bob")
    with pytest.raises(ValueError):
        aggregate_player([e1, e2])


def test_aggregate_profile_filters_rare_skills():
    """min_player_share=0.5 should drop skills mentioned by <50% of players."""
    from src.stage2_skills.aggregator import (
        PlayerSkillProfile,
        SkillEntry,
    )

    def mk_player(name: str, skill_names: list[str]) -> PlayerSkillProfile:
        return PlayerSkillProfile(
            player_name=name,
            profile="Slam Hunter",
            n_chunks=1,
            n_boards_total=25,
            skills=[
                SkillEntry(name=s, description=f"{s} desc", n_mentions=1,
                           confidence_avg=1.0)
                for s in skill_names
            ],
        )

    players = [
        mk_player("A", ["Slam Bidding", "Active Defence"]),
        mk_player("B", ["Slam Bidding", "Constructive Game Bidding"]),
        mk_player("C", ["Slam Bidding"]),
        mk_player("D", ["Slam Bidding", "Active Defence"]),
    ]

    sig = aggregate_profile(players, "Slam Hunter", min_player_share=0.5)
    names = [s.name for s in sig.skills]
    # Slam Bidding 4/4 — in
    # Active Defence 2/4 = 50% — in (>= threshold)
    # Constructive 1/4 — out
    assert "Slam Bidding" in names
    assert "Active Defence" in names
    assert "Constructive Game Bidding" not in names


def test_aggregate_player_empty_extractions_raises():
    with pytest.raises(ValueError):
        aggregate_player([])
