"""
tests/test_prompts.py
=====================
Unit tests for the Stage 3 prompt library (src/shared/prompts.py).

These tests are pure — no LLM, no API key, no network. They verify that the
character-card builders assemble correct prompts from Stage 2 signatures.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.shared.prompts import (
    PROFILE_NAMES,
    ProfileSignature,
    build_bridge_system_prompt,
    build_negotiation_system_prompt,
    load_profile_signatures,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_signature() -> ProfileSignature:
    return ProfileSignature(
        profile="Slam Hunter",
        skills=[
            {"name": "Aggressive Splinter Bidding",
             "description": "Makes splinter bids to invite slam."},
            {"name": "Control Cue-Bidding",
             "description": "Uses cue-bids to explore slam."},
        ],
        n_players=5,
    )


@pytest.fixture
def signatures_json(tmp_path: Path) -> Path:
    """Write a minimal Stage-2-shaped JSON to a temp file."""
    data = {
        "profile_signatures": [
            {"profile": "Slam Hunter", "n_players": 5,
             "skills": [{"name": "X", "description": "desc"}]},
            {"profile": "Fighter", "n_players": 5,
             "skills": [{"name": "Y", "description": "desc"}]},
        ]
    }
    p = tmp_path / "sigs.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


# ── load_profile_signatures ───────────────────────────────────────────────────

def test_load_returns_all_five_profiles(signatures_json: Path):
    """Even when JSON has only 2 profiles, all 5 PROFILE_NAMES are present."""
    sigs = load_profile_signatures(signatures_json)
    assert set(sigs.keys()) == set(PROFILE_NAMES)


def test_load_synthesises_empty_generalist(signatures_json: Path):
    """Generalist is absent from JSON but must appear as an empty baseline."""
    sigs = load_profile_signatures(signatures_json)
    assert sigs["Generalist"].skills == []
    assert sigs["Generalist"].n_players == 0


def test_load_preserves_real_skills(signatures_json: Path):
    sigs = load_profile_signatures(signatures_json)
    assert len(sigs["Slam Hunter"].skills) == 1
    assert sigs["Slam Hunter"].skills[0]["name"] == "X"


def test_load_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_profile_signatures("does/not/exist.json")


# ── skill_lines rendering ─────────────────────────────────────────────────────

def test_skill_lines_includes_name_and_description(sample_signature):
    text = sample_signature.skill_lines()
    assert "Aggressive Splinter Bidding" in text
    assert "Makes splinter bids to invite slam." in text


def test_skill_lines_empty_profile_has_fallback():
    sig = ProfileSignature(profile="Generalist", skills=[])
    text = sig.skill_lines()
    assert "standard bridge" in text.lower()


def test_skill_lines_caps_at_max(sample_signature):
    many = ProfileSignature(
        profile="Fighter",
        skills=[{"name": f"S{i}", "description": "d"} for i in range(20)],
    )
    rendered = many.skill_lines(max_skills=3)
    assert rendered.count("\n") == 2  # 3 lines => 2 newlines


# ── bridge prompt ─────────────────────────────────────────────────────────────

def test_bridge_prompt_contains_identity(sample_signature):
    prompt = build_bridge_system_prompt(sample_signature)
    assert "Slam Hunter" in prompt


def test_bridge_prompt_contains_skills(sample_signature):
    prompt = build_bridge_system_prompt(sample_signature)
    assert "Aggressive Splinter Bidding" in prompt


def test_bridge_prompt_contains_rules_and_schema(sample_signature):
    prompt = build_bridge_system_prompt(sample_signature)
    assert "BRIDGE BIDDING RULES" in prompt
    assert '"bid"' in prompt
    assert '"reasoning"' in prompt


# ── negotiation prompt ────────────────────────────────────────────────────────

def test_negotiation_prompt_contains_identity_and_schema(sample_signature):
    prompt = build_negotiation_system_prompt(sample_signature)
    assert "Slam Hunter" in prompt
    assert '"action"' in prompt
    assert '"willing_to_close"' in prompt


def test_negotiation_prompt_links_to_bridge_style(sample_signature):
    """The anti-tautology rule: negotiation must flow from bridge style."""
    prompt = build_negotiation_system_prompt(sample_signature)
    assert "FLOW FROM your bridge style" in prompt
