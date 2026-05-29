"""
src/stage2_skills/aggregator.py
=================================
Aggregate skill extractions into player-level and profile-level summaries.

After Stage 2's extractor runs on every chunk, we end up with many
ChunkExtraction objects:
  - Per player: 1–10 chunks (depending on how many boards they played)
  - Per profile: 1–20 players × chunks

This module rolls those chunk-level extractions up into:
  1. **Player-level skills** — skills that recur across that player's chunks
  2. **Profile-level skills** — skills shared by most players in the same profile

The aggregation strategy is deliberately simple:
  - Group skills by name (case-insensitive)
  - Count how many extractions mention each skill
  - Rank by frequency
  - For semantically-similar skills, do a final LLM-based deduplication step
    (optional — controlled by `dedupe_with_llm=True`)

Output structures:
  - PlayerSkillProfile     — for one player
  - ProfileSkillSignature  — for one profile (e.g. "Slam Hunter")
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.shared.llm_client import LLMClient
from src.stage2_skills.extractor import ChunkExtraction

logger = logging.getLogger(__name__)


# ── Output containers ─────────────────────────────────────────────────────────

@dataclass
class SkillEntry:
    """One aggregated skill with its evidence."""

    name: str
    description: str        # canonical description (longest or most common)
    n_mentions: int         # how many extractions mentioned it
    confidence_avg: float   # avg confidence (high=1, medium=0.5, low=0.25)
    evidence_boards: list[int] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)  # alternative names found


@dataclass
class PlayerSkillProfile:
    """All skills extracted for one player across all their chunks."""

    player_name: str
    profile: str | None
    n_chunks: int
    n_boards_total: int
    skills: list[SkillEntry] = field(default_factory=list)
    summary: str = ""              # last (longest) summary across chunks
    total_cost_usd: float = 0.0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "player_name": self.player_name,
            "profile": self.profile,
            "n_chunks": self.n_chunks,
            "n_boards_total": self.n_boards_total,
            "summary": self.summary,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "skills": [
                {
                    "name": s.name,
                    "description": s.description,
                    "n_mentions": s.n_mentions,
                    "confidence_avg": round(s.confidence_avg, 3),
                    "evidence_boards": s.evidence_boards,
                    "aliases": s.aliases,
                }
                for s in self.skills
            ],
            "errors": self.errors,
        }


@dataclass
class ProfileSkillSignature:
    """Skills that characterise a whole profile (across all its players)."""

    profile: str
    n_players: int
    skills: list[SkillEntry] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "n_players": self.n_players,
            "skills": [
                {
                    "name": s.name,
                    "description": s.description,
                    "n_mentions": s.n_mentions,
                    "confidence_avg": round(s.confidence_avg, 3),
                    "aliases": s.aliases,
                }
                for s in self.skills
            ],
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

_CONF_VALUES = {"high": 1.0, "medium": 0.5, "low": 0.25}


def _normalize_name(name: str) -> str:
    """Lower-case + strip punctuation for grouping similar skill names."""
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# ── Player-level aggregation ──────────────────────────────────────────────────

def aggregate_player(extractions: list[ChunkExtraction]) -> PlayerSkillProfile:
    """Merge multiple chunk extractions for the same player into one profile.

    Args:
        extractions: List of ChunkExtractions, all for the SAME player.
                     Order does not matter.

    Returns:
        PlayerSkillProfile with skills ranked by mention frequency.
    """
    if not extractions:
        raise ValueError("Cannot aggregate empty list of extractions")

    player_name = extractions[0].player_name
    profile = extractions[0].profile
    if not all(e.player_name == player_name for e in extractions):
        raise ValueError(
            "All extractions in aggregate_player() must be for the same player"
        )

    # Bucket skills by normalised name
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    raw_names: dict[str, list[str]] = defaultdict(list)
    summaries: list[str] = []
    errors: list[str] = []
    total_boards = 0
    total_cost = 0.0

    for ext in extractions:
        total_boards += ext.n_boards
        total_cost += ext.cost_usd
        if ext.error:
            errors.append(ext.error)
            continue
        if ext.summary:
            summaries.append(ext.summary)
        for skill in ext.skills:
            name = skill.get("name", "")
            if not name:
                continue
            key = _normalize_name(name)
            buckets[key].append(skill)
            raw_names[key].append(name)

    # Build SkillEntry per bucket
    entries: list[SkillEntry] = []
    for key, skill_list in buckets.items():
        # Most common original name = canonical
        name_counts = Counter(raw_names[key])
        canonical_name = name_counts.most_common(1)[0][0]
        aliases = sorted({n for n in raw_names[key] if n != canonical_name})

        # Longest description = canonical
        descriptions = [s.get("description", "") for s in skill_list]
        canonical_desc = max(descriptions, key=len) if descriptions else ""

        # Confidence avg
        confs = [_CONF_VALUES.get(s.get("confidence", "low"), 0.25) for s in skill_list]
        conf_avg = sum(confs) / len(confs) if confs else 0.0

        # Evidence: union of all board lists
        evidence: list[int] = []
        for s in skill_list:
            ev = s.get("evidence_boards", [])
            if isinstance(ev, list):
                evidence.extend(int(b) for b in ev if isinstance(b, (int, float)))
        evidence = sorted(set(evidence))

        entries.append(SkillEntry(
            name=canonical_name,
            description=canonical_desc,
            n_mentions=len(skill_list),
            confidence_avg=conf_avg,
            evidence_boards=evidence,
            aliases=aliases,
        ))

    # Rank: most mentions first, ties broken by confidence
    entries.sort(key=lambda e: (-e.n_mentions, -e.confidence_avg, e.name))

    # Use the longest summary as the canonical
    canonical_summary = max(summaries, key=len) if summaries else ""

    return PlayerSkillProfile(
        player_name=player_name,
        profile=profile,
        n_chunks=len(extractions),
        n_boards_total=total_boards,
        skills=entries,
        summary=canonical_summary,
        total_cost_usd=total_cost,
        errors=errors,
    )


# ── Profile-level aggregation ─────────────────────────────────────────────────

def aggregate_profile(
    player_profiles: list[PlayerSkillProfile],
    profile_name: str,
    min_player_share: float = 0.3,
    top_n_skills: int = 7,
) -> ProfileSkillSignature:
    """Merge per-player skill profiles into a profile-wide signature.

    Args:
        player_profiles: List of PlayerSkillProfiles, all from the SAME profile.
        profile_name:    Profile label (e.g. "Slam Hunter").
        min_player_share: A skill must appear for at least this fraction of
                          players to be included (default 0.3 = 30%).
        top_n_skills:    Cap output to N strongest skills (default 7).

    Returns:
        ProfileSkillSignature with the top recurring skills.
    """
    if not player_profiles:
        return ProfileSkillSignature(profile=profile_name, n_players=0)

    n_players = len(player_profiles)
    min_mentions = max(1, int(round(n_players * min_player_share)))

    # Bucket skills across players
    buckets: dict[str, list[SkillEntry]] = defaultdict(list)
    raw_names: dict[str, list[str]] = defaultdict(list)

    for player in player_profiles:
        seen_keys_for_this_player: set[str] = set()
        for skill in player.skills:
            key = _normalize_name(skill.name)
            if key in seen_keys_for_this_player:
                continue  # don't double-count one player
            seen_keys_for_this_player.add(key)
            buckets[key].append(skill)
            raw_names[key].append(skill.name)

    # Build profile-level entries
    entries: list[SkillEntry] = []
    for key, skill_list in buckets.items():
        n_mentions = len(skill_list)
        if n_mentions < min_mentions:
            continue

        name_counts = Counter(raw_names[key])
        canonical_name = name_counts.most_common(1)[0][0]
        aliases = sorted({n for n in raw_names[key] if n != canonical_name})

        descriptions = [s.description for s in skill_list]
        canonical_desc = max(descriptions, key=len) if descriptions else ""

        conf_avg = sum(s.confidence_avg for s in skill_list) / len(skill_list)

        entries.append(SkillEntry(
            name=canonical_name,
            description=canonical_desc,
            n_mentions=n_mentions,
            confidence_avg=conf_avg,
            aliases=aliases,
        ))

    entries.sort(key=lambda e: (-e.n_mentions, -e.confidence_avg, e.name))
    entries = entries[:top_n_skills]

    return ProfileSkillSignature(
        profile=profile_name,
        n_players=n_players,
        skills=entries,
    )


# ── Persistence ───────────────────────────────────────────────────────────────

def save_skill_profiles(
    player_profiles: list[PlayerSkillProfile],
    profile_signatures: list[ProfileSkillSignature],
    out_path: str | Path,
) -> None:
    """Write the full Stage 2 output as JSON."""
    out = {
        "n_players": len(player_profiles),
        "n_profiles": len(profile_signatures),
        "total_cost_usd": round(
            sum(p.total_cost_usd for p in player_profiles), 6
        ),
        "profile_signatures": [s.to_dict() for s in profile_signatures],
        "player_profiles": [p.to_dict() for p in player_profiles],
    }
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    logger.info("Saved skill profiles to %s", path)


__all__ = [
    "SkillEntry",
    "PlayerSkillProfile",
    "ProfileSkillSignature",
    "aggregate_player",
    "aggregate_profile",
    "save_skill_profiles",
]
