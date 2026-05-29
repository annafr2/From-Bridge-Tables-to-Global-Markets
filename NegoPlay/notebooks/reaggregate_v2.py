"""
notebooks/reaggregate_v2.py
============================
Re-aggregate Stage 2 v2 results using the improved semantic-clustering
aggregator — WITHOUT making any new LLM calls.

Reads:   results/stage2_sample_v2_focused_prompt/skill_profiles.json
Writes:  results/stage2_sample_v2_focused_prompt/skill_profiles_reagg.json
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

from src.stage2_skills.aggregator import (
    PlayerSkillProfile,
    SkillEntry,
    aggregate_profile,
    save_skill_profiles,
)

IN_PATH  = Path("results/stage2_sample_v2_focused_prompt/skill_profiles.json")
OUT_PATH = Path("results/stage2_sample_v2_focused_prompt/skill_profiles_reagg_t040.json")

NON_GENERALIST_PROFILES = ["Slam Hunter", "Insurance Player", "Fighter", "NT Specialist"]

# ── Load existing player profiles from JSON (no LLM calls needed) ─────────────

with IN_PATH.open(encoding="utf-8") as f:
    raw = json.load(f)

player_profiles: list[PlayerSkillProfile] = []
for p in raw["player_profiles"]:
    skills = [
        SkillEntry(
            name=s["name"],
            description=s["description"],
            n_mentions=s["n_mentions"],
            confidence_avg=s["confidence_avg"],
            evidence_boards=s.get("evidence_boards", []),
            aliases=s.get("aliases", []),
        )
        for s in p["skills"]
    ]
    player_profiles.append(PlayerSkillProfile(
        player_name=p["player_name"],
        profile=p["profile"],
        n_chunks=p["n_chunks"],
        n_boards_total=p["n_boards_total"],
        skills=skills,
        summary=p.get("summary", ""),
        total_cost_usd=p.get("total_cost_usd", 0.0),
        errors=p.get("errors", []),
    ))

print(f"Loaded {len(player_profiles)} player profiles from {IN_PATH}")

# ── Re-aggregate with semantic clustering ─────────────────────────────────────

profile_signatures = []
for prof in NON_GENERALIST_PROFILES:
    players_in_prof = [p for p in player_profiles if p.profile == prof]
    sig = aggregate_profile(players_in_prof, profile_name=prof)
    profile_signatures.append(sig)

# ── Print results ─────────────────────────────────────────────────────────────

print()
print("=" * 60)
print("PROFILE SIGNATURES (re-aggregated with semantic clustering)")
print("=" * 60)
for sig in profile_signatures:
    print()
    print(f"### {sig.profile}  ({sig.n_players} players)")
    if not sig.skills:
        print("  (no recurring skills found — try lowering threshold)")
        continue
    for i, s in enumerate(sig.skills, 1):
        print(f"  {i}. {s.name}  (mentions: {s.n_mentions}/{sig.n_players} players, conf: {s.confidence_avg:.2f})")
        if s.aliases:
            print(f"     aliases: {s.aliases[:4]}")
        print(f"     {s.description[:130]}")

# ── Save ──────────────────────────────────────────────────────────────────────

save_skill_profiles(
    player_profiles=player_profiles,
    profile_signatures=profile_signatures,
    out_path=OUT_PATH,
)
print()
print(f"Saved: {OUT_PATH}")
