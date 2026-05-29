"""
notebooks/run_stage2_sample.py
================================
Run Stage 2 (skill extraction) on a SMALL SAMPLE — 5 players per non-Generalist
profile, max 50 boards per player. Use this to validate the pipeline before
committing to a full run.

Expected cost: ~$0.05 (well below budget).

Output:
    results/stage2_sample/skill_profiles.json
    results/stage2_sample/run_summary.txt
"""

from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

# Load .env
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip()

import pandas as pd

from src.shared.data_loader import load_matches
from src.shared.llm_client import LLMClient
from src.stage2_skills.aggregator import (
    aggregate_player,
    aggregate_profile,
    save_skill_profiles,
)
from src.stage2_skills.chunker import build_player_chunks
from src.stage2_skills.extractor import extract_skills_from_chunk

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("stage2_sample")

# ── Configuration ────────────────────────────────────────────────────────────

DATA = (
    r"C:\Users\annaf\OneDrive\Desktop\דוקטורט\דוקטורט אנה"
    r"\דוקטורט - שילוב בינה מלאכותית\collectBridgeData"
    r"\data\processed\all_matches_full.csv"
)
PROFILES_CSV = "data/processed/player_profiles.csv"
OUT_DIR = Path("results/stage2_sample_v2_focused_prompt")

NON_GENERALIST_PROFILES = [
    "Slam Hunter", "Insurance Player", "Fighter", "NT Specialist",
]
PLAYERS_PER_PROFILE = 5
MAX_BOARDS_PER_PLAYER = 50
CHUNK_SIZE = 25
SEED = 42

# ── Step 1: Load data and select sample ───────────────────────────────────────

print("=" * 60)
print("STAGE 2 SAMPLE RUN — 5 players × 4 profiles = 20 players")
print("=" * 60)
print()

print("Loading data...")
df = load_matches(DATA)
profiles_df = pd.read_csv(PROFILES_CSV, encoding="utf-8-sig")

# Sample 5 players per non-Generalist profile
parts = []
for prof in NON_GENERALIST_PROFILES:
    sub = profiles_df[profiles_df["profile"] == prof]
    if len(sub) == 0:
        continue
    parts.append(sub.sample(min(len(sub), PLAYERS_PER_PROFILE), random_state=SEED))
sample = pd.concat(parts, ignore_index=True)
print(f"Sample size: {len(sample)} players across {sample['profile'].nunique()} profiles")
print(sample.groupby("profile").size().to_string())
print()

# ── Step 2: Run extraction ────────────────────────────────────────────────────

client = LLMClient()
print(f"Provider: {client.provider}  |  Model: {client.model}")
print(f"Budget cap: ${client.logger.budget_cap_usd}")
print(f"Starting cumulative cost: ${client.cumulative_cost():.6f}")
print()

player_profiles_out = []
total_chunks = 0

for i, row in sample.iterrows():
    player = row["player_name"]
    profile = row["profile"]
    print(f"[{i+1:2d}/{len(sample)}] {profile:18s} — {player}")

    chunks = build_player_chunks(
        df=df,
        player_name=player,
        profile=profile,
        chunk_size=CHUNK_SIZE,
        max_boards=MAX_BOARDS_PER_PLAYER,
        seed=SEED,
    )
    if not chunks:
        log.warning("  No chunks — skipped")
        continue

    extractions = []
    for j, chunk in enumerate(chunks):
        ext = extract_skills_from_chunk(client, chunk)
        extractions.append(ext)
        total_chunks += 1
        status = "OK" if not ext.error else f"ERR: {ext.error[:40]}"
        print(
            f"   chunk {j+1}/{len(chunks)}: "
            f"{ext.n_boards} boards, "
            f"{len(ext.skills)} skills, "
            f"${ext.cost_usd:.6f}, "
            f"{ext.latency_sec:.1f}s — {status}"
        )

    player_profile = aggregate_player(extractions)
    player_profiles_out.append(player_profile)

print()
print(f"Total chunks processed: {total_chunks}")
print(f"Total cost: ${client.cumulative_cost():.6f}")

# ── Step 3: Profile-level aggregation ─────────────────────────────────────────

print()
print("Aggregating profile-level signatures...")
profile_signatures = []
for prof in NON_GENERALIST_PROFILES:
    players_in_prof = [p for p in player_profiles_out if p.profile == prof]
    sig = aggregate_profile(players_in_prof, profile_name=prof)
    profile_signatures.append(sig)

# ── Step 4: Save ──────────────────────────────────────────────────────────────

OUT_DIR.mkdir(parents=True, exist_ok=True)
save_skill_profiles(
    player_profiles=player_profiles_out,
    profile_signatures=profile_signatures,
    out_path=OUT_DIR / "skill_profiles.json",
)

# Summary
print()
print("=" * 60)
print("PROFILE SIGNATURES")
print("=" * 60)
for sig in profile_signatures:
    print()
    print(f"### {sig.profile}  ({sig.n_players} players)")
    for i, s in enumerate(sig.skills, 1):
        print(f"  {i}. {s.name}  (mentions: {s.n_mentions}, conf: {s.confidence_avg:.2f})")
        if s.aliases:
            print(f"     aliases: {s.aliases[:3]}")
        print(f"     {s.description[:120]}")

print()
print("=" * 60)
print(f"Saved: {OUT_DIR / 'skill_profiles.json'}")
print(f"Final cost: ${client.cumulative_cost():.6f}")
print("=" * 60)
