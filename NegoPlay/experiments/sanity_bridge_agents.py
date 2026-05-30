"""
experiments/sanity_bridge_agents.py
===================================
Stage 3 sanity check: do the five profile agents BID DIFFERENTLY on the
same hand?

This is the first real LLM call in Stage 3. It is intentionally tiny
(5 agents x 1 decision = 5 calls, ~$0.001 total). It does NOT prove the
research hypothesis — it only confirms the wiring works and that personality
visibly changes the bid.

Run (needs GOOGLE_API_KEY in .env):
    python experiments/sanity_bridge_agents.py

Output: prints each agent's bid + reasoning, and saves a JSON record to
experiments/sanity_bridge_results.json for the record.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

# Allow running as a plain script: add project root to sys.path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

from src.shared.llm_client import LLMClient
from src.shared.prompts import PROFILE_NAMES
from src.sdk import build_bridge_agents

load_dotenv()

# A genuinely slam-interesting hand: 20 HCP, strong 5-card spade suit.
# A Slam Hunter should get excited; an Insurance Player should stay calm.
TEST_HAND = {"S": "AKQ72", "H": "AK4", "D": "A83", "C": "Q2"}

# Auction: partner opened 1S, opponent passed. Now it's our turn.
TEST_AUCTION = ["1S", "Pass"]

OUT = Path("experiments/sanity_bridge_results.json")


def main() -> None:
    if not os.getenv("GOOGLE_API_KEY"):
        raise SystemExit("GOOGLE_API_KEY not set — add it to .env first.")

    # One shared client => one combined cost log / budget across all 5 agents.
    client = LLMClient()
    agents = build_bridge_agents(client=client)

    print("=" * 70)
    print("STAGE 3 SANITY CHECK — same hand, five personalities")
    print("=" * 70)
    print(f"Hand:    S:{TEST_HAND['S']}  H:{TEST_HAND['H']}  "
          f"D:{TEST_HAND['D']}  C:{TEST_HAND['C']}  (20 HCP)")
    print(f"Auction: {' '.join(TEST_AUCTION)}  (partner opened 1S)")
    print("-" * 70)

    records = []
    for name in PROFILE_NAMES:
        agent = agents[name]
        out = agent.make_bid(TEST_HAND, TEST_AUCTION)
        print(f"\n{name}:")
        print(f"   BID:       {out['bid']}   (legal: {out['legal']})")
        print(f"   reasoning: {out['reasoning']}")
        records.append({"profile": name, **out})

    # Summary: how many DISTINCT bids did we see?
    distinct = sorted({r["bid"] for r in records})
    print("\n" + "=" * 70)
    print(f"Distinct bids across 5 agents: {distinct}")
    print(f"Total LLM cost so far: ${client.cumulative_cost():.6f}")
    print("=" * 70)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "date": date.today().isoformat(),
                "hand": TEST_HAND,
                "auction": TEST_AUCTION,
                "results": records,
                "distinct_bids": distinct,
                "total_cost_usd": round(client.cumulative_cost(), 6),
            },
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
