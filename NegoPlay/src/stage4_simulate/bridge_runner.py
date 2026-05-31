"""
src/stage4_simulate/bridge_runner.py
====================================
Stage 4a runner: each profile bids many deals against an OBJECTIVE par target,
and we record a per-profile bridge "win rate".

Design decisions (chosen with the researcher):
  - OBJECTIVE opponent: every profile bids the SAME deals; the score is how
    close the bid lands to the par contract level (bridge_game.score_bid).
    No agent-vs-agent randomness — cleaner profile separation.
  - 3 RUNS PER DEAL: because Gemini is non-deterministic even at temperature 0,
    each (profile, deal) is bid 3 times and we take the MAJORITY bid (ties ->
    the bid with the highest objective score). This shrinks per-call noise.
  - Every raw bid is persisted (JSONL) so the analysis is fully reproducible
    from saved data even if regeneration differs.

The agent sits South and is the RESPONDER: partner (North) opens with a VARYING
strength, cycled across boards (1C~13 / 1NT~16 / 2C~22 HCP). The agent sees only
its OWN 13 cards (incomplete information) plus partner's opening bid. Par is
computed from the SAME information the agent has — its own HCP plus partner's
shown HCP — so scoring is fair, par varies across deals (including ~6% slam
deals), and the profiles separate (a Slam Hunter is penalised for overbidding
weak hands, an Insurance Player for underbidding strong ones).

Outputs:
  results/stage4/bridge_simulations.jsonl   (one row per profile x deal x run)
  results/stage4/bridge_winrates.csv        (one row per profile)
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path

# Allow running as a plain script + load the API key from .env.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from dotenv import load_dotenv

from src.shared.llm_client import LLMClient
from src.shared.prompts import PROFILE_NAMES
from src.sdk import build_bridge_agents
from src.stage4_simulate.bridge_game import Deal, deal_board, score_bid

load_dotenv()
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results/stage4")
JSONL_PATH = RESULTS_DIR / "bridge_simulations.jsonl"
WINRATES_PATH = RESULTS_DIR / "bridge_winrates.csv"

DEFAULT_N_BOARDS = 50
DEFAULT_RUNS_PER_DEAL = 3
SEED = 42

# Partner opens with a strength that cycles across boards. Both the agent (via
# the auction token) and the objective scorer (via the HCP) use this same value,
# so the agent is fairly informed and par spans all contract levels.
PARTNER_OPENINGS = [
    ("1C", 13),   # sound minimum opening
    ("1NT", 16),  # strong balanced
    ("2C", 22),   # very strong / game-forcing
]


def _majority_bid(bids: list[str], partnership_hcp: int) -> str:
    """Representative bid across runs: majority vote, ties -> best score."""
    if not bids:
        return "Pass"
    counts = Counter(bids)
    top = counts.most_common()
    best_n = top[0][1]
    tied = [b for b, n in top if n == best_n]
    if len(tied) == 1:
        return tied[0]
    return max(tied, key=lambda b: score_bid(b, partnership_hcp).score)


def run_bridge_stage4(
    n_boards: int = DEFAULT_N_BOARDS,
    runs_per_deal: int = DEFAULT_RUNS_PER_DEAL,
    seed: int = SEED,
    profiles: list[str] | None = None,
    client: LLMClient | None = None,
) -> dict[str, float]:
    """Run the Stage 4a bridge experiment.

    Returns:
        Mapping {profile: mean_score} — the bridge "win rate" per profile
        (mean objective score across all deals, in [0, 1]).
    """
    profiles = profiles or PROFILE_NAMES
    client = client or LLMClient()
    agents = build_bridge_agents(client=client)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if JSONL_PATH.exists():
        JSONL_PATH.unlink()

    deals: list[Deal] = [deal_board(b, seed=seed) for b in range(1, n_boards + 1)]
    profile_scores: dict[str, list[float]] = {p: [] for p in profiles}

    for i, deal in enumerate(deals):
        south_hand = deal.hands["S"]
        partner_bid, partner_hcp = PARTNER_OPENINGS[i % len(PARTNER_OPENINGS)]
        partnership_hcp = deal.hcp("S") + partner_hcp
        auction = [partner_bid]

        for profile in profiles:
            agent = agents[profile]
            run_bids: list[str] = []
            for run_idx in range(runs_per_deal):
                out = agent.make_bid(south_hand, auction)
                run_bids.append(out["bid"])
                _append_jsonl({
                    "profile": profile,
                    "board": deal.board,
                    "run": run_idx,
                    "partner_open": partner_bid,
                    "bid": out["bid"],
                    "legal": out["legal"],
                    "partnership_hcp": partnership_hcp,
                    "reasoning": out.get("reasoning", ""),
                })

            chosen = _majority_bid(run_bids, partnership_hcp)
            sc = score_bid(chosen, partnership_hcp)
            profile_scores[profile].append(sc.score)
            logger.info(
                "board=%d profile=%-16s bids=%s -> %s (par=%s, score=%.2f)",
                deal.board, profile, run_bids, chosen, sc.par_level_class, sc.score,
            )

    winrates = {
        p: round(sum(scores) / len(scores), 4) if scores else 0.0
        for p, scores in profile_scores.items()
    }
    _write_winrates(winrates, n_boards, runs_per_deal, client.cumulative_cost())
    return winrates


def _append_jsonl(record: dict) -> None:
    with JSONL_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_winrates(
    winrates: dict[str, float],
    n_boards: int,
    runs_per_deal: int,
    total_cost: float,
) -> None:
    with WINRATES_PATH.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["profile", "bridge_winrate", "n_boards", "runs_per_deal"])
        for profile, rate in winrates.items():
            w.writerow([profile, rate, n_boards, runs_per_deal])
    logger.info("Wrote %s (total LLM cost $%.4f)", WINRATES_PATH, total_cost)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not os.getenv("GOOGLE_API_KEY"):
        raise SystemExit("GOOGLE_API_KEY not set — add it to .env first.")
    winrates = run_bridge_stage4()
    print("\n=== BRIDGE WIN RATES (per profile) ===")
    for p, r in sorted(winrates.items(), key=lambda x: -x[1]):
        print(f"  {p:18s} {r:.3f}")


if __name__ == "__main__":
    main()


__all__ = ["run_bridge_stage4", "JSONL_PATH", "WINRATES_PATH"]
