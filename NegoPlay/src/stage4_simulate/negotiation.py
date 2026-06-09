"""
src/stage4_simulate/negotiation.py
==================================
Stage 4b: each profile negotiates the SAME business scenarios against the SAME
fixed, objective counterpart, and we record a per-profile negotiation "win rate".

Design — mirrors Stage 4a (bridge) so the two domains are comparable:
  - OBJECTIVE counterpart: a simple, fixed seller/other-side that opens high and
    concedes a little each round. Every profile faces the identical counterpart
    on the identical scenarios, so differences come from the PROFILE, not from a
    random opponent. (Same logic as the objective par target in bridge.)
  - The agent is always the BUYER; it wants a LOW price. "Winning" = capturing
    more of the surplus between the fair value and the seller's opening ask.
  - 3 runs per scenario, results averaged (Gemini is non-deterministic).
  - Every raw turn is persisted to JSONL for reproducible analysis.

Win-rate metric (per scenario), in [0, 1]:
    surplus_captured = (seller_open - deal_price) / (seller_open - seller_floor)
  i.e. how far the buyer pulled the price down across the seller's FULL
  concession range (from its opening ask down to its floor). 1.0 = pushed the
  seller all the way to its floor; 0.0 = paid the asking price (or walked away
  with no deal). The seller bargains hard (see _seller_respond), so reaching the
  floor is difficult and the profiles separate. Averaged over scenarios = the
  win rate.

Outputs:
  results/stage4/negotiation_simulations.jsonl
  results/stage4/negotiation_winrates.csv
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from dotenv import load_dotenv

from src.shared.llm_client import LLMClient
from src.shared.prompts import PROFILE_NAMES
from src.sdk import build_nego_agents

load_dotenv()
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results/stage4")
JSONL_PATH = RESULTS_DIR / "negotiation_simulations.jsonl"
WINRATES_PATH = RESULTS_DIR / "negotiation_winrates.csv"

DEFAULT_RUNS = 3
SEED = 42

# Four business scenarios. The agent is always the BUYER (wants a low price).
# fair_value = independent valuation; seller_open = the opening ask.
# The objective seller starts at seller_open and will accept anything >= its
# floor (a bit below fair value), conceding gradually if pushed.
SCENARIOS: list[dict] = [
    {
        "title": "Acquire a SaaS startup",
        "fair_value": 9.0, "seller_open": 13.0, "seller_floor": 8.0,
    },
    {
        "title": "Buy out a co-founder's equity",
        "fair_value": 4.0, "seller_open": 6.5, "seller_floor": 3.5,
    },
    {
        "title": "Acquire a competitor's customer book",
        "fair_value": 2.0, "seller_open": 3.5, "seller_floor": 1.7,
    },
    {
        "title": "Purchase enterprise software licence (multi-year)",
        "fair_value": 1.2, "seller_open": 2.0, "seller_floor": 1.0,
    },
]

MAX_ROUNDS = 4   # buyer counter -> seller concede, up to this many rounds

# Seller "red line" (walk-away), CALIBRATED FROM REAL DATA.
# We validated against 5,247 real Craigslist negotiations
# (notebooks/validate_negotiation_features.py): aggressive lowballing there
# CAPTURES MORE SURPLUS and the deal rate barely drops (~0.83 vs ~0.86) — real
# sellers tolerate hard bargaining and only walk away on *absurd* offers. So the
# red line is mild and data-grounded: the seller walks only when the buyer offers
# well below its own floor (< 70% of floor — an offer it would lose money on),
# after a couple of such insults. This reproduces the real finding (aggression
# pays) while still penalising a truly nonsensical agent, rather than the earlier
# over-harsh threshold that wrongly crushed all aggression.
INSULT_FACTOR = 0.70   # only an offer below floor * 0.70 is "absurd" (data-grounded)
MAX_INSULTS = 2        # walk away on the 2nd absurd offer


def _build_scenario_for_agent(sc: dict) -> dict:
    """Turn a scenario spec into the dict shape NegotiationAgent expects."""
    return {
        "title": sc["title"],
        "role": "buyer (acquirer) — you want the LOWEST price",
        "description": (
            f"Independent valuation puts fair value at ${sc['fair_value']}M. "
            f"The seller has opened at ${sc['seller_open']}M. "
            "Negotiate the price down."
        ),
        "terms": {"price_musd": {
            "min": sc["seller_floor"], "max": sc["seller_open"], "unit": "M USD"}},
        "your_target": {"price_musd": sc["fair_value"] * 0.9},
        "your_limit": {"price_musd": sc["fair_value"] * 1.2},
    }


def _seller_respond(buyer_price: float, sc: dict, current_ask: float) -> tuple[bool, float]:
    """Objective seller that BARGAINS HARD.

    The seller accepts only if the buyer's offer is close to the seller's current
    ask (within 5%). Otherwise it concedes a small, fixed fraction (15%) of the
    gap down toward the buyer — but never below its floor. This makes the seller
    a real obstacle: a buyer who anchors low and holds firm pulls the price down
    further; a buyer who caves quickly pays more. That is exactly the behaviour
    that should separate aggressive profiles from cautious ones.

    Returns (accepted, new_ask).
    """
    floor = sc["seller_floor"]
    # Accept if the buyer has met (or nearly met) the current ask.
    if buyer_price >= current_ask * 0.95:
        return True, min(buyer_price, current_ask)
    # Otherwise concede 15% of the way toward the buyer, never below the floor.
    target = max(buyer_price, floor)
    new_ask = current_ask - 0.15 * (current_ask - target)
    new_ask = max(new_ask, floor)
    return False, round(new_ask, 3)


def _run_one_negotiation(agent, sc: dict, log_rows: list[dict], profile: str, run: int) -> float:
    """Play one buyer-vs-objective-seller negotiation. Returns surplus captured [0,1]."""
    scen = _build_scenario_for_agent(sc)
    current_ask = sc["seller_open"]
    floor = sc["seller_floor"]
    history: list[dict] = []
    deal_price: float | None = None
    insults = 0

    for rnd in range(MAX_ROUNDS):
        resp = agent.respond_to_offer(
            scen, current_offer={"price_musd": current_ask}, history=history)
        action = resp["action"]
        log_rows.append({
            "profile": profile, "scenario": sc["title"], "run": run, "round": rnd,
            "seller_ask": current_ask, "action": action,
            "offer": resp["offer"], "reasoning": resp.get("reasoning", ""),
        })

        if action == "accept":
            deal_price = current_ask
            break
        if action == "walk_away":
            deal_price = None
            break
        # counter
        buyer_price = resp["offer"].get("price_musd") if resp["offer"] else None
        if buyer_price is None:
            buyer_price = current_ask  # no number -> treat as no progress

        # Seller red line: an insulting (near-floor) offer erodes patience; on the
        # MAX_INSULTS-th insult the seller walks and the deal collapses.
        if buyer_price < floor * INSULT_FACTOR:
            insults += 1
            if insults >= MAX_INSULTS:
                deal_price = None  # seller walked away -> no deal
                break

        accepted, new_ask = _seller_respond(buyer_price, sc, current_ask)
        history.append({"buyer_offer": buyer_price, "seller_ask": current_ask})
        if accepted:
            deal_price = buyer_price
            break
        current_ask = new_ask
    else:
        # ran out of rounds — deal closes at the last ask if buyer was engaged
        deal_price = current_ask

    # Surplus captured: how far the buyer pulled the price down across the
    # seller's full concession range [floor, open]. 1.0 = reached the floor.
    span = sc["seller_open"] - sc["seller_floor"]
    if deal_price is None:        # walked away, no deal = captured nothing
        surplus = 0.0
    elif span <= 0:
        surplus = 1.0
    else:
        surplus = (sc["seller_open"] - deal_price) / span
        surplus = max(0.0, min(1.0, surplus))
    return round(surplus, 4)


def run_negotiation_stage4(
    runs: int = DEFAULT_RUNS,
    profiles: list[str] | None = None,
    client: LLMClient | None = None,
) -> dict[str, float]:
    """Run Stage 4b. Returns {profile: mean surplus captured} in [0, 1]."""
    profiles = profiles or PROFILE_NAMES
    client = client or LLMClient()
    agents = build_nego_agents(client=client)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if JSONL_PATH.exists():
        JSONL_PATH.unlink()

    log_rows: list[dict] = []
    profile_scores: dict[str, list[float]] = {p: [] for p in profiles}

    for profile in profiles:
        agent = agents[profile]
        for sc in SCENARIOS:
            for run in range(runs):
                surplus = _run_one_negotiation(agent, sc, log_rows, profile, run)
                profile_scores[profile].append(surplus)
                logger.info("%-16s | %-38s run%d -> surplus %.2f",
                            profile, sc["title"], run, surplus)

    with JSONL_PATH.open("w", encoding="utf-8") as f:
        for row in log_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    winrates = {
        p: round(sum(s) / len(s), 4) if s else 0.0
        for p, s in profile_scores.items()
    }
    _write_winrates(winrates, runs, client.cumulative_cost())
    return winrates


def _write_winrates(winrates: dict[str, float], runs: int, total_cost: float) -> None:
    with WINRATES_PATH.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["profile", "negotiation_winrate", "n_scenarios", "runs"])
        for p, r in winrates.items():
            w.writerow([p, r, len(SCENARIOS), runs])
    logger.info("Wrote %s (total LLM cost $%.4f)", WINRATES_PATH, total_cost)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not os.getenv("GOOGLE_API_KEY"):
        raise SystemExit("GOOGLE_API_KEY not set — add it to .env first.")
    winrates = run_negotiation_stage4()
    print("\n=== NEGOTIATION WIN RATES (surplus captured, per profile) ===")
    for p, r in sorted(winrates.items(), key=lambda x: -x[1]):
        print(f"  {p:18s} {r:.3f}")


if __name__ == "__main__":
    main()


__all__ = ["run_negotiation_stage4", "SCENARIOS", "JSONL_PATH", "WINRATES_PATH"]
