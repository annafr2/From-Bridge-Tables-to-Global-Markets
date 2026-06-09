"""
notebooks/inverse_prompt_control.py
===================================
Anti-tautology control for the STYLE->STYLE result (ρ=+0.80).

Worry: "the agent bargains aggressively only because we *labelled* it an
aggressive profile." If true, the transfer is circular.

Test: build SKILL-SWAPPED negotiation agents — each profile keeps its identity but
is injected with the OPPOSITE profile's bridge skills (Fighter gets Insurance's
skills, etc.). Then re-measure negotiation aggression.

  - If behaviour follows the SWAPPED SKILLS, the aggression of bridge-aggressive
    profiles drops -> the bridge<->negotiation aggression correlation FLIPS/breaks
    -> the original +0.80 was driven by the (bridge-derived) skills, NOT the label.
    => NON-tautological.  ✅
  - If behaviour follows the LABEL, ρ stays ~+0.80 even with swapped skills
    => tautological.  ❌

Costs ~60 LLM calls (one negotiation set with the swapped agents).
Output: console comparison (matched ρ vs inverse-control ρ).
"""

from __future__ import annotations

import dataclasses
import json
import logging

from dotenv import load_dotenv

from scipy.stats import spearmanr

from src.shared.llm_client import LLMClient
from src.shared.prompts import load_profile_signatures
from src.stage3_agents.nego_agent import NegotiationAgent
from src.stage4_simulate.negotiation import SCENARIOS, _run_one_negotiation
from notebooks.style_alignment import bridge_aggression

load_dotenv()
logging.disable(logging.WARNING)

# Aggression order (from the style result). The inverse pairs each profile with
# its aggression-opposite; the middle (NT) maps to itself.
ORDER = ["Fighter", "Slam Hunter", "NT Specialist", "Generalist", "Insurance Player"]
INVERSE = {p: ORDER[len(ORDER) - 1 - i] for i, p in enumerate(ORDER)}


def _nego_aggression(agents: dict, client: LLMClient) -> dict[str, float]:
    spans = {s["title"]: (s["seller_open"], s["seller_floor"]) for s in SCENARIOS}
    depth: dict[str, list[float]] = {p: [] for p in agents}
    for profile, agent in agents.items():
        for sc in SCENARIOS:
            for run in range(3):
                rows: list[dict] = []
                _run_one_negotiation(agent, sc, rows, profile, run)
                for r in rows:
                    if r.get("action") == "counter" and r.get("offer"):
                        price = r["offer"].get("price_musd")
                        if price is None:
                            continue
                        open_, floor = spans[sc["title"]]
                        depth[profile].append(max(0.0, min(1.3, (open_ - price) / (open_ - floor))))
    return {p: (sum(v) / len(v) if v else 0.0) for p, v in depth.items()}


def main() -> None:
    sigs = load_profile_signatures()
    client = LLMClient(model="gemini-2.5-flash-lite")

    # Skill-swapped agents: identity stays P, skills become INVERSE(P)'s.
    swapped = {}
    for p in ORDER:
        sig_swapped = dataclasses.replace(sigs[p], skills=sigs[INVERSE[p]].skills)
        swapped[p] = NegotiationAgent(sig_swapped, client=client)

    inv_aggr = _nego_aggression(swapped, client)
    b = bridge_aggression()

    profiles = [p for p in ORDER if p in b.index]
    rho, pval = spearmanr([b[p] for p in profiles], [inv_aggr[p] for p in profiles])

    print("Inverse-prompt control (identity P, skills of INVERSE(P)):")
    print(f"  {'profile':18s} {'bridge_aggr':>12} {'nego_aggr(swapped)':>18}")
    for p in profiles:
        print(f"  {p:18s} {b[p]:>12.3f} {inv_aggr[p]:>18.3f}  (skills <- {INVERSE[p]})")
    print()
    print(f"  MATCHED  style->style rho = +0.80")
    print(f"  INVERSE control       rho = {rho:+.2f} (p={pval:.3f})")
    print(f"  LLM cost this run: ${client.cumulative_cost():.4f}")
    print()
    if rho < 0.3:
        print("  => Aggression FOLLOWED THE SWAPPED SKILLS, not the label.")
        print("     The +0.80 is skill-driven, NOT tautological.  PASS.")
    else:
        print("  => Aggression stayed high despite swapped skills -> possible tautology.")


if __name__ == "__main__":
    main()
