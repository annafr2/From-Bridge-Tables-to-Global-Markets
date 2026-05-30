"""
experiments/sanity_nego_agents.py
=================================
Stage 3 sanity check: does each profile's BRIDGE personality carry over into a
BUSINESS NEGOTIATION?

This is the cross-domain twin of sanity_bridge_agents.py. The same five profiles
(built from the SAME Stage 2 bridge skills) each respond to the SAME opening
offer in an M&A scenario. We expect the bridge-derived character to show:
Slam Hunter bold / Insurance quick-to-close / Fighter hard counters / NT
Specialist data-driven / Generalist middle-of-road.

It does NOT prove the alignment hypothesis (that comes in Stage 4 with win
rates over many sessions) — it confirms the wiring works and personality is
visible.

Outputs:
  - experiments/sanity_nego_results.json
  - experiments/sanity_nego_report.md   (expert/human-readable)

Run (needs GOOGLE_API_KEY in .env):
    python experiments/sanity_nego_agents.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

from src.shared.llm_client import LLMClient
from src.shared.prompts import PROFILE_NAMES
from src.sdk import build_nego_agents

load_dotenv()

# A single M&A scenario. The agent is the BUYER; the seller has just asked $13M.
SCENARIO: dict = {
    "title": "Acquiring a SaaS startup",
    "role": "buyer (acquirer)",
    "description": (
        "You want to acquire a profitable SaaS startup. Independent valuation "
        "puts fair value around $9M. The seller has just opened at $13M."
    ),
    "terms": {"price_musd": {"min": 5.0, "max": 15.0, "unit": "M USD"}},
    "your_target": {"price_musd": 8.0},
    "your_limit": {"price_musd": 11.0},
}
OPENING_OFFER = {"price_musd": 13.0}

JSON_OUT = Path("experiments/sanity_nego_results.json")
MD_OUT = Path("experiments/sanity_nego_report.md")


def run() -> dict:
    if not os.getenv("GOOGLE_API_KEY"):
        raise SystemExit("GOOGLE_API_KEY not set — add it to .env first.")

    client = LLMClient()
    agents = build_nego_agents(client=client)

    results: list[dict] = []
    for name in PROFILE_NAMES:
        out = agents[name].respond_to_offer(SCENARIO, current_offer=OPENING_OFFER)
        results.append({"profile": name, **out})

    # Integrity guard: exactly the five expected profiles, in order.
    got = [r["profile"] for r in results]
    assert got == PROFILE_NAMES, f"expected {PROFILE_NAMES}, got {got}"

    return {
        "date": date.today().isoformat(),
        "model": client.model,
        "scenario": SCENARIO,
        "opening_offer": OPENING_OFFER,
        "results": results,
        "total_cost_usd": round(client.cumulative_cost(), 6),
    }


def write_outputs(data: dict) -> None:
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Stage 3 — Negotiation Agent Sanity Report")
    lines.append("")
    lines.append(
        "> Cross-domain check: each agent's character comes from its **bridge** "
        "skills (Stage 2). Does that character show in a **business** "
        "negotiation? Reviewer question: is each response coherent with the "
        "profile, and within the scenario's rules?"
    )
    lines.append("")
    lines.append(f"- **Date:** {data['date']}")
    lines.append(f"- **Model:** {data['model']}")
    lines.append(f"- **Scenario:** {data['scenario']['title']} "
                 f"(you are the {data['scenario']['role']})")
    lines.append(f"- **Fair value ≈ $9M; your target $8M; walk-away $11M.**")
    lines.append(f"- **Seller's opening offer:** ${data['opening_offer']['price_musd']}M")
    lines.append(f"- **Total LLM cost:** ${data['total_cost_usd']:.6f}")
    lines.append("")
    lines.append("| Profile | Action | Counter price | Close? | Reasoning |")
    lines.append("|---------|--------|---------------|--------|-----------|")
    for r in data["results"]:
        price = r["offer"].get("price_musd", "—") if r["offer"] else "—"
        close = "yes" if r["willing_to_close"] else "no"
        reasoning = r["reasoning"].replace("|", "/")
        lines.append(
            f"| {r['profile']} | {r['action']} | {price} | {close} | {reasoning} |"
        )
    lines.append("")
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    data = run()
    write_outputs(data)

    print("=" * 70)
    print("STAGE 3 NEGOTIATION SANITY — same offer, five personalities")
    print("=" * 70)
    print(f"Scenario: buy a SaaS startup. Fair ~$9M, target $8M, limit $11M.")
    print(f"Seller opened at ${OPENING_OFFER['price_musd']}M.")
    print("-" * 70)
    for r in data["results"]:
        price = r["offer"].get("price_musd", "—") if r["offer"] else "—"
        print(f"\n{r['profile']}:")
        print(f"   action: {r['action']}  | counter: {price}  | "
              f"close: {r['willing_to_close']}")
        print(f"   reasoning: {r['reasoning']}")
    print(f"\nTotal cost: ${data['total_cost_usd']:.6f}")
    print(f"Saved: {JSON_OUT}")
    print(f"Saved: {MD_OUT}")


if __name__ == "__main__":
    main()
