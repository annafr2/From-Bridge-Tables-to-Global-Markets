"""
experiments/sanity_bridge_agents.py
===================================
Stage 3 sanity check: do the five profile agents BID DIFFERENTLY on the
same hand(s)?

This is a small real-LLM check (5 agents x N hands). It does NOT prove the
research hypothesis — it confirms the wiring works, that personality visibly
changes the bid, and that every bid is LEGAL (no hallucinated calls).

It produces TWO outputs:
  - experiments/sanity_bridge_results.json   (machine-readable record)
  - experiments/sanity_bridge_report.md      (human/expert-readable report)

The Markdown report is designed to be handed to a bridge expert for review:
it shows each hand, the auction, and every agent's bid + reasoning + a
legality flag, so the expert can confirm the calls are sound and not invented.

Run (needs GOOGLE_API_KEY in .env):
    python experiments/sanity_bridge_agents.py
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

# Test hands chosen to separate the profiles. Each is a dict by suit plus the
# auction the agent faces and a short note on what a correct read looks like.
TEST_CASES: list[dict] = [
    {
        "label": "Strong balanced-ish, partner opened 1S (slam-interest hand)",
        "hand": {"S": "AKQ72", "H": "AK4", "D": "A83", "C": "Q2"},
        "hcp": 20,
        "auction": ["1S", "Pass"],
        "note": "20 HCP with a spade fit — slam exploration is reasonable; "
                "a cautious player signs off in 4S.",
    },
    {
        "label": "Flat 12-count, opponents opened 1H (defensive decision)",
        "hand": {"S": "KJ95", "H": "Q72", "D": "KQ4", "C": "J83"},
        "hcp": 12,
        "auction": ["1H"],
        "note": "Balanced minimum over 1H — pass / double / 1S overcall are "
                "all defensible; a Fighter is more likely to act.",
    },
]

JSON_OUT = Path("experiments/sanity_bridge_results.json")
MD_OUT = Path("experiments/sanity_bridge_report.md")


def _fmt_hand(h: dict[str, str]) -> str:
    return f"S:{h['S']}  H:{h['H']}  D:{h['D']}  C:{h['C']}"


def run() -> dict:
    if not os.getenv("GOOGLE_API_KEY"):
        raise SystemExit("GOOGLE_API_KEY not set — add it to .env first.")

    client = LLMClient()           # one shared client => one combined cost log
    agents = build_bridge_agents(client=client)

    cases_out: list[dict] = []
    for case in TEST_CASES:
        results: list[dict] = []
        for name in PROFILE_NAMES:
            out = agents[name].make_bid(case["hand"], case["auction"])
            results.append({"profile": name, **out})

        # Integrity guard: exactly the 5 expected profiles, in order, no dupes.
        got = [r["profile"] for r in results]
        assert got == PROFILE_NAMES, (
            f"Profile integrity check FAILED: expected {PROFILE_NAMES}, got {got}"
        )
        # Each legal record's bid must match the normalised raw_bid.
        for r in results:
            if r["legal"]:
                assert r["bid"] == r["raw_bid"] or r["bid"] != "Pass", (
                    f"bid/raw mismatch for {r['profile']}: {r}"
                )

        cases_out.append({
            "label": case["label"],
            "hand": case["hand"],
            "hcp": case["hcp"],
            "auction": case["auction"],
            "note": case["note"],
            "results": results,
            "distinct_bids": sorted({r["bid"] for r in results}),
        })

    return {
        "date": date.today().isoformat(),
        "model": client.model,
        "cases": cases_out,
        "total_cost_usd": round(client.cumulative_cost(), 6),
    }


def write_json(data: dict) -> None:
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def write_markdown(data: dict) -> None:
    lines: list[str] = []
    lines.append("# Stage 3 — Bridge Agent Sanity Report")
    lines.append("")
    lines.append(
        "> For bridge-expert review. Each agent is an LLM conditioned on a "
        "player profile whose skills were extracted from **real** tournament "
        "hands (Stage 2). The question for the reviewer: **are these calls "
        "legal and sensible for the stated profile, with no hallucinated bids?**"
    )
    lines.append("")
    lines.append(f"- **Date:** {data['date']}")
    lines.append(f"- **Model:** {data['model']}")
    lines.append(f"- **Total LLM cost:** ${data['total_cost_usd']:.6f}")
    lines.append("")

    for i, case in enumerate(data["cases"], 1):
        lines.append(f"## Hand {i}: {case['label']}")
        lines.append("")
        lines.append(f"- **Hand:** `{_fmt_hand(case['hand'])}`  ({case['hcp']} HCP)")
        lines.append(f"- **Auction so far:** `{' '.join(case['auction'])}`")
        lines.append(f"- **Reviewer note:** {case['note']}")
        lines.append(f"- **Distinct bids across agents:** "
                     f"{', '.join(case['distinct_bids'])}")
        lines.append("")
        lines.append("| Profile | Bid | Legal? | Agent's reasoning |")
        lines.append("|---------|-----|--------|-------------------|")
        for r in case["results"]:
            legal = "✅" if r["legal"] else "❌ (fell back to Pass)"
            reasoning = r["reasoning"].replace("|", "/")
            lines.append(
                f"| {r['profile']} | **{r['bid']}** | {legal} | {reasoning} |"
            )
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("### How to read the *Legal?* column")
    lines.append(
        "A ✅ means the bid passed our local Laws-of-Bridge legality check "
        "(correct rank over the previous contract, valid double/redouble "
        "context). A ❌ means the LLM proposed an illegal call and the system "
        "automatically substituted **Pass** — this is the guardrail against "
        "hallucinated bids. Zero ❌ rows means every agent produced a legal call."
    )
    lines.append("")

    MD_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    data = run()
    write_json(data)
    write_markdown(data)

    # Console summary
    print("=" * 70)
    print("STAGE 3 SANITY CHECK — clean run")
    print("=" * 70)
    for i, case in enumerate(data["cases"], 1):
        print(f"\nHand {i}: {_fmt_hand(case['hand'])} ({case['hcp']} HCP), "
              f"auction {' '.join(case['auction'])}")
        for r in case["results"]:
            flag = "" if r["legal"] else "  <ILLEGAL->PASS>"
            print(f"   {r['profile']:18s} -> {r['bid']:4s}{flag}")
        print(f"   distinct bids: {case['distinct_bids']}")
    print(f"\nTotal cost: ${data['total_cost_usd']:.6f}")
    print(f"Saved: {JSON_OUT}")
    print(f"Saved: {MD_OUT}  (give this one to the bridge expert)")


if __name__ == "__main__":
    main()
