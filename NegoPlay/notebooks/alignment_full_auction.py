"""
notebooks/alignment_full_auction.py
===================================
Task T-D — re-run the cross-domain alignment on the FIXED metrics:
  - bridge: full multi-round auction, scored by double-dummy
    (results/stage4/bridge_auction_winrates.csv, includes the monkey)
  - negotiation: surplus captured (results/stage4/negotiation_winrates.csv);
    the monkey is run through the same scenarios here (free).

It builds the normalized floor->ceiling skill scale Rami asked for
    skill% = (agent - monkey) / (1 - monkey)        # monkey = 0%, perfect = 100%
in BOTH domains, computes Spearman ρ over the 5 profiles, and writes a scatter
plus a report. No LLM cost (bridge is read from disk; the negotiation monkey is
random/free).

Outputs:
  results/stage4/alignment_full_auction_report.md
  docs/images/alignment_full_auction.png
"""

from __future__ import annotations

import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr

from src.stage4_simulate.monkey_agent import MONKEY_PROFILE, MonkeyAgent
from src.stage4_simulate.negotiation import SCENARIOS, _run_one_negotiation

RESULTS = Path("results/stage4")
BRIDGE_CSV = RESULTS / "bridge_auction_winrates.csv"
NEGO_CSV = RESULTS / "negotiation_winrates.csv"
REPORT = RESULTS / "alignment_full_auction_report.md"
PLOT = Path("docs/images/alignment_full_auction.png")

PROFILE_COLORS = {
    "Generalist": "#9aa0a6", "Slam Hunter": "#d62728",
    "Insurance Player": "#1f77b4", "Fighter": "#ff7f0e", "NT Specialist": "#2ca02c",
}


def _monkey_negotiation() -> float:
    monkey = MonkeyAgent(seed=42)
    logs, scores = [], []
    for sc in SCENARIOS:
        for run in range(3):
            scores.append(_run_one_negotiation(monkey, sc, logs, MONKEY_PROFILE, run))
    return st.mean(scores)


def _normalize(x: float, floor: float) -> float:
    """skill% = (x - monkey) / (1 - monkey), clipped to [0, 1]."""
    denom = 1.0 - floor
    if denom <= 0:
        return 0.0
    return max(0.0, min(1.0, (x - floor) / denom))


def build() -> dict:
    bdf = pd.read_csv(BRIDGE_CSV)
    bridge = dict(zip(bdf["profile"], bdf["bridge_dd_winrate"]))
    ndf = pd.read_csv(NEGO_CSV)
    nego = dict(zip(ndf["profile"], ndf["negotiation_winrate"]))

    monkey_bridge = bridge.pop(MONKEY_PROFILE, 0.0)
    monkey_nego = _monkey_negotiation()

    profiles = [p for p in bridge if p in nego]
    rows = []
    for p in profiles:
        rows.append({
            "profile": p,
            "bridge_raw": bridge[p],
            "nego_raw": nego[p],
            "bridge_skill": _normalize(bridge[p], monkey_bridge),
            "nego_skill": _normalize(nego[p], monkey_nego),
        })
    df = pd.DataFrame(rows).sort_values("bridge_skill", ascending=False).reset_index(drop=True)

    rho, p = spearmanr(df["bridge_skill"], df["nego_skill"])
    return {
        "df": df, "rho": float(rho), "p": float(p),
        "monkey_bridge": monkey_bridge, "monkey_nego": monkey_nego,
    }


def make_plot(res: dict) -> None:
    df, rho = res["df"], res["rho"]
    fig, ax = plt.subplots(figsize=(9, 7))
    for _, r in df.iterrows():
        c = PROFILE_COLORS.get(r["profile"], "#333")
        ax.scatter(r["bridge_skill"], r["nego_skill"], s=200, color=c,
                   edgecolors="black", zorder=3)
        ax.annotate(r["profile"], (r["bridge_skill"], r["nego_skill"]),
                    xytext=(8, 5), textcoords="offset points", fontsize=10)
    ax.plot([0, 1], [0, 1], ls="--", color="gray", alpha=0.6, label="perfect agreement")
    ax.scatter(0, 0, marker="X", s=160, color="#c0392b", zorder=3)
    ax.annotate("Monkey = 0% (floor)", (0, 0), xytext=(10, -14),
                textcoords="offset points", fontsize=9, color="#c0392b")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Bridge skill %  (monkey=0, perfect=100)  — full auction, double-dummy", fontsize=11)
    ax.set_ylabel("Negotiation skill %  (monkey=0, perfect=100)", fontsize=11)
    ax.set_title(f"Cross-domain alignment on FIXED metrics\nSpearman ρ = {rho:+.2f}",
                 fontweight="bold", fontsize=13)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    PLOT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_report(res: dict) -> None:
    df, rho, p = res["df"], res["rho"], res["p"]
    verdict = ("SUPPORTED" if rho >= 0.70 else
               "PARTIALLY SUPPORTED" if rho >= 0.5 else
               "WEAK / INCONCLUSIVE" if rho >= 0 else "NEGATIVE")
    lines = [
        "# Stage 4 — Alignment on the fixed metrics (full auction + double-dummy)",
        "",
        f"**Spearman ρ = {rho:+.3f}** (p = {p:.3f}, n = {len(df)} profiles) → **{verdict}**",
        "",
        f"Floors (the random monkey): bridge = {res['monkey_bridge']:.3f}, "
        f"negotiation = {res['monkey_nego']:.3f}. Performance is normalized to "
        "skill% = (agent − monkey) / (1 − monkey).",
        "",
        "| Profile | Bridge raw | Bridge skill% | Nego raw | Nego skill% |",
        "|---------|-----------|---------------|----------|-------------|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['profile']} | {r['bridge_raw']:.3f} | {r['bridge_skill']:.2f} | "
            f"{r['nego_raw']:.3f} | {r['nego_skill']:.2f} |")
    lines += [
        "",
        "## Notes",
        "- Bridge now uses a real multi-round auction scored by double-dummy, so a "
        "random agent scores ~0 and aggressive profiles are no longer auto-punished.",
        "- The monkey anchors 0% in BOTH domains (skill beats random in each).",
        "- n=5 keeps cross-domain power low; ρ is an indication.",
        "",
        "*Generated by notebooks/alignment_full_auction.py (no LLM cost).*",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    res = build()
    make_plot(res)
    write_report(res)
    print(res["df"].to_string(index=False))
    print(f"\nmonkey: bridge={res['monkey_bridge']:.3f} nego={res['monkey_nego']:.3f}")
    print(f"Spearman rho = {res['rho']:+.3f} (p={res['p']:.3f})")
    print(f"saved {PLOT} and {REPORT}")


if __name__ == "__main__":
    main()
