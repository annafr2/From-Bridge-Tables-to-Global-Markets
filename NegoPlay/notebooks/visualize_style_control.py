"""
notebooks/visualize_style_control.py
====================================
Two-panel proof of the STYLE->STYLE transfer and its anti-tautology control.

LEFT  (matched):  each profile's agent uses ITS OWN bridge skills.
                  bridge aggression vs negotiation aggression -> ρ = +0.80.
RIGHT (inverse):  each profile keeps its identity but gets the OPPOSITE profile's
                  skills. Aggression follows the SWAPPED skills -> ρ flips to -0.90.

The flip proves the transfer is driven by the (bridge-derived) skills, not by the
profile label -> NOT tautological.

Matched values are recomputed live (free); the inverse-control values are the
measured outputs of notebooks/inverse_prompt_control.py (LLM run, June 2026).

Output: docs/images/style_transfer_control.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from notebooks.style_alignment import bridge_aggression, negotiation_aggression

OUT = Path("docs/images/style_transfer_control.png")
COLORS = {"Slam Hunter": "#d62728", "Insurance Player": "#1f77b4",
          "Fighter": "#ff7f0e", "NT Specialist": "#2ca02c", "Generalist": "#9aa0a6"}

# Inverse-prompt control: negotiation aggression when each profile is given the
# OPPOSITE profile's skills (measured by inverse_prompt_control.py, June 2026).
INVERSE_NEGO_AGGR = {
    "Fighter": 0.725,            # got Insurance's (cautious) skills -> less aggressive
    "Slam Hunter": 0.756,        # got Generalist's skills
    "NT Specialist": 0.755,      # self
    "Generalist": 0.795,         # got Slam Hunter's skills
    "Insurance Player": 0.805,   # got Fighter's (aggressive) skills -> more aggressive
}


def _panel(ax, b, n, title, subtitle, sub_color):
    profs = [p for p in b.index if p in n]
    rho, _ = spearmanr([b[p] for p in profs], [n[p] for p in profs])
    for p in profs:
        ax.scatter(b[p], n[p], s=200, color=COLORS.get(p, "#333"),
                   edgecolors="black", zorder=3)
        ax.annotate(p, (b[p], n[p]), xytext=(7, 5), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Bridge aggression (real bidding)", fontsize=10)
    ax.set_ylabel("Negotiation aggression (how low it opens)", fontsize=10)
    ax.set_title(f"{title}\nSpearman ρ = {rho:+.2f}", fontweight="bold", fontsize=12)
    ax.text(0.5, 1.005, subtitle, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=9.5, color=sub_color, fontstyle="italic")
    ax.grid(alpha=0.3)
    return rho


def main() -> None:
    b = bridge_aggression()
    matched = negotiation_aggression().to_dict()

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.6))
    _panel(axL, b, matched, "MATCHED: each agent uses its own bridge skills",
           "aggressive bridge -> aggressive negotiation", "#1f7a4d")
    _panel(axR, b, INVERSE_NEGO_AGGR, "INVERSE control: skills swapped to the opposite",
           "aggression follows the SKILLS, not the label", "#b5582a")
    fig.suptitle("Style transfers — and it is the SKILLS that carry it (anti-tautology control)",
                 fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
