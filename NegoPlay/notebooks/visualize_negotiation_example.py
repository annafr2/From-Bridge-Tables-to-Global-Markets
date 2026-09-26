"""
notebooks/visualize_negotiation_example.py
==========================================
A concrete picture of ONE negotiation: the five style-agents (the BUYER) bargain
against the same fixed seller for a SaaS startup. Real run-0 offers from
results/stage4/negotiation_simulations.jsonl. No LLM cost.

Scenario: "Acquire a SaaS startup": seller opens $13M, fair value $9M,
seller floor $8M (it never sells below that). Each agent is the buyer and pushes
the price down. Aggressive styles anchor near the floor and hold; cautious styles
give up and drift toward the seller.

Output: docs/images/negotiation_example.png
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROUNDS = [1, 2, 3, 4]
# Real buyer offers (run 0), SaaS startup scenario.
SERIES = {
    "Slam Hunter":      ([8.5, 8.25, 8.0, 8.0],   "#d62728"),
    "Fighter":          ([8.5, 8.0, 8.0, 8.75],   "#ff7f0e"),
    "NT Specialist":    ([8.5, 8.2, 9.25, 9.0],   "#2ca02c"),
    "Insurance Player": ([8.0, 10.75, 10.8, 10.8], "#1f77b4"),
    "Generalist":       ([8.5, 10.5, 11.0, 11.25], "#9aa0a6"),
}
OUT = "docs/images/negotiation_example.png"

# Sized for print: the paper places this at the LNCS text width (4.8 in), so
# fonts are set at their final size (>= 7 pt, Springer minimum 6 pt).
fig, ax = plt.subplots(figsize=(4.8, 3.7))

# reference price levels
_bb = dict(facecolor="white", edgecolor="none", pad=1.2)
ax.axhline(13, color="#888", ls="--", lw=1.2)
ax.text(4.15, 13, "Seller opens: $13M", va="center", fontsize=7.5, color="#555", bbox=_bb)
ax.axhline(9, color="#1f7a4d", ls=":", lw=1.4)
ax.text(4.15, 9, "Fair value: $9M", va="center", fontsize=7.5, color="#1f7a4d", bbox=_bb)
ax.axhline(8, color="#c0392b", ls="--", lw=1.2)
ax.text(4.15, 8, "Floor: $8M (never lower)", va="center", fontsize=7.5, color="#c0392b", bbox=_bb)

for name, (vals, c) in SERIES.items():
    ax.plot(ROUNDS, vals, "-o", color=c, lw=1.8, ms=4.5, label=name, zorder=3)

# group annotations
ax.annotate("Aggressive: anchor low and hold", xy=(2.5, 8.05), xytext=(2.0, 8.55),
            fontsize=7.5, color="#7a3b3b", fontstyle="italic")
ax.annotate("Cautious: give up, drift up", xy=(3, 11.0), xytext=(1.7, 11.6),
            fontsize=7.5, color="#234e70", fontstyle="italic")

ax.set_xticks(ROUNDS)
ax.tick_params(labelsize=8)
ax.set_xlabel("Negotiation round (the buyer's offer)", fontsize=8.5)
ax.set_ylabel("Price the buyer offers ($M)", fontsize=8.5)
ax.set_ylim(7.5, 14.9)          # headroom for the legend above the $13M line
ax.set_xlim(0.85, 5.9)
ax.set_title("How the five styles negotiate: buying a SaaS startup",
             fontweight="bold", color="#16324F", fontsize=9.5, pad=18)
ax.text(0.5, 1.015,
        "Each style is the buyer; the seller is the same fixed, rule-based counterpart.",
        transform=ax.transAxes, ha="center", fontsize=7, color="#555", fontstyle="italic")
ax.legend(loc="upper center", ncol=3, fontsize=7, framealpha=0.95)
ax.grid(alpha=0.22)
fig.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print("saved", OUT)
