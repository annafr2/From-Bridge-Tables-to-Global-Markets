"""
notebooks/visualize_skill_spectrum.py
=====================================
The "floor -> ceiling" skill spectrum: where each agent sits between a random
MONKEY (0%, floor) and DOUBLE-DUMMY perfect play (100%, ceiling).

Both anchors come from the full-auction bridge run (bridge_auction_winrates.csv),
where the double-dummy score is already on a 0-1 scale (1.0 = the optimal
contract). So:
    monkey  ~ 0   (random, no skill)
    profiles  in between
    double-dummy = 1.0  (perfect play, the ceiling)

This is the single picture that shows skill beating random AND how far real
profiles still are from perfect. No LLM cost (reads saved results).

Output: docs/images/skill_spectrum.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

CSV = Path("results/stage4/bridge_auction_winrates.csv")
OUT = Path("docs/images/skill_spectrum.png")
MONKEY = "Monkey (random)"

COLORS = {
    "Slam Hunter": "#d62728", "Insurance Player": "#1f77b4", "Fighter": "#ff7f0e",
    "NT Specialist": "#2ca02c", "Generalist": "#9aa0a6", MONKEY: "#c0392b",
}


def make_figure() -> None:
    df = pd.read_csv(CSV).sort_values("bridge_dd_winrate")
    names = df["profile"].tolist()
    vals = df["bridge_dd_winrate"].tolist()

    fig, ax = plt.subplots(figsize=(11, 5.5))
    y = range(len(names))

    # Floor (0) and ceiling (1.0 = double-dummy perfect) reference bands.
    ax.axvspan(0, 0.06, color="#f6d5d2", alpha=0.6)
    ax.axvline(1.0, color="#1f7a4d", ls="--", lw=2)
    ax.text(1.0, len(names) - 0.3, "  double-dummy\n  perfect play (100%)",
            color="#1f7a4d", fontsize=10, va="top", fontweight="bold")
    ax.text(0.0, -0.9, "monkey floor (0%)", color="#c0392b", fontsize=9.5)

    for i, (n, v) in enumerate(zip(names, vals)):
        c = COLORS.get(n, "#444")
        ax.plot([0, v], [i, i], color=c, lw=2.5, zorder=2)          # stem
        ax.scatter(v, i, s=240, color=c, edgecolors="black", zorder=3)
        label = "🐵 " + n if n == MONKEY else n
        ax.text(v + 0.012, i, f"{label}  ({v*100:.0f}%)", va="center", fontsize=10,
                fontweight="bold" if n == MONKEY else "normal")

    ax.set_yticks([])
    ax.set_xlim(-0.02, 1.12)
    ax.set_ylim(-1.4, len(names) - 0.2)
    ax.set_xlabel("Bridge skill: 0 = random monkey  →  1.0 = double-dummy perfect play",
                  fontsize=11)
    ax.set_title("Skill spectrum — every agent between the random floor and the "
                 "perfect-play ceiling\n(full-auction bridge, double-dummy scored)",
                 fontweight="bold", fontsize=12.5)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"saved {OUT}")
    for n, v in zip(names, vals):
        print(f"  {n:18s} {v*100:5.1f}%")


if __name__ == "__main__":
    make_figure()
