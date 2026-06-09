"""
notebooks/visualize_real_skill_spectrum.py
==========================================
The REAL-DATA skill spectrum: random MONKEY (floor) -> real elite profiles ->
DOUBLE-DUMMY perfect play (ceiling), all measured on the REAL EuroBridge boards.

Everything is on ONE axis: average IMP versus the board datum (the mean of the
two real tables). So a player/agent at 0 is exactly average; positive = better
than the field on identical cards.

  - Monkey: a random legal contract, scored double-dummy from the real 52 cards.
  - Real profiles: how the actual players in each profile did (the same IMP-datum
    measure that gives the +0.50 cross-domain correlation).
  - Double-dummy perfect: the par contract from the real cards (the ceiling).

Result: real elite players sit FAR above the random monkey but still short of
perfect, and among them the aggressive profiles (Slam Hunter, Fighter) lead —
consistent with the real-data finding. No LLM cost.

Output: docs/images/real_skill_spectrum.png
"""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.features.double_dummy import _ALL_STRAINS, _best_ns_points, _contract_dd_points

# Reuse the validated profile IMP values from the alignment builder.
import sys
sys.path.insert(0, "notebooks")
from alignment_real_bridge import build as build_alignment  # noqa: E402

_MASTER = None
for _p in [Path("../data/processed/all_matches.parquet"),
           Path("data/processed/all_matches.parquet"),
           Path(__file__).resolve().parents[2] / "data" / "processed" / "all_matches.parquet"]:
    if _p.exists():
        _MASTER = _p
        break

OUT = Path("docs/images/real_skill_spectrum.png")
N_SAMPLE = 2000
COLORS = {
    "Slam Hunter": "#d62728", "Insurance Player": "#1f77b4", "Fighter": "#ff7f0e",
    "NT Specialist": "#2ca02c", "Generalist": "#9aa0a6",
}
_IMP_BREAKS = [(20, 0), (50, 1), (90, 2), (130, 3), (170, 4), (220, 5), (270, 6),
               (320, 7), (370, 8), (430, 9), (500, 10), (600, 11), (750, 12),
               (900, 13), (1100, 14), (1300, 15), (1500, 16), (1750, 17),
               (2000, 18), (2250, 19), (2500, 20), (3000, 21), (3500, 22), (4000, 23)]


def _to_imp(diff: float) -> int:
    a, sign = abs(diff), (1 if diff >= 0 else -1)
    for lo, pts in _IMP_BREAKS:
        if a < lo:
            return sign * pts
    return sign * 24


def _suit(x) -> str:
    return "".join("T" if r == "10" else r for r in str(x).split())


def _hands(row) -> dict:
    m = {"north": "N", "south": "S", "east": "E", "west": "W"}
    return {m[p]: {"S": _suit(row[f"{p}_spades"]), "H": _suit(row[f"{p}_hearts"]),
                   "D": _suit(row[f"{p}_diamonds"]), "C": _suit(row[f"{p}_clubs"])}
            for p in ["north", "south", "east", "west"]}


def compute_anchors() -> dict:
    """Monkey floor + double-dummy ceiling, in IMP-vs-datum, on real boards."""
    df = pd.read_parquet(_MASTER)
    cardcols = [f"{p}_{s}" for p in ["north", "south", "east", "west"]
                for s in ["spades", "hearts", "diamonds", "clubs"]]
    d = df[df[cardcols].notna().all(axis=1)]
    g = d.groupby(["match_id", "board"])
    keys = list(g.groups.keys())
    random.Random(42).shuffle(keys)
    rng = random.Random(1)

    monkey, perfect, used = [], [], 0
    for k in keys:
        if used >= N_SAMPLE:
            break
        rows = g.get_group(k)
        ns = pd.to_numeric(rows["ns_score"], errors="coerce").dropna()
        if len(ns) < 2:
            continue
        datum = ns.mean()
        try:
            h = _hands(rows.iloc[0])
            best = _best_ns_points(h, False)
            lvl, strain = rng.randint(1, 7), rng.choice(_ALL_STRAINS)
            mk = _contract_dd_points(lvl, strain, h, False)
        except Exception:
            continue
        perfect.append(_to_imp(best - datum))
        monkey.append(_to_imp(mk - datum))
        used += 1
    return {"monkey": sum(monkey) / len(monkey),
            "perfect": sum(perfect) / len(perfect), "n": used}


def make_figure() -> None:
    anchors = compute_anchors()
    res = build_alignment()
    prof = res["df"][["profile", "bridge_imp"]].sort_values("bridge_imp")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 5.2),
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    # ── Panel A: the full spectrum, monkey -> players -> perfect ──────────────
    mk, pf = anchors["monkey"], anchors["perfect"]
    axL.axvspan(mk - 0.6, mk + 0.6, color="#f6d5d2", alpha=0.5)
    axL.scatter([mk], [0], s=300, color="#c0392b", edgecolors="black", zorder=3)
    axL.text(mk, 0.5, "random monkey\n(random contract)", ha="center", fontsize=10,
             color="#c0392b", fontweight="bold")
    axL.axvline(pf, color="#1f7a4d", ls="--", lw=2)
    axL.text(pf, 0.5, "double-dummy\nperfect play", ha="center", fontsize=10,
             color="#1f7a4d", fontweight="bold")
    axL.scatter([0], [0], s=120, color="#444", zorder=3)
    axL.text(0, -0.6, "real elite players\n(all 5 profiles, ≈ average)", ha="center",
             fontsize=9.5)
    axL.annotate("", xy=(0.25, 0), xytext=(0.05, 0),
                 arrowprops=dict(arrowstyle="-[,widthB=0.6", color="#444"))
    axL.axhline(0, color="#bbb", lw=1, zorder=1)
    axL.set_ylim(-1.4, 1.4)
    axL.set_yticks([])
    axL.set_xlabel("Bridge skill: average IMP vs the field (real boards)", fontsize=11)
    axL.set_title("Real skill spectrum: random → elite → perfect",
                  fontweight="bold", fontsize=12)
    axL.spines[["top", "right", "left"]].set_visible(False)
    axL.grid(axis="x", alpha=0.3)

    # ── Panel B: zoom on the real profiles (they cluster near 0) ──────────────
    for i, (_, r) in enumerate(prof.iterrows()):
        c = COLORS.get(r["profile"], "#444")
        axR.plot([0, r["bridge_imp"]], [i, i], color=c, lw=2.5, zorder=2)
        axR.scatter(r["bridge_imp"], i, s=220, color=c, edgecolors="black", zorder=3)
        axR.text(r["bridge_imp"] + 0.004, i, f"{r['profile']}  ({r['bridge_imp']:+.2f})",
                 va="center", fontsize=10)
    axR.axvline(0, color="#999", ls=":", lw=1.2)
    axR.text(0, -0.85, "average pair", ha="center", fontsize=9, color="#777")
    axR.set_yticks([])
    axR.set_xlim(-0.05, 0.25)
    axR.set_ylim(-1.3, len(prof) - 0.4)
    axR.set_xlabel("average IMP vs the field (zoom on the 5 profiles)", fontsize=11)
    axR.set_title("Among the elite: aggressive profiles lead",
                  fontweight="bold", fontsize=12)
    axR.spines[["top", "right", "left"]].set_visible(False)
    axR.grid(axis="x", alpha=0.3)

    fig.suptitle("Real-data skill spectrum — measured on the actual EuroBridge boards",
                 fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"saved {OUT}")
    print(f"  monkey {anchors['monkey']:+.2f} IMP | perfect {anchors['perfect']:+.2f} IMP "
          f"| n={anchors['n']} boards")
    for _, r in prof.iterrows():
        print(f"  {r['profile']:18s} {r['bridge_imp']:+.3f} IMP")


if __name__ == "__main__":
    make_figure()
