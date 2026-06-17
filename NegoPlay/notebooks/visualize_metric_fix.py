"""
notebooks/visualize_metric_fix.py
=================================
Visualize the metric fix (Tasks T-A + T-C): a random "monkey" baseline exposes
that the OLD par-proxy bridge metric was broken (monkey beats the experts), and
the NEW double-dummy metric fixes it (monkey is last, skill wins).

Everything is recomputed from saved data (bridge_simulations.jsonl) + a fresh
seeded monkey run — no LLM calls, no cost. Two panels, same profiles:
  LEFT  — old par-proxy metric  -> monkey on top (BROKEN)
  RIGHT — new double-dummy metric -> monkey at bottom (FIXED)

Output: docs/images/metric_fix_monkey_dd.png
"""

from __future__ import annotations

import collections
import json
import statistics as st
from pathlib import Path

import matplotlib.pyplot as plt

from src.features.double_dummy import dd_bid_score01
from src.stage4_simulate.bridge_game import deal_board, score_bid
from src.stage4_simulate.monkey_agent import MonkeyAgent

N_BOARDS = 50
SEED = 42
PARTNER_HCP = [("1C", 13), ("1NT", 16), ("2C", 22)]   # cycles by board (as in runner)
JSONL = Path("results/stage4/bridge_simulations.jsonl")
OUT = Path("docs/images/metric_fix_monkey_dd.png")

MONKEY = "ZI-C (random)"


def _partnership_hcp(board: int) -> int:
    deal = deal_board(board, seed=SEED)
    _, phcp = PARTNER_HCP[(board - 1) % 3]
    return deal.hcp("S") + phcp


def profile_scores() -> dict[str, dict[str, float]]:
    """Old (par-proxy) and new (double-dummy) mean score per profile, from saved bids."""
    rows = [json.loads(line) for line in JSONL.open(encoding="utf-8")]
    by_prof: dict[str, dict[int, list[str]]] = collections.defaultdict(
        lambda: collections.defaultdict(list))
    for r in rows:
        by_prof[r["profile"]][r["board"]].append(r["bid"])

    out: dict[str, dict[str, float]] = {}
    for prof, boards in by_prof.items():
        old, new = [], []
        for board, bids in boards.items():
            majority = collections.Counter(bids).most_common(1)[0][0]
            deal = deal_board(int(board), seed=SEED)
            old.append(score_bid(majority, _partnership_hcp(int(board))).score)
            new.append(dd_bid_score01(majority, deal.hands))
        out[prof] = {"old": st.mean(old), "new": st.mean(new)}
    return out


def monkey_scores() -> dict[str, float]:
    monkey = MonkeyAgent(seed=SEED)
    old, new = [], []
    for board in range(1, N_BOARDS + 1):
        deal = deal_board(board, seed=SEED)
        partner_bid = PARTNER_HCP[(board - 1) % 3][0]
        bid = monkey.make_bid(deal.hands["S"], [partner_bid])["bid"]
        old.append(score_bid(bid, _partnership_hcp(board)).score)
        new.append(dd_bid_score01(bid, deal.hands))
    return {"old": st.mean(old), "new": st.mean(new)}


def make_figure() -> None:
    profs = profile_scores()
    monkey = monkey_scores()

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    panels = [
        ("old", "OLD metric: par-level proxy (4 coarse classes)",
         "The ZI-C agent BEATS the experts → metric is broken", axes[0]),
        ("new", "NEW metric: double-dummy (true perfect-play result)",
         "The ZI-C agent is LAST → skill wins, as it must", axes[1]),
    ]

    for key, title, subtitle, ax in panels:
        items = [(p, s[key]) for p, s in profs.items()]
        items.append((MONKEY, monkey[key]))
        items.sort(key=lambda x: x[1], reverse=True)
        names = [n for n, _ in items]
        vals = [v for _, v in items]
        colors = ["#c0392b" if n == MONKEY else "#2c6fb0" for n in names]

        bars = ax.barh(range(len(names)), vals, color=colors, edgecolor="#16324f")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("bridge score (0–1)")
        ax.set_title(title, fontsize=11, fontweight="bold", color="#16324f")
        ax.text(0.5, 1.045, subtitle, transform=ax.transAxes, ha="center",
                fontsize=10, color="#7a3b3b" if key == "old" else "#1f7a4d",
                fontstyle="italic")
        for b, v in zip(bars, vals):
            ax.text(v + 0.015, b.get_y() + b.get_height() / 2, f"{v:.2f}",
                    va="center", fontsize=9, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle(
        "Why we added a zero-intelligence (ZI-C) baseline + double-dummy: fixing the bridge metric",
        fontsize=12, fontweight="bold", color="#16324f")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"saved {OUT}")
    print("monkey:", {k: round(v, 3) for k, v in monkey.items()})
    for p, s in profs.items():
        print(f"  {p:18s} old={s['old']:.3f}  new={s['new']:.3f}")


if __name__ == "__main__":
    make_figure()
