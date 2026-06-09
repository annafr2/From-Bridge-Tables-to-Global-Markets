"""
notebooks/style_alignment.py
============================
STYLE -> STYLE cross-domain test (the behavioural framing).

Instead of correlating *winning* in bridge with *winning* in negotiation (noisy,
outcome-based), we correlate the **behavioural STYLE** directly:

  - Bridge aggression  = how aggressively the profile's real players bid
      (composite z-score of slam_rate + preempt_rate + penalty_double_rate).
  - Negotiation aggression = how aggressively the profile's AGENT bargains
      (how low it opens/offers, from the saved negotiation logs — no LLM cost).

The thesis premise is that the *risk style* transfers: an aggressive bridge
profile should bargain aggressively. This is a cleaner, lower-noise test than the
outcome correlation, and it is exactly the original "behavioural alignment" RQ.

Output: docs/images/style_alignment.png + console.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr

from src.stage4_simulate.negotiation import SCENARIOS

PROFILES_CSV = Path("data/processed/player_profiles.csv")
NEGO_JSONL = Path("results/stage4/negotiation_simulations.jsonl")
OUT = Path("docs/images/style_alignment.png")

# Core "risk / aggression" bridge markers (bidding big, opening high, fighting).
AGGR_FEATURES = ["slam_rate", "preempt_rate", "penalty_double_rate"]
COLORS = {"Slam Hunter": "#d62728", "Insurance Player": "#1f77b4",
          "Fighter": "#ff7f0e", "NT Specialist": "#2ca02c", "Generalist": "#9aa0a6"}


def bridge_aggression() -> pd.Series:
    """Composite bridge-aggression per profile (mean z-score of aggressive features)."""
    pf = pd.read_csv(PROFILES_CSV)
    means = pf.groupby("profile")[AGGR_FEATURES].mean()
    z = (means - means.mean()) / means.std(ddof=0)   # z-score each feature across profiles
    return z.mean(axis=1)                              # average -> one aggression score


def negotiation_aggression() -> pd.Series:
    """How aggressively each profile's agent bargains: mean 'demand depth' of offers.

    For each counter-offer, depth = (seller_open - offer) / (seller_open - floor):
    0 = asked for the opening price, 1 = pushed all the way to the floor. Higher =
    more aggressive bargaining. Averaged over all the profile's offers.
    """
    spans = {s["title"]: (s["seller_open"], s["seller_floor"]) for s in SCENARIOS}
    rows = [json.loads(line) for line in NEGO_JSONL.open(encoding="utf-8")]
    depth: dict[str, list[float]] = {}
    for r in rows:
        if r.get("action") != "counter" or not r.get("offer"):
            continue
        price = r["offer"].get("price_musd")
        if price is None or r["scenario"] not in spans:
            continue
        open_, floor = spans[r["scenario"]]
        if open_ <= floor:
            continue
        d = (open_ - price) / (open_ - floor)
        depth.setdefault(r["profile"], []).append(max(0.0, min(1.3, d)))
    return pd.Series({p: sum(v) / len(v) for p, v in depth.items()})


def main() -> None:
    b = bridge_aggression()
    n = negotiation_aggression()
    profiles = [p for p in b.index if p in n.index]
    df = pd.DataFrame({
        "profile": profiles,
        "bridge_aggression": [b[p] for p in profiles],
        "nego_aggression": [n[p] for p in profiles],
    }).sort_values("bridge_aggression", ascending=False).reset_index(drop=True)
    rho, p = spearmanr(df["bridge_aggression"], df["nego_aggression"])

    fig, ax = plt.subplots(figsize=(9, 7))
    for _, r in df.iterrows():
        ax.scatter(r["bridge_aggression"], r["nego_aggression"], s=210,
                   color=COLORS.get(r["profile"], "#333"), edgecolors="black", zorder=3)
        ax.annotate(r["profile"], (r["bridge_aggression"], r["nego_aggression"]),
                    xytext=(8, 5), textcoords="offset points", fontsize=10)
    ax.set_xlabel("Bridge aggression  (real bidding: slam + preempt + doubles, z-score)", fontsize=11)
    ax.set_ylabel("Negotiation aggression  (how low the agent opens/offers)", fontsize=11)
    ax.set_title(f"STYLE -> STYLE: does aggression transfer?\nSpearman ρ = {rho:+.2f}",
                 fontweight="bold", fontsize=13)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")

    print(df.round(3).to_string(index=False))
    print(f"\nStyle->style Spearman rho = {rho:+.3f} (p = {p:.3f}, n = {len(df)})")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
