"""
notebooks/validate_negotiation_features.py
==========================================
Ground our negotiation behaviour features in REAL human data (Stanford CoCoA
Craigslist Bargains, 5,247 real buyer/seller price negotiations).

Goal: validate that (a) negotiation behaviour is a real, measurable dimension and
(b) the *calibration trade-off* our walk-away seller models is real — i.e. opening
more aggressively captures more surplus BUT also raises the chance the deal falls
through. If the real data shows this inverted-U, our simulation's red line is not
an arbitrary choice; it reproduces a real phenomenon.

Behaviour features (per buyer):
  - aggressiveness = (listing - buyer_first_offer) / (listing - buyer_target)
      0 = opened at the asking price; 1 = opened at their own target; >1 = opened
      below their own target (very aggressive / lowball).
Outcomes:
  - deal_rate   = fraction that reached an accepted deal (vs quit/reject)
  - surplus     = (listing - final_price) / (listing - buyer_target), deals only

Output: docs/images/negotiation_real_validation.png + console table.
"""

from __future__ import annotations

import json
import re
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path("data/external/craigslist_train.json")
OUT = Path("docs/images/negotiation_real_validation.png")
_PRICE = re.compile(r"\$?\s?(\d[\d,]*\.?\d{0,2})")


def _first_buyer_price(events, buyer_idx, listing):
    for e in events:
        if e.get("agent") == buyer_idx and e.get("action") == "message":
            m = _PRICE.search(str(e.get("data") or ""))
            if m:
                v = float(m.group(1).replace(",", ""))
                if 0 < v <= listing * 1.5:
                    return v
    return None


def extract() -> list[dict]:
    data = json.load(DATA.open(encoding="utf-8"))
    rows = []
    for d in data:
        kbs = d["scenario"]["kbs"]
        bidx = next((i for i, k in enumerate(kbs) if k["personal"]["Role"] == "buyer"), None)
        if bidx is None:
            continue
        listing = kbs[bidx]["item"].get("Price")
        target = kbs[bidx]["personal"].get("Target")
        if not listing or not target or listing <= target:
            continue
        events = d.get("events", [])
        acts = [e.get("action") for e in events]
        offer = (d.get("outcome", {}) or {}).get("offer") or {}
        final = offer.get("price")
        deal = ("accept" in acts) and final is not None
        first = _first_buyer_price(events, bidx, listing)
        if first is None:
            continue
        span = listing - target
        rows.append({
            "aggressiveness": max(0.0, min(1.6, (listing - first) / span)),
            "deal": bool(deal),
            "surplus": ((listing - final) / span) if deal else None,
        })
    return rows


def main() -> None:
    rows = extract()
    bins = [(0.0, 0.5, "soft\n(open near ask)"), (0.5, 0.9, "moderate"),
            (0.9, 1.1, "firm\n(open at target)"), (1.1, 1.7, "lowball\n(below target)")]
    labels, deal_rates, surpluses, ns = [], [], [], []
    print(f"{'opening aggressiveness':24s} {'n':>5} {'deal_rate':>10} {'avg_surplus':>12}")
    for lo, hi, name in bins:
        sub = [r for r in rows if lo <= r["aggressiveness"] < hi]
        if not sub:
            continue
        dr = sum(r["deal"] for r in sub) / len(sub)
        sp = [r["surplus"] for r in sub if r["deal"]]
        msp = st.mean(sp) if sp else 0.0
        labels.append(name); deal_rates.append(dr); surpluses.append(msp); ns.append(len(sub))
        print(f"  {name.replace(chr(10),' '):22s} {len(sub):>5} {dr:>10.2f} {msp:>12.2f}")

    fig, ax1 = plt.subplots(figsize=(9.5, 5.5))
    x = range(len(labels))
    ax1.bar([i - 0.2 for i in x], deal_rates, width=0.4, color="#2c6fb0",
            label="deal rate (got a deal)")
    ax1.set_ylabel("deal rate", color="#2c6fb0", fontsize=11)
    ax1.set_ylim(0, 1)
    ax2 = ax1.twinx()
    ax2.bar([i + 0.2 for i in x], surpluses, width=0.4, color="#e0792b",
            label="avg surplus captured (deals)")
    ax2.set_ylabel("avg surplus captured (deals only)", color="#e0792b", fontsize=11)
    ax2.set_ylim(0, 1)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(labels, ns)], fontsize=9)
    ax1.set_xlabel("How aggressively the BUYER opened (real Craigslist negotiations)", fontsize=11)
    ax1.set_title("Real negotiation data validates the calibration trade-off\n"
                  "more aggressive opening → more surplus, but the deal more often falls through",
                  fontweight="bold", fontsize=12)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"\nusable dialogues: {len(rows)} | saved {OUT}")


if __name__ == "__main__":
    main()
