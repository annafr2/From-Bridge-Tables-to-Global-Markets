"""
notebooks/rederive_bridge_winrates.py
=====================================
Re-derive Stage 4a bridge win rates from the raw JSONL log — NO LLM calls.

Reproducibility safeguard: the win-rate CSV is regenerated purely from the
persisted per-bid records, so even if the live run's CSV was corrupted (e.g.
overlapping processes appending duplicate rows), the analysis is recomputed
cleanly from raw data. Also validates JSONL integrity.

Reads:   results/stage4/bridge_simulations.jsonl
Writes:  results/stage4/bridge_winrates.csv   (clean, deduplicated)
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.stage4_simulate.bridge_game import score_bid

JSONL = Path("results/stage4/bridge_simulations.jsonl")
CSV_OUT = Path("results/stage4/bridge_winrates.csv")


def main() -> None:
    rows = []
    bad = 0
    for line in JSONL.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            bad += 1
            continue
        # must look like a real bid record
        if {"profile", "board", "run", "bid", "partnership_hcp"} <= r.keys():
            rows.append(r)
        else:
            bad += 1

    print(f"Parsed {len(rows)} valid bid records ({bad} skipped as malformed).")

    # Deduplicate on (profile, board, run): keep first occurrence.
    seen: set[tuple] = set()
    deduped: list[dict] = []
    for r in rows:
        key = (r["profile"], r["board"], r["run"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    if len(deduped) != len(rows):
        print(f"Removed {len(rows) - len(deduped)} duplicate (profile,board,run) rows.")

    # Group by (profile, board) -> majority bid -> score.
    by_pb: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in deduped:
        by_pb[(r["profile"], r["board"])].append(r)

    prof_scores: dict[str, list[float]] = defaultdict(list)
    for (prof, _board), recs in by_pb.items():
        hcp = recs[0]["partnership_hcp"]
        bids = [x["bid"] for x in recs]
        cnt = Counter(bids)
        best = cnt.most_common(1)[0][1]
        tied = [b for b, n in cnt.items() if n == best]
        chosen = max(tied, key=lambda b: score_bid(b, hcp).score)
        prof_scores[prof].append(score_bid(chosen, hcp).score)

    n_boards = len({b for _p, b in by_pb})
    winrates = {p: round(sum(s) / len(s), 4) for p, s in prof_scores.items()}

    with CSV_OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["profile", "bridge_winrate", "n_boards"])
        for p, rate in sorted(winrates.items(), key=lambda x: -x[1]):
            w.writerow([p, rate, n_boards])

    print(f"\n=== BRIDGE WIN RATES (re-derived, {n_boards} boards) ===")
    for p, rate in sorted(winrates.items(), key=lambda x: -x[1]):
        n = len(prof_scores[p])
        print(f"  {p:18s} {rate:.3f}  (scored on {n} boards)")
    print(f"\nWrote clean {CSV_OUT}")


if __name__ == "__main__":
    main()
