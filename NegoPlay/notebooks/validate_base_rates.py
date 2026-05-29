"""
notebooks/validate_base_rates.py
================================
Empirical validation of Stage 2 skill claims — NO LLM CALLS.

For each profile we sampled (Slam Hunter / Fighter / Insurance / NT Specialist),
this script computes the EMPIRICAL rate at which the profile's defining
behaviour appears in those players' actual bidding strings.

If the LLM is correct, the rates should differ markedly from a Generalist
baseline. If a profile's rate is identical to Generalist, the LLM may have
hallucinated.

Output: a table you can paste into your thesis log.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

# ── Configuration ────────────────────────────────────────────────────────────

DATA = (
    r"C:\Users\annaf\OneDrive\Desktop\דוקטורט\דוקטורט אנה"
    r"\דוקטורט - שילוב בינה מלאכותית\collectBridgeData"
    r"\data\processed\all_matches_full.csv"
)
PROFILES_CSV = "data/processed/player_profiles.csv"

# Same 5-per-profile sample as the Stage 2 run (seed 42)
NON_GENERALIST_PROFILES = ["Slam Hunter", "Insurance Player", "Fighter", "NT Specialist"]
PLAYERS_PER_PROFILE = 5
SEED = 42


# ── Helpers: extract behaviours from a bidding string ────────────────────────
#
# Bidding format example:  "W:- N:1NT E:Pass S:2C | W:Pass N:2D E:Pass S:3NT..."
# Each token is "POS:CALL" separated by " | "
# Calls: "Pass", "X" (double), "XX" (redouble), or like "1H", "3NT", "6S"
#
# IMPORTANT: a player's calls are identified by the position (N/S/E/W). The
# CSV has open_north/closed_north/etc to map player names to position per row.

_BID_TOKEN_RE = re.compile(r"([NSEW]):\s*([0-9][CDHSN]T?|Pass|X|XX|-)", re.IGNORECASE)


def _parse_bidding(bidding: str) -> list[tuple[str, str]]:
    """Return list of (position, call) for every actual call (excludes '-')."""
    if not isinstance(bidding, str) or not bidding.strip():
        return []
    out = []
    for pos, call in _BID_TOKEN_RE.findall(bidding):
        if call == "-":
            continue
        out.append((pos.upper(), call.upper()))
    return out


def _player_positions_in_row(row: pd.Series, player_name: str) -> set[str]:
    """Return which of {N,S,E,W} corresponds to this player in this row.

    Bridge boards are played in two rooms (open + closed), so a player can
    appear in either room. We check all 8 columns.
    """
    positions = set()
    for room in ("open", "closed"):
        for pos in ("north", "south", "east", "west"):
            col = f"{room}_{pos}"
            if col in row and row[col] == player_name:
                positions.add(pos[0].upper())
    return positions


def _contract_level(contract: str) -> int | None:
    """1H → 1, 6NT → 6, Pass → None."""
    if not isinstance(contract, str):
        return None
    m = re.match(r"^([1-7])", contract.strip())
    return int(m.group(1)) if m else None


def _contract_denom(contract: str) -> str | None:
    """1H → 'H', 3NT → 'NT', else None."""
    if not isinstance(contract, str):
        return None
    m = re.search(r"[1-7]([CDHSN]T?)", contract.strip(), re.IGNORECASE)
    return m.group(1).upper() if m else None


# ── Compute behaviour rates for one player ───────────────────────────────────

_POSITION_MAP = {"N": "north", "S": "south", "E": "east", "W": "west"}


def _get_declarer_name_for_row(row: pd.Series) -> str | None:
    """Map (room, declarer-position) -> the actual player name. Same logic as Stage 1."""
    pos = row.get("declarer")
    room = row.get("room")
    if not isinstance(pos, str) or pos.strip() in ("-", ""):
        return None
    if not isinstance(room, str):
        return None
    suffix = _POSITION_MAP.get(pos.upper().strip())
    if suffix is None:
        return None
    prefix = "open" if room.strip().lower() == "open" else "closed"
    name = row.get(f"{prefix}_{suffix}")
    if not isinstance(name, str) or not name.strip():
        return None
    return name.strip()


def player_stats(df: pd.DataFrame, player_name: str) -> dict[str, float]:
    """Return per-behaviour rates for one player using SAME logic as Stage 1.

    Three independent denominators (matching Stage 1 features.py + bidding_parser.py):

    1. CONTRACT-LEVEL features (slam_rate, partscore_rate, nt_rate):
       denominator = boards where this player was the DECLARER.

    2. PENALTY_DOUBLE_RATE (per-board, per-player):
       denominator = boards-with-bidding where the player sat in any seat.
       Matches Stage 1 bidding_parser.player_bidding_features (made_double mean).

    3. (deprecated) double_rate_per_call — kept for reference but not used
       in profile validation since Stage 1 uses per-board, not per-call.
    """

    # ── Denominator A: boards where this player was DECLARER ──
    if "_declarer_name" not in df.columns:
        df = df.copy()
        df["_declarer_name"] = df.apply(_get_declarer_name_for_row, axis=1)

    declared = df[df["_declarer_name"] == player_name]
    n_declared = len(declared)

    levels = declared["contract"].apply(_contract_level)
    denoms = declared["contract"].apply(_contract_denom)
    slam_rate      = (levels >= 6).sum() / n_declared if n_declared else 0.0
    partscore_rate = ((levels >= 1) & (levels <= 3)).sum() / n_declared if n_declared else 0.0
    nt_rate        = (denoms == "NT").sum() / n_declared if n_declared else 0.0

    # ── Denominator B: boards with bidding where this player participated ──
    mask = False
    for col in [
        "open_north", "open_south", "open_east", "open_west",
        "closed_north", "closed_south", "closed_east", "closed_west",
    ]:
        if col in df.columns:
            mask = mask | (df[col] == player_name)
    participated = df[mask]

    # Of those, keep only boards with valid bidding strings
    n_bidding_boards_made_double = 0
    n_bidding_boards = 0
    n_doubles_per_call = 0
    n_calls = 0
    for _, row in participated.iterrows():
        positions = _player_positions_in_row(row, player_name)
        if not positions:
            continue
        bidding = row.get("bidding", "")
        if not isinstance(bidding, str) or not bidding.strip():
            continue

        # Walk THIS player's calls in THIS board
        player_made_double_on_this_board = False
        for pos, call in _parse_bidding(bidding):
            if pos not in positions:
                continue
            n_calls += 1
            if call == "X":
                n_doubles_per_call += 1
                player_made_double_on_this_board = True

        # Count this board if the player had any seat in it (they always do here)
        # We count per (board, player) — same as Stage 1's bidding_parser groupby
        for _ in positions:
            n_bidding_boards += 1
            if player_made_double_on_this_board:
                n_bidding_boards_made_double += 1

    return {
        "n_declared": n_declared,
        "n_bidding_boards": n_bidding_boards,
        "n_calls": n_calls,
        "slam_rate": round(slam_rate, 4),
        "partscore_rate": round(partscore_rate, 4),
        "nt_rate": round(nt_rate, 4),
        # The CORRECT metric for Fighter, matching Stage 1:
        "penalty_double_rate": round(
            n_bidding_boards_made_double / max(n_bidding_boards, 1), 4
        ),
        # Kept for diagnostic comparison only:
        "double_rate_per_call": round(n_doubles_per_call / max(n_calls, 1), 4),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading data...")
    df = pd.read_csv(DATA, encoding="utf-8-sig", low_memory=False)
    profiles_df = pd.read_csv(PROFILES_CSV, encoding="utf-8-sig")
    print(f"  Loaded {len(df):,} board-rows, {len(profiles_df)} players")
    print()

    # Same sample as Stage 2
    parts = []
    for prof in NON_GENERALIST_PROFILES:
        sub = profiles_df[profiles_df["profile"] == prof]
        if len(sub) > 0:
            parts.append(sub.sample(min(len(sub), PLAYERS_PER_PROFILE), random_state=SEED))

    # Add 5 Generalists as baseline
    gen = profiles_df[profiles_df["profile"] == "Generalist"]
    if len(gen) > 0:
        parts.append(gen.sample(min(len(gen), PLAYERS_PER_PROFILE), random_state=SEED))

    sample = pd.concat(parts, ignore_index=True)

    # Compute stats per player
    print("Computing per-player stats...")
    rows = []
    for _, r in sample.iterrows():
        s = player_stats(df, r["player_name"])
        rows.append({"player": r["player_name"], "profile": r["profile"], **s})
    stats = pd.DataFrame(rows)

    # ── Save to Excel for easy manual inspection ──
    out_xlsx = Path("results/stage2_sample_v2_focused_prompt/validation_table.xlsx")
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    # Sort: profile first, so all members of same profile are next to each other
    stats_sorted = stats.sort_values(["profile", "player"]).reset_index(drop=True)
    stats_sorted.to_excel(out_xlsx, index=False)
    print(f"\n[Saved table to: {out_xlsx}]\n")

    # ── Print: per-player ──
    print("=" * 90)
    print("PER-PLAYER BEHAVIOUR RATES")
    print("=" * 90)
    print(stats_sorted.to_string(index=False))

    # ── Print: aggregated per profile ──
    print()
    print("=" * 90)
    print("PER-PROFILE AVERAGES (mean across the 5 sampled players)")
    print("=" * 90)
    agg = stats.groupby("profile")[
        ["penalty_double_rate", "slam_rate", "partscore_rate", "nt_rate"]
    ].mean().round(4)
    # Force the row order so Generalist is the baseline at top
    order = ["Generalist"] + [p for p in NON_GENERALIST_PROFILES if p in agg.index]
    agg = agg.reindex([p for p in order if p in agg.index])
    print(agg.to_string())

    # ── Sanity check vs LLM claims ──
    print()
    print("=" * 90)
    print("VALIDATION CHECK — does the LLM claim match the empirical rate?")
    print("=" * 90)

    if "Generalist" in agg.index:
        baseline = agg.loc["Generalist"]
        # Add statistical tests
        try:
            from scipy.stats import mannwhitneyu
            scipy_ok = True
        except ImportError:
            scipy_ok = False

        gen_stats = stats[stats["profile"] == "Generalist"]

        print(f"  {'Profile':<18s}  {'Metric':<25s}  "
              f"{'Profile':<8s} {'Gener':<8s} {'Ratio':<8s} {'Cohen-d':<8s} {'p-val':<8s} {'Verdict'}")
        print(f"  {'-'*18}  {'-'*25}  {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*15}")

        for prof, claim_col, claim_label in [
            ("Fighter",          "penalty_double_rate", "Aggressive Penalty Doubling"),
            ("Insurance Player", "partscore_rate",      "Prioritizes Partscore Safety"),
            ("Slam Hunter",      "slam_rate",           "Slam-Seeking"),
            ("NT Specialist",    "nt_rate",             "Prioritizes NT"),
        ]:
            if prof not in agg.index:
                continue
            prof_val = agg.loc[prof, claim_col]
            base_val = baseline[claim_col]
            ratio = prof_val / base_val if base_val > 0 else float("inf")

            # Cohen's d (pooled SD)
            prof_vals = stats[stats["profile"] == prof][claim_col].values
            gen_vals = gen_stats[claim_col].values
            mean_diff = prof_vals.mean() - gen_vals.mean()
            pooled_sd = (((prof_vals.std(ddof=1) ** 2 + gen_vals.std(ddof=1) ** 2) / 2) ** 0.5)
            cohen_d = mean_diff / pooled_sd if pooled_sd > 0 else float("nan")

            # Mann-Whitney U (one-sided: profile > generalist)
            if scipy_ok and len(prof_vals) >= 3 and len(gen_vals) >= 3:
                _, p_value = mannwhitneyu(prof_vals, gen_vals, alternative="greater")
            else:
                p_value = float("nan")

            # Bridge-expert calibrated verdict: combine ratio + effect size + p-value
            if cohen_d >= 1.2 and p_value < 0.05:
                verdict = "[STRONG]"
            elif cohen_d >= 0.8 and p_value < 0.10:
                verdict = "[CONFIRMED]"
            elif cohen_d >= 0.5:
                verdict = "[WEAK]"
            else:
                verdict = "[FALSIFIED]"

            print(f"  {prof:<18s}  {claim_label:<25s}  "
                  f"{prof_val:<8.3f} {base_val:<8.3f} x{ratio:<7.2f} "
                  f"{cohen_d:<8.2f} {p_value:<8.4f} {verdict}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
