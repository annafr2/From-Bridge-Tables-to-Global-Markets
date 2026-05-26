"""
src/stage1_clustering/features.py
==================================
Compute per-player behavioural features from the bridge dataset.

Each row in the output represents one player (by name) with 5 features
derived from their declared contracts:

    slam_rate       — % of declarations at slam level (6 or 7)
    success_rate    — % of contracts made
    double_rate     — % of contracts that were doubled (opponent's double)
    avg_level       — average contract level (1–7), proxy for risk appetite
    risk_score      — composite risk index in [0, 10]

Features are computed only from rows where the player is the declarer.
This uses the full 149K dataset (no bidding sequence required).
"""

import logging
import re

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Minimum number of declared boards for a player to be included
MIN_BOARDS: int = 20

# Weights for the composite risk_score (must sum to 10)
RISK_WEIGHTS: dict[str, float] = {
    "slam_rate":    4.0,   # highest weight — bids to slam = clear risk
    "double_rate":  2.0,   # opponent doubled you = you bid aggressively
    "avg_level":    4.0,   # normalized: level 1→0, level 7→4
}

# Position abbreviation → column suffix
POSITION_MAP: dict[str, str] = {
    "N": "north",
    "S": "south",
    "E": "east",
    "W": "west",
}


# ── Internal helpers ────────────────────────────────────────────────────────

def _parse_contract_level(contract: str) -> int | None:
    """Extract the numeric level (1–7) from a contract string.

    Examples:
        '4H'   → 4
        '6NT*' → 6
        '7C**' → 7
        '-'    → None  (passed out)
        ''     → None
    """
    if not isinstance(contract, str) or not contract.strip():
        return None
    match = re.match(r"^(\d)", contract.strip())
    if match:
        level = int(match.group(1))
        return level if 1 <= level <= 7 else None
    return None


def _is_made(contract: str, tricks: int | float) -> bool | None:
    """Return True if the contract was made, False if down, None if unknown.

    A level-L contract requires L + 6 tricks.
    """
    level = _parse_contract_level(contract)
    if level is None or pd.isna(tricks):
        return None
    return int(tricks) >= level + 6


def _is_doubled(contract: str) -> bool:
    """Return True if the contract string contains a double marker (* or X).

    EuroBridge uses '*' for doubled and '**' for redoubled.
    """
    if not isinstance(contract, str):
        return False
    return "*" in contract or contract.upper().endswith("X")


def _get_declarer_name(row: pd.Series) -> str | None:
    """Map the declarer's compass position to their player name.

    Uses the room (Open/Closed) and position (N/S/E/W) to find the
    matching name column (e.g., open_north, closed_south).
    """
    pos = row.get("declarer")
    room = row.get("room")

    if not isinstance(pos, str) or pos.strip() in ("-", ""):
        return None
    if not isinstance(room, str):
        return None

    pos_suffix = POSITION_MAP.get(pos.upper().strip())
    if pos_suffix is None:
        return None

    prefix = "open" if room.strip().lower() == "open" else "closed"
    col = f"{prefix}_{pos_suffix}"

    name = row.get(col)
    if not isinstance(name, str) or name.strip() == "":
        return None
    return name.strip()


# ── Public API ───────────────────────────────────────────────────────────────

def compute_player_features(
    df: pd.DataFrame,
    min_boards: int = MIN_BOARDS,
) -> pd.DataFrame:
    """Compute behavioural features for every player with enough declarations.

    Args:
        df: Raw bridge dataset (output of data_loader.load_matches).
        min_boards: Minimum number of declared boards to include a player.

    Returns:
        DataFrame indexed by player name with columns:
            n_declared, slam_rate, success_rate, double_rate,
            avg_level, risk_score
    """
    logger.info("Computing player features from %d rows ...", len(df))

    # ── Step 1: Add helper columns ──────────────────────────────────────────
    work = df.copy()

    work["_level"] = work["contract"].apply(_parse_contract_level)
    work["_is_slam"] = work["_level"].apply(
        lambda lvl: lvl >= 6 if lvl is not None else None
    )
    work["_is_made"] = work.apply(
        lambda row: _is_made(row["contract"], row["tricks"]), axis=1
    )
    work["_is_doubled"] = work["contract"].apply(_is_doubled)
    work["_declarer_name"] = work.apply(_get_declarer_name, axis=1)

    # ── Step 2: Keep only rows where a player declared ──────────────────────
    declared = work.dropna(subset=["_declarer_name", "_level"]).copy()
    logger.info("Rows with identified declarer: %d", len(declared))

    # ── Step 3: Aggregate per player ────────────────────────────────────────
    agg = (
        declared
        .groupby("_declarer_name")
        .agg(
            n_declared=("_level", "count"),
            slam_rate=("_is_slam", "mean"),
            success_rate=("_is_made", "mean"),
            double_rate=("_is_doubled", "mean"),
            avg_level=("_level", "mean"),
        )
        .reset_index()
        .rename(columns={"_declarer_name": "player_name"})
    )

    # ── Step 4: Filter by minimum boards ────────────────────────────────────
    before = len(agg)
    agg = agg[agg["n_declared"] >= min_boards].copy()
    logger.info(
        "Players with >= %d boards: %d / %d",
        min_boards, len(agg), before,
    )

    # ── Step 5: Compute composite risk_score ────────────────────────────────
    # Normalize avg_level to [0, 1]: level ranges from 1 to 7
    agg["_avg_level_norm"] = (agg["avg_level"] - 1) / 6.0

    agg["risk_score"] = (
        agg["slam_rate"]       * RISK_WEIGHTS["slam_rate"]
        + agg["double_rate"]   * RISK_WEIGHTS["double_rate"]
        + agg["_avg_level_norm"] * RISK_WEIGHTS["avg_level"]
    ).clip(0, 10).round(4)

    agg = agg.drop(columns=["_avg_level_norm"])

    # Round for readability
    for col in ["slam_rate", "success_rate", "double_rate", "avg_level"]:
        agg[col] = agg[col].round(4)

    agg = agg.sort_values("n_declared", ascending=False).reset_index(drop=True)

    logger.info(
        "Feature matrix ready: %d players × %d features",
        len(agg), len(agg.columns) - 1,  # exclude player_name
    )

    return agg
