"""
src/stage1_clustering/features.py
==================================
Compute per-player behavioural features from the bridge dataset.

Each row in the output represents one player (by name) with 8 features
derived from their declared contracts (all from the 'contract' column —
no external data needed):

    slam_rate       — % of declarations at slam level (6 or 7)
    success_rate    — % of contracts made
    double_rate     — % of contracts that were doubled (opponent's double)
    avg_level       — average contract level (1–7), proxy for risk appetite
    nt_rate         — % of contracts in NoTrump  → identifies NT specialists
    partscore_rate  — % of contracts at level 1/2/3 → identifies cautious players
    game_rate       — % of contracts at game level (3NT/4H/4S/5m exactly)
    risk_score      — composite risk index in [0, 10]

All features come directly from the 'contract' column in the 149K dataset.
No new data is needed — we only parse what is already there.
"""

import logging
import re

import numpy as np
import pandas as pd

from src.stage1_clustering.bidding_parser import player_bidding_features

logger = logging.getLogger(__name__)

# Minimum number of declared boards for a player to be included
MIN_BOARDS: int = 50  # Raised from 20 (May 2026) — see PRD Stage 1 sample-size analysis

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
    Some records use 'x' suffix instead.
    """
    if not isinstance(contract, str):
        return False
    return "*" in contract or contract.lower().endswith("x")


def _is_nt(contract: str) -> bool:
    """Return True if the contract is in NoTrump (e.g. '3NT', '6NT*').

    Identifies players who prefer NT over suit contracts.
    Data source: 'contract' column, no new data needed.
    """
    if not isinstance(contract, str):
        return False
    return "NT" in contract.upper()


def _is_partscore(contract: str) -> bool:
    """Return True if the contract is a partscore (level 1, 2, or 3).

    Identifies cautious / insurance players who stop below game.
    Data source: 'contract' column, no new data needed.
    """
    level = _parse_contract_level(contract)
    return level is not None and level <= 3


def _is_game(contract: str) -> bool:
    """Return True if the contract is at game level (not slam, not partscore).

    Game = level 3 (3NT only), 4 (majors), or 5 (minors).
    Excludes slam (6/7) and partscore (1/2 and 3 non-NT).

    Data source: 'contract' column, no new data needed.
    """
    level = _parse_contract_level(contract)
    if level is None:
        return False
    if level >= 6:   # slam — not game
        return False
    if level <= 2:   # partscore — not game
        return False
    # level 3: game only if NoTrump (3NT), otherwise partscore
    if level == 3:
        return _is_nt(contract)
    # level 4 or 5: always game
    return True


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


def compute_bidding_features(df: pd.DataFrame) -> pd.DataFrame:
    """For each player, compute aggregate bidding-PROCESS features.

    Process features (from `bidding` column, 46K rows / 31% of data):
        opening_rate        — % of boards where this player opened the auction
        preempt_rate        — % of openings at level 2+ (weak/preemptive style)
        intervention_rate   — % of boards where they bid into opp's auction
        penalty_double_rate — % of boards where they made a double
        avg_bids_per_board  — mean number of real bids made per board
        n_bidding_boards    — count of boards with bidding info

    All four seats (N/S/E/W) per board contribute one record per room.
    """
    bid_df = df[df["has_bidding"] == True].copy()  # noqa: E712
    logger.info("Extracting bidding features from %d boards ...", len(bid_df))

    rows: list[dict] = []
    for tup in bid_df.itertuples(index=False):
        bidding = getattr(tup, "bidding", None)
        room = getattr(tup, "room", None)
        if not isinstance(bidding, str) or not isinstance(room, str):
            continue
        prefix = "open" if room.strip().lower() == "open" else "closed"
        for pos in ("N", "S", "E", "W"):
            name_col = f"{prefix}_{POSITION_MAP[pos]}"
            name = getattr(tup, name_col, None)
            if not isinstance(name, str) or not name.strip():
                continue
            feats = player_bidding_features(bidding, pos)
            feats["player_name"] = name.strip()
            rows.append(feats)

    if not rows:
        logger.warning("No bidding records extracted — returning empty DataFrame.")
        return pd.DataFrame(columns=[
            "player_name", "n_bidding_boards", "opening_rate",
            "preempt_rate", "intervention_rate",
            "penalty_double_rate", "avg_bids_per_board",
        ])

    long_df = pd.DataFrame(rows)
    logger.info("Aggregating bidding features over %d player-board records ...",
                len(long_df))

    agg = (
        long_df.groupby("player_name")
        .agg(
            n_bidding_boards=("opened", "count"),
            opening_rate=("opened", "mean"),
            preempt_rate=("is_preempt", "mean"),
            intervention_rate=("intervened", "mean"),
            penalty_double_rate=("made_double", "mean"),
            avg_bids_per_board=("n_bids", "mean"),
        )
        .reset_index()
    )

    # Round for readability
    for col in ["opening_rate", "preempt_rate", "intervention_rate",
                "penalty_double_rate", "avg_bids_per_board"]:
        agg[col] = agg[col].round(4)

    return agg


# ── Public API ───────────────────────────────────────────────────────────────

def compute_player_features(
    df: pd.DataFrame,
    min_boards: int = MIN_BOARDS,
    min_bidding_boards: int = 50,  # Raised from 20 (May 2026) — sample-size robustness
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
    work["_is_doubled"]    = work["contract"].apply(_is_doubled)
    work["_is_nt"]         = work["contract"].apply(_is_nt)
    work["_is_partscore"]  = work["contract"].apply(_is_partscore)
    work["_is_game"]       = work["contract"].apply(_is_game)
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
            nt_rate=("_is_nt", "mean"),
            partscore_rate=("_is_partscore", "mean"),
            game_rate=("_is_game", "mean"),
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
    for col in ["slam_rate", "success_rate", "double_rate",
                "avg_level", "nt_rate", "partscore_rate", "game_rate"]:
        agg[col] = agg[col].round(4)

    agg = agg.sort_values("n_declared", ascending=False).reset_index(drop=True)

    # ── Step 6: Merge bidding-PROCESS features (from `bidding` column) ──────
    # These give us stylistic decision-process signals: who opens, preempts,
    # intervenes, doubles. Only ~31% of boards have bidding, so missing values
    # are filled with 0.
    if "bidding" in df.columns and "has_bidding" in df.columns:
        bid_feats = compute_bidding_features(df)
        agg = agg.merge(bid_feats, on="player_name", how="left")

        # Fill NaN for players who have no bidding data with 0
        for col in ["opening_rate", "preempt_rate", "intervention_rate",
                    "penalty_double_rate", "avg_bids_per_board"]:
            if col in agg.columns:
                agg[col] = agg[col].fillna(0.0)
        if "n_bidding_boards" in agg.columns:
            agg["n_bidding_boards"] = agg["n_bidding_boards"].fillna(0).astype(int)

        logger.info(
            "Merged bidding features for %d / %d players",
            (agg["n_bidding_boards"] > 0).sum(),
            len(agg),
        )

        # Filter: keep only players with enough bidding boards to trust their
        # process features (zeros would otherwise form a spurious cluster)
        if min_bidding_boards > 0 and "n_bidding_boards" in agg.columns:
            before = len(agg)
            agg = agg[agg["n_bidding_boards"] >= min_bidding_boards].copy()
            logger.info(
                "Players with >= %d bidding boards: %d / %d",
                min_bidding_boards, len(agg), before,
            )

    logger.info(
        "Feature matrix ready: %d players × %d features",
        len(agg), len(agg.columns) - 1,  # exclude player_name
    )

    return agg
