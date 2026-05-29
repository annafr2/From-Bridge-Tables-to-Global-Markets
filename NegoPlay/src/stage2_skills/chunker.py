"""
src/stage2_skills/chunker.py
=============================
Prepare bridge games for LLM skill extraction.

Given a player and the raw matches DataFrame, this module:
  1. Finds all boards where the player sat at the table
  2. Identifies the player's position (N/S/E/W) per board
  3. Formats each board as a compact, LLM-friendly text representation
  4. Splits the boards into chunks of N (default 25) for batched LLM calls

The output format is intentionally terse — every token costs money — but it
preserves the four facts the LLM needs to characterise behaviour:
    1. What the player held (the 13 cards)
    2. What they bid (their actions in the auction)
    3. What the table did (full auction context)
    4. What the result was (contract + tricks + score)

Usage:
    from src.stage2_skills.chunker import build_player_chunks
    chunks = build_player_chunks(df, player_name="DONCIU Sabin-Horia", chunk_size=25)
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {len(chunk.boards)} boards, ~{chunk.estimated_tokens} tokens")
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Iterator

import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# 8 player-name columns in the matches dataset (room × position)
PLAYER_COLUMNS: list[str] = [
    "open_north",  "open_south",  "open_east",  "open_west",
    "closed_north", "closed_south", "closed_east", "closed_west",
]

# Map column → (room, position)
COLUMN_TO_SEAT: dict[str, tuple[str, str]] = {
    "open_north":   ("Open",   "N"),
    "open_south":   ("Open",   "S"),
    "open_east":    ("Open",   "E"),
    "open_west":    ("Open",   "W"),
    "closed_north": ("Closed", "N"),
    "closed_south": ("Closed", "S"),
    "closed_east":  ("Closed", "E"),
    "closed_west":  ("Closed", "W"),
}

# Hand suit columns by position
HAND_COLUMNS: dict[str, list[str]] = {
    "N": ["north_spades", "north_hearts", "north_diamonds", "north_clubs"],
    "S": ["south_spades", "south_hearts", "south_diamonds", "south_clubs"],
    "E": ["east_spades",  "east_hearts",  "east_diamonds",  "east_clubs"],
    "W": ["west_spades",  "west_hearts",  "west_diamonds",  "west_clubs"],
}

# Default chunk size — keeps total tokens per call well under 5K
DEFAULT_CHUNK_SIZE: int = 25

# Rough estimate: each formatted board ≈ 80 tokens (4 chars/token avg)
EST_TOKENS_PER_BOARD: int = 80


# ── Output containers ─────────────────────────────────────────────────────────

@dataclass
class BoardSummary:
    """One bridge board, from a specific player's perspective."""

    board_id: int
    player_seat: str            # N / S / E / W
    dealer: str | None          # who dealt
    vulnerability: str | None   # None / NS / EW / Both
    hand: str                   # "♠Q98 ♥J8 ♦76 ♣K953"
    bidding: str                # full auction string from the dataset
    contract: str | None        # e.g. "5Dx"
    declarer: str | None        # who played the contract
    tricks: int | None          # tricks won by declarer
    ns_score: int | None        # NS scoring
    player_role: str            # "Declarer" / "Defender" / "Partner" / "Dummy"

    def to_text(self) -> str:
        """Render this board as a compact text block for the LLM."""
        vul = self.vulnerability or "None"
        contract_str = self.contract or "?"
        result_str = (
            f"made {self.tricks} tricks" if self.tricks is not None else "result?"
        )
        score_str = f"NS {self.ns_score:+d}" if self.ns_score is not None else "score?"

        return (
            f"Board {self.board_id} (Dealer {self.dealer or '?'}, Vul: {vul})\n"
            f"You are {self.player_seat} ({self.player_role}). "
            f"Your hand: {self.hand}\n"
            f"Bidding: {self.bidding}\n"
            f"Result: {contract_str} by {self.declarer or '?'}, "
            f"{result_str}, {score_str}"
        )


@dataclass
class PlayerChunk:
    """A batch of board summaries for one player, ready for one LLM call."""

    player_name: str
    profile: str | None
    boards: list[BoardSummary] = field(default_factory=list)

    @property
    def estimated_tokens(self) -> int:
        return len(self.boards) * EST_TOKENS_PER_BOARD

    def to_prompt(self) -> str:
        """Render all boards as a single prompt body."""
        separator = "=" * 60
        header = (
            f"Player: {self.player_name}\n"
            f"Profile: {self.profile or 'unknown'}\n"
            f"Total boards in this chunk: {len(self.boards)}\n\n"
            f"{separator}\n\n"
        )
        body = "\n\n---\n\n".join(b.to_text() for b in self.boards)
        return header + body


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_hand(row: pd.Series, position: str) -> str:
    """Render the 13-card hand for one seat as '♠AKQ ♥... ♦... ♣...'."""
    cols = HAND_COLUMNS[position]
    suits = ["♠", "♥", "♦", "♣"]
    parts: list[str] = []
    for suit, col in zip(suits, cols):
        cards = row.get(col)
        if pd.isna(cards) or cards == "-":
            parts.append(f"{suit}—")
        else:
            # Cards stored as "A T 7 6 5 2" — compact to "AT7652"
            compact = "".join(str(cards).split())
            parts.append(f"{suit}{compact}")
    return " ".join(parts)


def _player_role(
    declarer: str | None,
    player_seat: str,
) -> str:
    """Classify the player's role in the deal: Declarer / Dummy / Defender."""
    if not declarer or pd.isna(declarer):
        return "Unknown"
    declarer_seat = str(declarer).strip().upper()[:1]

    # Partnership maps
    partner = {"N": "S", "S": "N", "E": "W", "W": "E"}

    if declarer_seat == player_seat:
        return "Declarer"
    if declarer_seat == partner.get(player_seat):
        return "Dummy"
    return "Defender"


def _row_to_board_summary(
    row: pd.Series,
    seat: str,
) -> BoardSummary:
    """Turn one matches-row into a BoardSummary from the player's seat."""

    # Vulnerability — dataset has it sometimes
    vul = row.get("vulnerability")
    if pd.isna(vul):
        vul = None
    else:
        vul = str(vul)

    dealer = row.get("dealer")
    if pd.isna(dealer):
        dealer = None
    else:
        dealer = str(dealer)

    tricks = row.get("tricks")
    if pd.isna(tricks):
        tricks_int: int | None = None
    else:
        tricks_int = int(tricks)

    ns_score = row.get("ns_score")
    ns_int = int(ns_score) if not pd.isna(ns_score) else None

    return BoardSummary(
        board_id=int(row.get("board") or 0),
        player_seat=seat,
        dealer=dealer,
        vulnerability=vul,
        hand=_format_hand(row, seat),
        bidding=str(row.get("bidding") or ""),
        contract=str(row.get("contract")) if not pd.isna(row.get("contract")) else None,
        declarer=str(row.get("declarer")) if not pd.isna(row.get("declarer")) else None,
        tricks=tricks_int,
        ns_score=ns_int,
        player_role=_player_role(row.get("declarer"), seat),
    )


# ── Public API ────────────────────────────────────────────────────────────────

def find_player_boards(
    df: pd.DataFrame,
    player_name: str,
    require_bidding: bool = True,
    require_cards: bool = True,
) -> pd.DataFrame:
    """Return the subset of rows where `player_name` sat at the table.

    Adds a `player_seat` column ('N' / 'S' / 'E' / 'W') to each row so the
    caller knows which hand was the target player's.

    Args:
        df:              Full matches DataFrame from load_matches().
        player_name:     Player name exactly as it appears in the open_/closed_ columns.
        require_bidding: Drop rows where has_bidding=False (default True).
        require_cards:   Drop rows where has_cards=False (default True).
    """
    # Build a mask across all 8 player columns; remember which one matched
    rows: list[pd.DataFrame] = []
    for col in PLAYER_COLUMNS:
        mask = df[col].astype(str).str.strip() == player_name.strip()
        if not mask.any():
            continue
        room, seat = COLUMN_TO_SEAT[col]
        sub = df.loc[mask].copy()
        sub["player_seat"] = seat
        sub["player_room"] = room
        rows.append(sub)

    if not rows:
        return df.iloc[0:0].assign(player_seat="", player_room="")

    result = pd.concat(rows, ignore_index=True)

    if require_bidding:
        result = result[result["has_bidding"].fillna(False)]
    if require_cards:
        result = result[result["has_cards"].fillna(False)]

    return result.reset_index(drop=True)


def build_player_chunks(
    df: pd.DataFrame,
    player_name: str,
    profile: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_boards: int | None = None,
    seed: int = 42,
) -> list[PlayerChunk]:
    """Build LLM-ready chunks of boards for one player.

    Sampling is deterministic (seeded) — same player always yields same boards.

    Args:
        df:           Full matches DataFrame.
        player_name:  Target player.
        profile:      Profile label to embed in the prompt header (optional).
        chunk_size:   Boards per chunk (default 25).
        max_boards:   Cap total boards (default None = all). Useful for cost control.
        seed:         Random seed for sampling when max_boards < total.

    Returns:
        List of PlayerChunk, each ready to feed into extractor.extract_skills().
    """
    player_rows = find_player_boards(df, player_name)
    n_total = len(player_rows)

    if n_total == 0:
        logger.warning("No boards found for player %r", player_name)
        return []

    # Apply max_boards cap with deterministic sampling
    if max_boards is not None and n_total > max_boards:
        player_rows = player_rows.sample(
            n=max_boards, random_state=seed
        ).reset_index(drop=True)
        logger.info(
            "Sampled %d / %d boards for %s (seed=%d)",
            max_boards, n_total, player_name, seed,
        )

    # Build BoardSummary list
    boards: list[BoardSummary] = []
    for _, row in player_rows.iterrows():
        seat = row["player_seat"]
        boards.append(_row_to_board_summary(row, seat))

    # Split into chunks
    chunks: list[PlayerChunk] = []
    for i in range(0, len(boards), chunk_size):
        chunk = PlayerChunk(
            player_name=player_name,
            profile=profile,
            boards=boards[i : i + chunk_size],
        )
        chunks.append(chunk)

    logger.info(
        "Built %d chunks for %s (%d boards total, %d boards/chunk)",
        len(chunks), player_name, len(boards), chunk_size,
    )
    return chunks


def iter_profile_chunks(
    df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_boards_per_player: int | None = 50,
    profile_filter: list[str] | None = None,
    max_players_per_profile: int | None = None,
    seed: int = 42,
) -> Iterator[PlayerChunk]:
    """Yield chunks across every player in profiles_df.

    Convenience wrapper that lets you stream all chunks for Stage 2 in one
    go, for cost control: `max_boards_per_player` caps work per player and
    `max_players_per_profile` caps players sampled per profile (useful for
    pilot runs).

    Args:
        df:                       Full matches DataFrame.
        profiles_df:              Output of Stage 1 — must have 'player_name'
                                  and 'profile' columns.
        chunk_size:               Boards per chunk.
        max_boards_per_player:    Cap per player (default 50).
        profile_filter:           Only process these profile names (default: all).
        max_players_per_profile:  Subsample N players per profile (default: all).
        seed:                     Sampling seed.

    Yields:
        PlayerChunk objects, one chunk at a time.
    """
    df_profiles = profiles_df.copy()
    if profile_filter is not None:
        df_profiles = df_profiles[df_profiles["profile"].isin(profile_filter)]

    if max_players_per_profile is not None:
        df_profiles = (
            df_profiles.groupby("profile", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), max_players_per_profile), random_state=seed))
            .reset_index(drop=True)
        )

    for _, row in df_profiles.iterrows():
        player_name = row["player_name"]
        profile = row["profile"]
        chunks = build_player_chunks(
            df=df,
            player_name=player_name,
            profile=profile,
            chunk_size=chunk_size,
            max_boards=max_boards_per_player,
            seed=seed,
        )
        for chunk in chunks:
            yield chunk


__all__ = [
    "BoardSummary",
    "PlayerChunk",
    "find_player_boards",
    "build_player_chunks",
    "iter_profile_chunks",
    "PLAYER_COLUMNS",
    "DEFAULT_CHUNK_SIZE",
]
