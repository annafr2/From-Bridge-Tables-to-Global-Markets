"""
src/stage1_clustering/bidding_parser.py
========================================
Parse bidding sequences into per-player process features.

Bidding format (from EuroBridge):
    'W:- N:1S E:2D S:3S | W:4S N:Pass E:5D S:Pass | W:Pass N:x E:Pass S:Pass'

    - Positions: W / N / E / S
    - '-'    → no bid yet (player wasn't dealer or auction not yet reached them)
    - 'Pass' → pass
    - '1S'   → bid at level 1 in spades
    - 'x'    → double
    - 'xx'   → redouble
    - '|'    → round separator (4 bids per round, one per position)

Outputs per (board, position) a dict describing what THAT player did:
    opened           — did this player open the auction?
    opening_bid      — their opening bid (if any), e.g. '1S', '2H', '3NT'
    opening_level    — level of the opening bid (1–7)
    is_preempt       — opening at level 2+ (weak preemptive style)
    is_strong_open   — opening 2C or higher conventional strong bid
    n_bids           — how many bids this player made (excl. Pass / '-')
    n_passes         — how many Pass calls
    made_double      — did they make a double (penalty or takeout)
    made_redouble    — did they redouble
    intervened       — did they bid AFTER an opponent opened?
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Tokens that are not real bids (do not count toward "n_bids made")
NON_BID_TOKENS: set[str] = {"-", "Pass", "PASS", "pass"}

# Regex for an actual contract bid like '1S', '2NT', '4H', '7C'
_BID_RE = re.compile(r"^[1-7](?:C|D|H|S|NT)$", re.IGNORECASE)


def _parse_token(token: str) -> Optional[str]:
    """Normalize one bidding token. Returns None if it's not a real action."""
    if token is None:
        return None
    t = token.strip()
    if t in NON_BID_TOKENS or t == "":
        return None
    return t  # could be '1S', '3NT', 'x', 'xx'


def _is_real_bid(token: str) -> bool:
    """True if token is a contract bid (level + strain), not a double/pass."""
    return bool(_BID_RE.match(token))


def _bid_level(token: str) -> Optional[int]:
    """Return level (1-7) of a contract bid, or None if not a bid."""
    if not isinstance(token, str) or len(token) < 1:
        return None
    if token[0] in "1234567":
        return int(token[0])
    return None


def parse_bidding(bidding: str) -> list[tuple[str, str]]:
    """Parse a bidding string into an ordered list of (position, token) pairs.

    Skips '-' placeholders (used before the dealer's turn).

    Example input:
        'W:- N:1S E:2D S:3S | W:4S N:Pass E:5D S:Pass | W:Pass'

    Returns:
        [('N','1S'), ('E','2D'), ('S','3S'),
         ('W','4S'), ('N','Pass'), ('E','5D'), ('S','Pass'),
         ('W','Pass')]
    """
    if not isinstance(bidding, str) or not bidding.strip():
        return []

    sequence: list[tuple[str, str]] = []
    # Split into rounds by '|', then split each round on whitespace
    rounds = bidding.split("|")
    for rnd in rounds:
        for chunk in rnd.strip().split():
            # Each chunk is 'POS:TOKEN'
            if ":" not in chunk:
                continue
            pos, tok = chunk.split(":", 1)
            pos = pos.strip().upper()
            tok = tok.strip()
            if tok == "-":
                continue   # not a bid event — auction hasn't reached this seat yet
            sequence.append((pos, tok))
    return sequence


def player_bidding_features(bidding: str, position: str) -> dict:
    """Extract per-board features for ONE player (declarer or any seat).

    Args:
        bidding: full bidding string for the board.
        position: 'N' / 'S' / 'E' / 'W' — the player whose actions we summarize.

    Returns:
        dict with: opened, opening_bid, opening_level, is_preempt,
        is_strong_open, n_bids, n_passes, made_double, made_redouble,
        intervened.
    """
    position = position.upper().strip()
    sequence = parse_bidding(bidding)

    out = {
        "opened": False,
        "opening_bid": None,
        "opening_level": None,
        "is_preempt": False,
        "is_strong_open": False,
        "n_bids": 0,
        "n_passes": 0,
        "made_double": False,
        "made_redouble": False,
        "intervened": False,
    }

    if not sequence:
        return out

    # ── Find the auction opener (first real bid by anyone) ───────────────────
    opener_pos = None
    opener_idx = None
    for i, (pos, tok) in enumerate(sequence):
        if _is_real_bid(tok):
            opener_pos = pos
            opener_idx = i
            break

    # ── Walk through THIS player's actions ────────────────────────────────────
    player_actions = [(i, tok) for i, (pos, tok) in enumerate(sequence) if pos == position]

    first_bid_seen = False
    for i, tok in player_actions:
        low = tok.lower()
        if low == "pass":
            out["n_passes"] += 1
            continue
        if low == "x":
            out["made_double"] = True
            continue
        if low == "xx":
            out["made_redouble"] = True
            continue
        if _is_real_bid(tok):
            out["n_bids"] += 1
            if not first_bid_seen:
                first_bid_seen = True
                # Was this player the auction opener?
                if opener_pos == position and i == opener_idx:
                    out["opened"] = True
                    out["opening_bid"] = tok
                    lvl = _bid_level(tok)
                    out["opening_level"] = lvl
                    if lvl is not None and lvl >= 2:
                        out["is_preempt"] = True
                    # Strong artificial 2C (or 2D in Precision) — convention varies,
                    # we just flag any level-2 club open as "potentially strong"
                    if tok.upper() == "2C":
                        out["is_strong_open"] = True
                else:
                    # Player bid AFTER someone else opened → intervention
                    if opener_pos is not None and opener_pos != position:
                        # Opponents are the "other side" — partner is across the table
                        partner = {"N": "S", "S": "N", "E": "W", "W": "E"}[position]
                        if opener_pos != partner:
                            out["intervened"] = True

    return out
