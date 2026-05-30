"""
src/stage4_simulate/bridge_game.py
==================================
Deterministic bridge "game" infrastructure for Stage 4a.

This module is PURE (no LLM, no network): it deals hands reproducibly from a
seed and scores a chosen contract level objectively against the hand's strength.
The agents (which DO call the LLM) are wired in by bridge_runner.py.

Why an objective scorer instead of a real bridge engine?
--------------------------------------------------------
A full double-dummy solver (e.g. the BEN engine) is deferred to PhD Year 2
(see PRD §8). For the course MVP we use a transparent, deterministic rubric:
the combined strength of the partnership implies a "par" contract level
(partscore / game / slam), and an agent's bid is scored by how close it lands
to that par. This rewards good bridge judgement and—crucially—separates the
profiles: a Slam Hunter is rewarded for reaching slam on strong hands but
penalised for overbidding weak ones; an Insurance Player is the mirror image.

The score is the behavioural signal we need for the alignment study; it is not
a claim about real-table results.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

# ── Card constants ────────────────────────────────────────────────────────────

SUITS = ["S", "H", "D", "C"]
RANKS = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]

# High-card points (Milton Work count).
_HCP = {"A": 4, "K": 3, "Q": 2, "J": 1}


# ── Par-contract thresholds (combined partnership HCP) ────────────────────────
# Transparent, standard bridge guidance:
#   ~37+ HCP -> grand slam,  ~33-36 -> small slam,
#   ~25-32  -> game,         else  -> partscore.
# We score on the LEVEL CLASS (partscore / game / slam), not the exact strain.
GRAND_SLAM_HCP = 37
SMALL_SLAM_HCP = 33
GAME_HCP = 25

# Level-class labels
PARTSCORE, GAME, SMALL_SLAM, GRAND_SLAM = "partscore", "game", "small_slam", "grand_slam"

# Order for distance computation
_CLASS_ORDER = [PARTSCORE, GAME, SMALL_SLAM, GRAND_SLAM]


# ── Data containers ───────────────────────────────────────────────────────────

@dataclass
class Deal:
    """One reproducible deal: four 13-card hands keyed by position."""

    board: int
    hands: dict[str, dict[str, str]]   # position -> {suit: "AKQ.."}
    seed: int

    def hcp(self, position: str) -> int:
        return hand_hcp(self.hands[position])

    def partnership_hcp(self, position: str) -> int:
        """Combined HCP of `position` and its partner (N+S or E+W)."""
        partner = {"N": "S", "S": "N", "E": "W", "W": "E"}[position]
        return self.hcp(position) + self.hcp(partner)


@dataclass
class BridgeScore:
    """Objective score for one bid on one deal."""

    bid_level_class: str       # the class the agent's bid corresponds to
    par_level_class: str       # the class implied by partnership HCP
    distance: int              # |index difference| between the two classes
    score: float               # 1.0 = perfect, decreasing with distance
    partnership_hcp: int
    raw_bid: str = ""
    notes: str = field(default="")


# ── Hand dealing (deterministic) ──────────────────────────────────────────────

def deal_board(board: int, seed: int = 42) -> Deal:
    """Deal one reproducible board.

    The same (board, seed) always yields the same four hands, so the whole
    Stage 4a experiment is reproducible regardless of LLM non-determinism.

    Args:
        board: Board number (also mixed into the RNG so each board differs).
        seed:  Base seed for the experiment.

    Returns:
        A Deal with N/S/E/W hands.
    """
    rng = random.Random(seed * 100_000 + board)
    deck = [r + s for s in SUITS for r in RANKS]
    rng.shuffle(deck)

    positions = ["N", "E", "S", "W"]
    hands: dict[str, dict[str, str]] = {p: {s: "" for s in SUITS} for p in positions}

    for i, card in enumerate(deck):
        pos = positions[i % 4]
        rank, suit = card[0], card[1]
        hands[pos][suit] += rank

    # Sort each suit high-to-low for readability.
    rank_order = {r: i for i, r in enumerate(RANKS)}
    for pos in positions:
        for suit in SUITS:
            hands[pos][suit] = "".join(
                sorted(hands[pos][suit], key=lambda r: rank_order[r])
            )

    return Deal(board=board, hands=hands, seed=seed)


# ── Hand evaluation ───────────────────────────────────────────────────────────

def hand_hcp(hand: dict[str, str]) -> int:
    """High-card points of one hand."""
    return sum(_HCP.get(card, 0) for suit in SUITS for card in hand.get(suit, ""))


def par_level_class(partnership_hcp: int) -> str:
    """Map combined partnership HCP to the par contract level class."""
    if partnership_hcp >= GRAND_SLAM_HCP:
        return GRAND_SLAM
    if partnership_hcp >= SMALL_SLAM_HCP:
        return SMALL_SLAM
    if partnership_hcp >= GAME_HCP:
        return GAME
    return PARTSCORE


def bid_level_class(bid: str) -> str | None:
    """Map a contract bid string to its level class.

    Returns None for non-contract calls (Pass / X / XX) — the caller decides
    how to score those.
    """
    if not isinstance(bid, str) or not bid:
        return None
    b = bid.strip().upper()
    if not b or b[0] not in "1234567":
        return None
    level = int(b[0])
    if level >= 7:
        return GRAND_SLAM
    if level == 6:
        return SMALL_SLAM
    # Game classes: 3NT, 4H/4S, 5C/5D. Simplify: level 5 = game,
    # level 4 = game if major/NT else partscore, level 3 = game only if NT.
    strain = b[1:]
    if level == 5:
        return GAME
    if level == 4:
        return GAME if strain in ("H", "S", "NT") else PARTSCORE
    if level == 3:
        return GAME if strain == "NT" else PARTSCORE
    return PARTSCORE


# ── Objective scoring ─────────────────────────────────────────────────────────

def score_bid(bid: str, partnership_hcp: int) -> BridgeScore:
    """Score one agent bid objectively against the hand's par level.

    Scoring rubric (transparent, deterministic):
      - distance 0  (right level class)      -> 1.00
      - distance 1  (one class off)          -> 0.50
      - distance 2                           -> 0.20
      - distance 3                           -> 0.00
      - Pass when par is partscore           -> 0.50 (acceptable: no game values)
      - Pass when par is game or higher      -> 0.00 (missed a makeable contract)

    Args:
        bid:             The agent's call ("4S", "Pass", "6NT", ...).
        partnership_hcp: Combined HCP of the bidding side.

    Returns:
        BridgeScore.
    """
    par = par_level_class(partnership_hcp)
    bclass = bid_level_class(bid)

    if bclass is None:
        # Non-contract call (Pass/X/XX). Only Pass is meaningfully scorable here.
        if bid.strip().lower() == "pass":
            score = 0.5 if par == PARTSCORE else 0.0
            return BridgeScore(
                bid_level_class="pass",
                par_level_class=par,
                distance=_CLASS_ORDER.index(par),
                score=score,
                partnership_hcp=partnership_hcp,
                raw_bid=bid,
                notes="passed",
            )
        # X / XX / garbage — neutral-low, not the focus of this rubric.
        return BridgeScore(
            bid_level_class="other",
            par_level_class=par,
            distance=99,
            score=0.0,
            partnership_hcp=partnership_hcp,
            raw_bid=bid,
            notes="non-contract call",
        )

    distance = abs(_CLASS_ORDER.index(bclass) - _CLASS_ORDER.index(par))
    score = {0: 1.0, 1: 0.5, 2: 0.2}.get(distance, 0.0)
    return BridgeScore(
        bid_level_class=bclass,
        par_level_class=par,
        distance=distance,
        score=score,
        partnership_hcp=partnership_hcp,
        raw_bid=bid,
    )


__all__ = [
    "SUITS",
    "RANKS",
    "Deal",
    "BridgeScore",
    "deal_board",
    "hand_hcp",
    "par_level_class",
    "bid_level_class",
    "score_bid",
    "PARTSCORE",
    "GAME",
    "SMALL_SLAM",
    "GRAND_SLAM",
]
