"""
src/features/double_dummy.py
============================
Double-dummy (perfect-play) bridge evaluation — Task T-C.

Why (Nezer, bridge expert): the Stage-4a bridge metric used an **HCP-based PROXY**
for the par contract. The bridge standard is **double-dummy (DD)**: the optimal
result when all 52 cards are visible. This module wraps the DDS solver (via
`endplay`, which embeds Bo Haglund's DDS) to give:

  - `dd_par_level_class(hands)` — the TRUE par level class for N-S (partscore /
    game / small_slam / grand_slam), replacing the HCP proxy.
  - `dd_bid_score01(bid, hands)` — a FINER, points-based score in [0, 1] of a
    single N-S contract bid, measured by its double-dummy score relative to the
    best score N-S can achieve. Unlike the old 4-class rubric, this properly
    punishes overbidding (a random 7NT that goes down scores ~0), which the
    coarse proxy did not — that is why a random "monkey" beat the profiles on
    the old metric.

Scope / limitation (state honestly): DD scores the CONTRACT (bidding side) from
the 52 cards, which we have. It does NOT score defensive *card play* quality,
which needs the trick-by-trick record we do not have. So this upgrades the
bidding metric; the Fighter's defensive strength still rides on the doubling
feature, not on DD.

All functions take our native hand format:
    hands = {"N": {"S": "AKQ72", "H": "K4", "D": "A83", "C": "Q92"}, "E": {...},
             "S": {...}, "W": {...}}
"""

from __future__ import annotations

from functools import lru_cache

from endplay.dds import calc_dd_table
from endplay.types import Contract, Deal as EPDeal, Denom, Player

# Our suit order for PBN rendering (PBN lists spades.hearts.diamonds.clubs).
_PBN_SUITS = ["S", "H", "D", "C"]

# Map our strain letters to endplay Denom and to the Contract-string letter.
_STRAIN_TO_DENOM = {
    "C": Denom.clubs, "D": Denom.diamonds, "H": Denom.hearts,
    "S": Denom.spades, "NT": Denom.nt,
}
_STRAIN_TO_LETTER = {"C": "C", "D": "D", "H": "H", "S": "S", "NT": "N"}

# Level-class labels (kept identical to bridge_game so scoring code lines up).
PARTSCORE, GAME, SMALL_SLAM, GRAND_SLAM = "partscore", "game", "small_slam", "grand_slam"

# Minimum DD tricks needed to MAKE game in each strain (book = 6 tricks).
#   3NT = 9, 4H/4S = 10, 5C/5D = 11.
_GAME_TRICKS = {"NT": 9, "H": 10, "S": 10, "C": 11, "D": 11}
_NS_DECLARERS = (Player.north, Player.south)
_ALL_STRAINS = ["C", "D", "H", "S", "NT"]


# ── Conversion ────────────────────────────────────────────────────────────────

def hands_to_pbn(hands: dict[str, dict[str, str]]) -> str:
    """Render our hand dict as an endplay/PBN deal string, dealer-anchored at N."""
    def one(pos: str) -> str:
        h = hands[pos]
        return ".".join((h.get(s, "") or "") for s in _PBN_SUITS)
    return "N:" + " ".join(one(p) for p in ("N", "E", "S", "W"))


@lru_cache(maxsize=4096)
def _dd_table_for_pbn(pbn: str):
    """Cached DD trick table for a PBN string (solver call is the expensive bit)."""
    return calc_dd_table(EPDeal(pbn))


def ns_makeable_tricks(hands: dict[str, dict[str, str]]) -> dict[str, int]:
    """Max DD tricks N-S can take in each strain (best of North/South declaring)."""
    table = _dd_table_for_pbn(hands_to_pbn(hands))
    out: dict[str, int] = {}
    for strain in _ALL_STRAINS:
        denom = _STRAIN_TO_DENOM[strain]
        out[strain] = max(int(table[denom, d]) for d in _NS_DECLARERS)
    return out


# ── True par level class (replaces the HCP proxy) ─────────────────────────────

def dd_par_level_class(hands: dict[str, dict[str, str]]) -> str:
    """The double-dummy par level class for the N-S partnership.

    Returns the highest contract class N-S can MAKE double-dummy:
      grand_slam (13 tricks) > small_slam (12) > game (3NT/4M/5m) > partscore.
    """
    tricks = ns_makeable_tricks(hands)
    if any(t >= 13 for t in tricks.values()):
        return GRAND_SLAM
    if any(t >= 12 for t in tricks.values()):
        return SMALL_SLAM
    if any(tricks[s] >= _GAME_TRICKS[s] for s in _ALL_STRAINS):
        return GAME
    return PARTSCORE


# ── Finer, points-based score of a single bid ─────────────────────────────────

def _contract_dd_points(level: int, strain: str, hands: dict, vul: bool) -> int:
    """Double-dummy points for N-S playing `level strain` (best N/S declarer)."""
    denom = _STRAIN_TO_DENOM[strain]
    table = _dd_table_for_pbn(hands_to_pbn(hands))
    # Pick the N-S declarer who takes more tricks in this strain.
    declarer = max(_NS_DECLARERS, key=lambda d: int(table[denom, d]))
    tricks = int(table[denom, declarer])
    result = tricks - (level + 6)                      # over/under tricks
    res_str = "=" if result == 0 else (f"+{result}" if result > 0 else str(result))
    decl_letter = "N" if declarer == Player.north else "S"
    contract = Contract(f"{level}{_STRAIN_TO_LETTER[strain]}{decl_letter}{res_str}")
    return contract.score(vul)


def _best_ns_points(hands: dict, vul: bool) -> int:
    """Best double-dummy score N-S can achieve over all makeable contracts."""
    best = 0
    for strain in _ALL_STRAINS:
        for level in range(1, 8):
            pts = _contract_dd_points(level, strain, hands, vul)
            if pts > best:
                best = pts
    return best


def _best_ns_partscore_points(hands: dict, vul: bool) -> int:
    """Best DD score among partscore-level contracts (what 'passing low' secures)."""
    best = 0
    for strain in _ALL_STRAINS:
        # Partscore = below game level for that strain.
        max_ps_level = {"NT": 2, "H": 3, "S": 3, "C": 4, "D": 4}[strain]
        for level in range(1, max_ps_level + 1):
            pts = _contract_dd_points(level, strain, hands, vul)
            if pts > best:
                best = pts
    return best


def dd_bid_score01(bid: str, hands: dict[str, dict[str, str]], vul: bool = False) -> float:
    """Score a single N-S bid in [0, 1] by its double-dummy outcome.

    1.0 = the bid realises N-S's best achievable double-dummy score; 0.0 = it
    scores nothing or goes negative (e.g. a doomed overbid). This is far finer
    than the 4-class proxy and punishes random overbidding, so genuine skill
    separates from a random baseline.

    Args:
        bid:  the agent's call ("4S", "3NT", "6NT", "Pass", "X" ...).
        hands: full 52-card deal.
        vul:  N-S vulnerability (affects scoring magnitudes).
    """
    best = _best_ns_points(hands, vul)
    if best <= 0:
        # N-S can make nothing positive double-dummy; passing is correct.
        return 1.0 if str(bid).strip().lower() == "pass" else 0.0

    b = str(bid).strip()
    low = b.lower()

    if low == "pass":
        # Passing secures the best partscore N-S can make.
        pts = _best_ns_partscore_points(hands, vul)
        return max(0.0, min(1.0, pts / best))

    if low in ("x", "xx"):
        # Not a constructive N-S contract in this single-bid simulation.
        return 0.0

    # Contract bid: parse level + strain.
    if len(b) >= 2 and b[0] in "1234567":
        level = int(b[0])
        strain = b[1:].upper()
        strain = "NT" if strain in ("N", "NT") else strain
        if strain in _STRAIN_TO_DENOM:
            pts = _contract_dd_points(level, strain, hands, vul)
            return max(0.0, min(1.0, pts / best))

    return 0.0  # unrecognised call


__all__ = [
    "hands_to_pbn", "ns_makeable_tricks", "dd_par_level_class",
    "dd_bid_score01", "PARTSCORE", "GAME", "SMALL_SLAM", "GRAND_SLAM",
]
