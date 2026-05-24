"""
EuroBridge Cards Scraper
========================
Fetches card holdings (the 52 cards dealt to N/S/E/W) for every board
that was already collected by the bulk scraper.

WHY this is a separate script:
  - The bulk scraper already collected bidding + results (71,928 rows).
  - Card holdings live on a different page: BoardAcross.asp
  - We do NOT need to re-scrape matches. We only need one BoardAcross
    request per unique (competition, category, round, board_number).

HOW it works:
  1. For each competition+category in competitions.yaml:
       - Read the existing matches.csv to find all rounds that exist
       - For each round, fetch BoardAcross pages for boards 1..16
       - Extract: dealer, vulnerability, N/S/E/W card holdings
       - Save to: data/raw/eurobridge/<competition>/<category>/cards.csv

  3. The cards CSV can then be joined to matches.csv on (round + board).

EFFICIENCY:
  - Many matches play the SAME board in the same round.
    One BoardAcross request covers ALL of them.
  - Total requests: ~5 competitions × 4 categories × ~10 rounds × 16 boards
    = roughly 3,200 requests (not 71,928!)
  - At 0.8s delay: ~45 minutes total.

Usage:
    # All competitions:
    python src/downloaders/eurobridge_cards_scraper.py

    # One competition only:
    python src/downloaders/eurobridge_cards_scraper.py --competitions EBL_Herning_2024

    # One category only:
    python src/downloaders/eurobridge_cards_scraper.py --categories Mixed

    # Test with just 1 round:
    python src/downloaders/eurobridge_cards_scraper.py --max-rounds 1
"""

import sys
import csv
import time
import logging
from pathlib import Path
from datetime import datetime

import yaml
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from eurobridge_scraper import EuroBridgeScraper

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE  = PROJECT_ROOT / "configs" / "competitions.yaml"
DATA_DIR     = PROJECT_ROOT / "data" / "raw" / "eurobridge"
CARDS_LOG    = PROJECT_ROOT / "logs" / "cards_scrape_log.csv"

# ── Logging ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_config() -> list[dict]:
    with open(CONFIG_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)["competitions"]


def already_fetched(competition: str, category: str, round_num: int,
                    board_num: int, log_path: Path) -> bool:
    """Return True if this board was already successfully fetched."""
    if not log_path.exists():
        return False
    with open(log_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("competition") == competition
                    and row.get("category") == category
                    and int(row.get("round", -1)) == round_num
                    and int(row.get("board", -1)) == board_num
                    and row.get("status") == "ok"):
                return True
    return False


def write_log(log_path: Path, row: dict):
    """Append one row to the audit log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["timestamp", "competition", "category",
                           "round", "board", "status", "note"]
        )
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def get_existing_rounds(matches_file: Path) -> list[int]:
    """
    Read matches.csv and return a sorted list of unique round numbers.
    This tells us which rounds actually have data (no need to guess).
    """
    if not matches_file.exists():
        return []
    df = pd.read_csv(matches_file, usecols=["round"])
    rounds = sorted(df["round"].dropna().unique().astype(int).tolist())
    return rounds


def get_boards_per_round(matches_file: Path) -> dict[int, list[int]]:
    """
    Read matches.csv and return {round_num: [board_numbers]} for all rounds.

    Many tournaments use board numbers 17-32 for even rounds (or second
    segments). The old code hardcoded boards 1-16, missing half the data.
    """
    if not matches_file.exists():
        return {}
    df = pd.read_csv(matches_file, usecols=["round", "board"])
    result = {}
    for round_num, group in df.groupby("round"):
        result[int(round_num)] = sorted(
            group["board"].dropna().unique().astype(int).tolist()
        )
    return result


def run(competition_filter=None, category_filter=None,
        delay: float = 0.8, max_rounds: int = None):
    """
    Main function: fetch card holdings for all boards in all competitions.
    """
    config = load_config()
    total_boards_fetched = 0
    total_requests = 0

    for comp in config:
        name = comp["name"]
        if competition_filter and name not in competition_filter:
            continue

        microsite = comp["microsite"].rstrip("/")
        boards_per_round = comp.get("boards_per_round", 16)

        log.info("=" * 60)
        log.info(f"Competition: {name}")

        scraper = EuroBridgeScraper(delay=delay,
                                    base_url=f"{microsite}/Asp")

        for category, tournament_id in comp["tournaments"].items():
            if category_filter and category not in category_filter:
                continue

            log.info(f"  ── Category: {category}  (tournid={tournament_id})")

            out_dir    = DATA_DIR / name / category
            cards_file = out_dir / "cards.csv"
            matches_file = out_dir / "matches.csv"

            # Find which rounds + boards already have match data
            existing_rounds = get_existing_rounds(matches_file)
            if not existing_rounds:
                log.info(f"    No matches.csv found — skipping (run bulk scraper first)")
                continue

            # Get the actual board numbers per round from matches.csv
            # (instead of hardcoding 1-16 — many rounds use 17-32!)
            boards_map = get_boards_per_round(matches_file)

            log.info(f"    Found {len(existing_rounds)} rounds with match data: {existing_rounds}")

            # Optionally limit rounds (useful for testing)
            rounds_to_process = existing_rounds
            if max_rounds:
                rounds_to_process = existing_rounds[:max_rounds]

            category_rows = []

            for round_num in rounds_to_process:
                actual_boards = boards_map.get(round_num,
                                               list(range(1, boards_per_round + 1)))
                log.info(f"    Round {round_num}: fetching {len(actual_boards)} boards "
                         f"({min(actual_boards)}-{max(actual_boards)})...")

                for board_num in actual_boards:

                    # Skip if already fetched
                    if already_fetched(name, category, round_num, board_num, CARDS_LOG):
                        log.debug(f"      Board {board_num}: already fetched, skipping")
                        continue

                    # Build the board_code: "001.01..2513"
                    board_code = f"{board_num:03d}.{round_num:02d}..{tournament_id}"

                    try:
                        ba = scraper.get_board_across(board_code)
                        total_requests += 1

                        # Extract just the card holdings (not the per-table results —
                        # those are already in matches.csv)
                        card_row = {
                            "competition":       name,
                            "category":          category,
                            "tournament_id":     tournament_id,
                            "round":             round_num,
                            "board":             board_num,
                            "dealer":            ba.dealer,
                            "vulnerability":     ba.vulnerability,
                            # North
                            "north_spades":      ba.north_spades,
                            "north_hearts":      ba.north_hearts,
                            "north_diamonds":    ba.north_diamonds,
                            "north_clubs":       ba.north_clubs,
                            # South
                            "south_spades":      ba.south_spades,
                            "south_hearts":      ba.south_hearts,
                            "south_diamonds":    ba.south_diamonds,
                            "south_clubs":       ba.south_clubs,
                            # East
                            "east_spades":       ba.east_spades,
                            "east_hearts":       ba.east_hearts,
                            "east_diamonds":     ba.east_diamonds,
                            "east_clubs":        ba.east_clubs,
                            # West
                            "west_spades":       ba.west_spades,
                            "west_hearts":       ba.west_hearts,
                            "west_diamonds":     ba.west_diamonds,
                            "west_clubs":        ba.west_clubs,
                        }
                        category_rows.append(card_row)
                        total_boards_fetched += 1

                        write_log(CARDS_LOG, {
                            "timestamp":   datetime.now().isoformat(),
                            "competition": name,
                            "category":    category,
                            "round":       round_num,
                            "board":       board_num,
                            "status":      "ok",
                            "note":        f"dealer={ba.dealer} vul={ba.vulnerability}",
                        })
                        log.info(f"      ✓ Board {board_num:2d}: "
                                 f"dealer={ba.dealer or '?'} "
                                 f"vul={ba.vulnerability or '?'} "
                                 f"N:{ba.north_spades}.{ba.north_hearts}.{ba.north_diamonds}.{ba.north_clubs}")

                    except Exception as e:
                        log.warning(f"      ✗ Board {board_num} failed: {e}")
                        write_log(CARDS_LOG, {
                            "timestamp":   datetime.now().isoformat(),
                            "competition": name,
                            "category":    category,
                            "round":       round_num,
                            "board":       board_num,
                            "status":      "error",
                            "note":        str(e)[:200],
                        })

            # Save cards for this competition+category
            if category_rows:
                out_dir.mkdir(parents=True, exist_ok=True)
                df_cards = pd.DataFrame(category_rows)

                # Append to existing cards.csv (or create new)
                if cards_file.exists():
                    existing = pd.read_csv(cards_file)
                    df_cards = pd.concat([existing, df_cards], ignore_index=True)
                    df_cards.drop_duplicates(
                        subset=["competition", "category", "round", "board"],
                        inplace=True
                    )

                df_cards.to_csv(cards_file, index=False, encoding="utf-8-sig")
                log.info(f"    Saved {len(df_cards)} card rows → {cards_file}")

    log.info("=" * 60)
    log.info(f"DONE.")
    log.info(f"  Total BoardAcross pages fetched: {total_requests}")
    log.info(f"  Total boards with cards saved:   {total_boards_fetched}")
    log.info(f"  Log: {CARDS_LOG}")
    log.info("")
    log.info("Next step: run src/pipeline.py to join cards + matches into one dataset.")


# ── CLI ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch card holdings for all boards already scraped by the bulk scraper."
    )
    parser.add_argument("--competitions", nargs="*",
                        help="Limit to specific competition names (default: all)")
    parser.add_argument("--categories", nargs="*",
                        help="Limit to: Open Women Senior Mixed (default: all)")
    parser.add_argument("--delay", type=float, default=0.8,
                        help="Seconds between requests (default: 0.8)")
    parser.add_argument("--max-rounds", type=int, default=None,
                        help="Max rounds per category (useful for testing, e.g. --max-rounds 2)")
    args = parser.parse_args()

    run(
        competition_filter=args.competitions,
        category_filter=args.categories,
        delay=args.delay,
        max_rounds=args.max_rounds,
    )
