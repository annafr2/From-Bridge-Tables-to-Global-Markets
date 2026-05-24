"""
EuroBridge Player Names Backfill
=================================
Adds player names (N/S/E/W per room) to already-scraped matches.csv files.

WHAT IT DOES:
  - Reads each existing matches.csv
  - Finds all unique match_ids
  - Fetches ONE page per match (BoardDetails.asp) — just for player names
  - Adds 8 new columns: open_north/south/east/west + closed_north/south/east/west
  - Saves the updated matches.csv (with backup)

WHY NOT RE-SCRAPE EVERYTHING:
  - We already have all bidding + card data (78K rows)
  - Player names are on the match page — ONE request per match covers ALL 16 boards
  - This is ~5-10x faster than a full re-scrape

TIMING ESTIMATE:
  ~200 matches per competition × 0.5s = ~100 seconds (~2 min) per competition
  5 competitions × 4 categories = up to 20 runs total = ~40 minutes worst case

Usage:
    # All competitions:
    python src/downloaders/eurobridge_players_backfill.py

    # One competition only:
    python src/downloaders/eurobridge_players_backfill.py --competitions EBL_Herning_2024

    # Dry run (show what would happen, don't save):
    python src/downloaders/eurobridge_players_backfill.py --dry-run
"""

import sys
import argparse
import logging
import shutil
import time
from pathlib import Path
from datetime import datetime

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eurobridge_scraper import EuroBridgeScraper

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE  = PROJECT_ROOT / "configs" / "competitions.yaml"
DATA_DIR     = PROJECT_ROOT / "data" / "raw" / "eurobridge"

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PLAYER_COLS = [
    "open_north", "open_south", "open_east", "open_west",
    "closed_north", "closed_south", "closed_east", "closed_west",
]


def load_config() -> list[dict]:
    with open(CONFIG_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)["competitions"]


def backfill_competition(comp: dict, category: str, tournament_id: int,
                         delay: float, dry_run: bool):
    """Add player names to one competition+category matches.csv."""
    name = comp["name"]
    microsite = comp["microsite"].rstrip("/")
    matches_file = DATA_DIR / name / category / "matches.csv"

    if not matches_file.exists():
        log.warning(f"  {name}/{category}: no matches.csv found, skipping")
        return

    df = pd.read_csv(matches_file, encoding="utf-8-sig")
    log.info(f"  {name}/{category}: {len(df):,} rows, {df['match_id'].nunique()} unique matches")

    # Skip if already backfilled (check if any player column has data)
    if "open_north" in df.columns and df["open_north"].notna().any():
        non_empty = df["open_north"].notna().sum()
        log.info(f"  Already partially backfilled ({non_empty:,} rows have names). Filling gaps...")
        already_done = set(df.loc[df["open_north"].notna(), "match_id"].unique())
    else:
        already_done = set()

    # Add player columns if missing, and force them to string dtype.
    # Without astype(str), columns that were saved as all-NaN get read back
    # as float64, causing "Invalid value 'NAME' for dtype float64" on assignment.
    for col in PLAYER_COLS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str).replace("nan", "")

    # Get unique match_ids that still need player data
    all_match_ids = df["match_id"].unique()
    todo_match_ids = [mid for mid in all_match_ids if mid not in already_done]

    log.info(f"  Matches to fetch: {len(todo_match_ids)} "
             f"(already done: {len(already_done)})")

    if dry_run:
        log.info(f"  DRY RUN — would fetch {len(todo_match_ids)} matches")
        return

    if not todo_match_ids:
        log.info(f"  All matches already have player data. Nothing to do.")
        return

    scraper = EuroBridgeScraper(delay=delay, base_url=f"{microsite}/Asp")

    found_count = 0
    missing_count = 0

    for i, match_id in enumerate(sorted(todo_match_ids)):
        try:
            match = scraper.get_match_details(int(match_id))

            # Check if we got any player names
            has_players = bool(match.open_north or match.closed_north)

            if has_players:
                # Update all rows for this match_id
                mask = df["match_id"] == match_id
                df.loc[mask, "open_north"]   = match.open_north
                df.loc[mask, "open_south"]   = match.open_south
                df.loc[mask, "open_east"]    = match.open_east
                df.loc[mask, "open_west"]    = match.open_west
                df.loc[mask, "closed_north"] = match.closed_north
                df.loc[mask, "closed_south"] = match.closed_south
                df.loc[mask, "closed_east"]  = match.closed_east
                df.loc[mask, "closed_west"]  = match.closed_west
                found_count += 1

                if i % 20 == 0 or i == len(todo_match_ids) - 1:
                    log.info(f"    [{i+1}/{len(todo_match_ids)}] match {match_id}: "
                             f"Open N={match.open_north} S={match.open_south} | "
                             f"Closed N={match.closed_north} S={match.closed_south}")
            else:
                missing_count += 1
                if i % 20 == 0:
                    log.debug(f"    [{i+1}] match {match_id}: no player names found "
                              f"(older format — expected for pre-2022 competitions)")

        except Exception as e:
            log.warning(f"    match {match_id}: error — {e}")
            missing_count += 1

    # Save updated CSV (with backup)
    backup_path = matches_file.with_suffix(f".backup_{datetime.now():%Y%m%d_%H%M%S}.csv")
    shutil.copy(matches_file, backup_path)
    log.info(f"  Backup saved: {backup_path.name}")

    df.to_csv(matches_file, index=False, encoding="utf-8-sig")

    rows_with_names = df["open_north"].notna() & (df["open_north"] != "")
    log.info(f"  Saved. Rows with player names: {rows_with_names.sum():,}/{len(df):,} "
             f"({100*rows_with_names.mean():.0f}%)")
    log.info(f"  Matches: found={found_count}, no-names={missing_count}")


def run(competition_filter=None, category_filter=None,
        delay: float = 0.5, dry_run: bool = False):
    config = load_config()

    log.info("=" * 60)
    log.info("PLAYER NAME BACKFILL")
    log.info("=" * 60)
    if dry_run:
        log.info("DRY RUN MODE — nothing will be saved")

    total_fetched = 0

    for comp in config:
        name = comp["name"]
        if competition_filter and name not in competition_filter:
            continue

        log.info(f"\nCompetition: {name}")
        microsite = comp["microsite"]

        for category, tournament_id in comp["tournaments"].items():
            if category_filter and category not in category_filter:
                continue

            log.info(f"  Category: {category}")
            backfill_competition(comp, category, tournament_id, delay, dry_run)

    log.info("\n" + "=" * 60)
    log.info("BACKFILL COMPLETE")
    log.info("Next step: re-run pipeline to rebuild all_matches.parquet")
    log.info("  python src/pipeline.py")
    log.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add player names to existing matches.csv files")
    parser.add_argument("--competitions", nargs="*",
                        help="Limit to specific competitions (default: all)")
    parser.add_argument("--categories", nargs="*",
                        help="Limit to: Open Women Senior Mixed (default: all)")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Seconds between requests (default: 0.5)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would happen without saving")
    args = parser.parse_args()

    run(
        competition_filter=args.competitions,
        category_filter=args.categories,
        delay=args.delay,
        dry_run=args.dry_run,
    )
