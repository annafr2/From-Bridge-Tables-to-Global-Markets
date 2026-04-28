"""
EuroBridge Bulk Scraper
=======================
Scrapes multiple EuroBridge competitions automatically.

Reads competition definitions from configs/competitions.yaml, then for each
competition → tournament category → round, discovers all match IDs and
downloads the board-level data using the existing EuroBridgeScraper.

Output:
    data/raw/eurobridge/<competition_name>/<category>/matches.csv
    data/raw/eurobridge/<competition_name>/<category>/boards_across/board_<N>_round_<R>.csv
    logs/scrape_log.csv   ← audit trail of every match attempted

Usage:
    python src/downloaders/eurobridge_bulk_scraper.py

    # Or from another script:
    from src.downloaders.eurobridge_bulk_scraper import BulkScraper
    scraper = BulkScraper()
    scraper.run()
"""

import sys
import csv
import time
import re
import logging
from pathlib import Path
from datetime import datetime

import requests
import yaml
import pandas as pd
from bs4 import BeautifulSoup

# So we can import eurobridge_scraper from the same folder
sys.path.insert(0, str(Path(__file__).parent))
from eurobridge_scraper import EuroBridgeScraper

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]   # collectBridgeData/
CONFIG_FILE  = PROJECT_ROOT / "configs" / "competitions.yaml"
DATA_DIR     = PROJECT_ROOT / "data" / "raw" / "eurobridge"
LOG_FILE     = PROJECT_ROOT / "logs" / "scrape_log.csv"

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_config() -> list[dict]:
    """Load competition list from YAML config file."""
    with open(CONFIG_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)["competitions"]


def already_scraped(match_id: int, log_path: Path) -> bool:
    """Return True if this match_id already appears (successfully) in the log."""
    if not log_path.exists():
        return False
    with open(log_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row.get("match_id", -1)) == match_id and row.get("status") == "ok":
                return True
    return False


def write_log(log_path: Path, row: dict):
    """Append one row to the CSV audit log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "competition", "category",
                                                "round", "match_id", "boards", "status", "note"])
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def discover_match_ids(microsite: str, tournament_id: int, round_num: int,
                       session: requests.Session, delay: float = 0.5) -> list[int]:
    """
    Fetch the RoundTeams page for a given tournament+round and return all match IDs.
    Returns an empty list if the round does not exist or has no matches.
    """
    url = f"{microsite}/Asp/RoundTeams.asp?qtournid={tournament_id}&qroundno={round_num}"
    try:
        resp = session.get(url, timeout=20)
        time.sleep(delay)
    except Exception as e:
        log.warning(f"  Request failed for {url}: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    ids = []
    for a in soup.find_all("a", href=True):
        m = re.search(r"qmatchid=(\d+)", a["href"])
        if m:
            ids.append(int(m.group(1)))
    return list(set(ids))  # deduplicate


# ──────────────────────────────────────────────────────────────────────────────
# Main Bulk Scraper
# ──────────────────────────────────────────────────────────────────────────────

class BulkScraper:
    def __init__(self, delay: float = 1.0, dry_run: bool = False):
        """
        delay:   seconds to wait between HTTP requests (be polite!)
        dry_run: if True, only discover match IDs but do not download board data
        """
        self.delay   = delay
        self.dry_run = dry_run
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0 (research-scraper)"})
        self.config  = load_config()

    def _make_scraper(self, microsite: str) -> EuroBridgeScraper:
        """Create an EuroBridgeScraper pointed at the right competition."""
        return EuroBridgeScraper(delay=self.delay, base_url=f"{microsite}/Asp")

    def run(self, competition_filter: list[str] | None = None,
            category_filter: list[str] | None = None):
        """
        Run the bulk scraper.

        competition_filter: optional list of competition names to restrict to
                            e.g. ["EBL_Herning_2024"]
        category_filter:    optional list of categories e.g. ["Open", "Mixed"]
        """
        total_matches = 0
        total_boards  = 0

        for comp in self.config:
            name = comp["name"]
            if competition_filter and name not in competition_filter:
                continue

            microsite = comp["microsite"].rstrip("/")
            year      = comp["year"]
            max_rounds = comp.get("max_rounds", 30)

            log.info(f"{'='*60}")
            log.info(f"Competition: {name}  ({year})")
            log.info(f"Microsite:   {microsite}")

            scraper = self._make_scraper(microsite)

            for category, tournament_id in comp["tournaments"].items():
                if category_filter and category not in category_filter:
                    continue

                log.info(f"  ── Category: {category}  (tournid={tournament_id})")

                # Output folder for this competition+category
                out_dir = DATA_DIR / name / category
                out_dir.mkdir(parents=True, exist_ok=True)
                matches_file = out_dir / "matches.csv"

                all_match_rows = []
                consecutive_empty = 0

                for round_num in range(1, max_rounds + 1):
                    match_ids = discover_match_ids(
                        microsite, tournament_id, round_num,
                        self.session, delay=self.delay
                    )

                    if not match_ids:
                        consecutive_empty += 1
                        log.info(f"    Round {round_num}: no matches (empty={consecutive_empty})")
                        if consecutive_empty >= 3:
                            log.info(f"    Stopping: 3 empty rounds in a row")
                            break
                        continue

                    consecutive_empty = 0
                    log.info(f"    Round {round_num}: {len(match_ids)} matches found")

                    for match_id in sorted(match_ids):
                        # Skip if already successfully downloaded
                        if already_scraped(match_id, LOG_FILE):
                            log.info(f"      Match {match_id}: already scraped, skipping")
                            continue

                        if self.dry_run:
                            log.info(f"      [DRY RUN] Would scrape match {match_id}")
                            write_log(LOG_FILE, {
                                "timestamp": datetime.now().isoformat(),
                                "competition": name, "category": category,
                                "round": round_num, "match_id": match_id,
                                "boards": 0, "status": "dry_run", "note": ""
                            })
                            continue

                        # Scrape the match
                        try:
                            log.info(f"      Scraping match {match_id}...")
                            match = scraper.get_match_details(match_id)
                            df = scraper.match_to_dataframe(match)

                            # Add metadata columns
                            df["competition"] = name
                            df["year"]        = year
                            df["category"]    = category

                            # Append to the running matches CSV (only write if non-empty)
                            if len(df) > 0:
                                write_mode = "a" if matches_file.exists() else "w"
                                header = not matches_file.exists()
                                df.to_csv(matches_file, mode=write_mode, header=header,
                                          index=False, encoding="utf-8-sig")

                            boards_scraped = len(df)
                            total_matches += 1
                            total_boards  += boards_scraped

                            write_log(LOG_FILE, {
                                "timestamp": datetime.now().isoformat(),
                                "competition": name, "category": category,
                                "round": round_num, "match_id": match_id,
                                "boards": boards_scraped, "status": "ok", "note": ""
                            })
                            log.info(f"      ✓ {boards_scraped} board-rows saved")

                        except Exception as e:
                            log.warning(f"      ✗ Match {match_id} failed: {e}")
                            write_log(LOG_FILE, {
                                "timestamp": datetime.now().isoformat(),
                                "competition": name, "category": category,
                                "round": round_num, "match_id": match_id,
                                "boards": 0, "status": "error", "note": str(e)[:200]
                            })

        log.info(f"{'='*60}")
        log.info(f"DONE. Total matches scraped: {total_matches}")
        log.info(f"      Total board-rows saved: {total_boards}")
        log.info(f"      Log file: {LOG_FILE}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EuroBridge Bulk Scraper")
    parser.add_argument("--competitions", nargs="*",
                        help="Limit to specific competition names (default: all)")
    parser.add_argument("--categories", nargs="*",
                        help="Limit to categories: Open Women Senior Mixed (default: all)")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Seconds between requests (default: 1.0)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Discover matches without downloading data")
    args = parser.parse_args()

    scraper = BulkScraper(delay=args.delay, dry_run=args.dry_run)
    scraper.run(
        competition_filter=args.competitions,
        category_filter=args.categories,
    )
