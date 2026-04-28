"""
Bridge Data Pipeline
====================
Combines all raw scraped data into one clean master dataset.

What it does:
  1. Loads all matches.csv files (bidding, contract, result, teams)
  2. Loads all cards.csv files (52 cards per board, dealer, vulnerability)
  3. Joins them together — one row per board with everything
  4. Filters out boards with missing critical data
  5. Saves to data/processed/all_matches.parquet (and .csv)

Why parquet?
  - Much faster to load than CSV for large datasets
  - Preserves column types (int, float, string)
  - Smaller file size

Usage:
    python src/pipeline.py

    # Preview without saving:
    python src/pipeline.py --dry-run

    # Save CSV too (slower, larger):
    python src/pipeline.py --save-csv
"""

import argparse
import glob
import logging
from pathlib import Path

import pandas as pd

# ── Paths ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR      = PROJECT_ROOT / "data" / "raw" / "eurobridge"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# ── Logging ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────
# Step 1: Load matches
# ────────────────────────────────────────────────────────────────────────

def load_matches() -> pd.DataFrame:
    """Load and combine all matches.csv files."""
    files = sorted(RAW_DIR.glob("**/matches.csv"))
    if not files:
        raise FileNotFoundError(f"No matches.csv files found under {RAW_DIR}")

    log.info(f"Found {len(files)} matches.csv files:")
    dfs = []
    for f in files:
        df = pd.read_csv(f, encoding="utf-8-sig")
        log.info(f"  {f.relative_to(PROJECT_ROOT)}: {len(df):,} rows")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    # Deduplicate: same match + board + room should appear only once
    before = len(combined)
    combined.drop_duplicates(subset=["match_id", "board", "room"], inplace=True)
    dupes = before - len(combined)
    if dupes:
        log.info(f"  Removed {dupes:,} duplicate rows")

    log.info(f"  Total matches rows: {len(combined):,}")
    return combined


# ────────────────────────────────────────────────────────────────────────
# Step 2: Load cards
# ────────────────────────────────────────────────────────────────────────

def load_cards() -> pd.DataFrame:
    """Load and combine all cards.csv files."""
    files = sorted(RAW_DIR.glob("**/cards.csv"))
    if not files:
        log.warning("No cards.csv files found — card columns will be empty in output")
        return pd.DataFrame()

    log.info(f"Found {len(files)} cards.csv files:")
    dfs = []
    for f in files:
        df = pd.read_csv(f, encoding="utf-8-sig")
        log.info(f"  {f.relative_to(PROJECT_ROOT)}: {len(df):,} rows")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    # Drop boards where cards are empty (even rounds / rounds without card display)
    card_cols = ["north_spades", "north_hearts", "north_diamonds", "north_clubs"]
    before = len(combined)
    combined = combined[
        combined["north_spades"].notna() &
        (combined["north_spades"].str.strip() != "")
    ]
    dropped = before - len(combined)
    if dropped:
        log.info(f"  Dropped {dropped:,} boards with empty card data (expected — even rounds)")

    # Deduplicate cards: same competition + category + round + board
    combined.drop_duplicates(
        subset=["competition", "category", "round", "board"], inplace=True
    )

    log.info(f"  Total card rows: {len(combined):,}")
    return combined


# ────────────────────────────────────────────────────────────────────────
# Step 3: Join matches + cards
# ────────────────────────────────────────────────────────────────────────

def join_matches_and_cards(df_matches: pd.DataFrame,
                           df_cards: pd.DataFrame) -> pd.DataFrame:
    """
    Join match data with card holdings.

    KEY: EBL uses paired rounds:
      Odd rounds  (5, 7, 9...) = Open Room  → has BoardAcross card data
      Even rounds (6, 8, 10..) = Closed Room → same boards, no separate card page

    So we map even rounds to their odd partner before joining.
    Join key: competition + category + card_round + board
    (must include competition+category to avoid cross-joining different tournaments)
    """
    if df_cards.empty:
        log.warning("No card data available — skipping join")
        return df_matches

    # Map even rounds to their odd partner
    df_matches = df_matches.copy()
    df_matches["card_round"] = df_matches["round"].apply(
        lambda r: int(r) if int(r) % 2 == 1 else int(r) - 1
    )

    # Rename cards 'round' to avoid collision, keep competition+category for join
    df_cards = df_cards.rename(columns={"round": "card_round"})

    # Drop only tournament_id (not needed after join)
    if "tournament_id" in df_cards.columns:
        df_cards = df_cards.drop(columns=["tournament_id"])

    merged = df_matches.merge(
        df_cards,
        on=["competition", "category", "card_round", "board"],
        how="left"
    )

    # Report join quality
    total = len(merged)
    with_cards = merged["north_spades"].notna().sum() if "north_spades" in merged.columns else 0
    pct = 100 * with_cards / total if total > 0 else 0
    log.info(f"  Join result: {with_cards:,} / {total:,} rows have card data ({pct:.1f}%)")

    return merged


# ────────────────────────────────────────────────────────────────────────
# Step 4: Clean and validate
# ────────────────────────────────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: fix types, add quality flags, drop only truly broken rows."""

    # Ensure numeric types
    for col in ["match_id", "board", "round", "tricks", "ns_score", "ew_score",
                "home_imp", "visiting_imp", "year"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop only rows where identity fields are missing (truly broken)
    # Do NOT drop for missing contract/bidding — some tournaments scraped partially
    identity_fields = ["match_id", "board", "room"]
    before = len(df)
    df = df.dropna(subset=[c for c in identity_fields if c in df.columns])
    dropped = before - len(df)
    if dropped:
        log.info(f"  Dropped {dropped:,} rows missing identity fields (match_id/board/room)")

    # Add quality flag: has_bidding — useful for filtering in analysis
    if "bidding" in df.columns:
        df["has_bidding"] = df["bidding"].notna() & (df["bidding"].str.strip() != "")
        pct = 100 * df["has_bidding"].mean()
        log.info(f"  Rows with bidding data: {df['has_bidding'].sum():,} ({pct:.1f}%)")

    # Add quality flag: has_cards
    if "north_spades" in df.columns:
        df["has_cards"] = df["north_spades"].notna() & (df["north_spades"].str.strip() != "")
        pct = 100 * df["has_cards"].mean()
        log.info(f"  Rows with card data:    {df['has_cards'].sum():,} ({pct:.1f}%)")

    # Drop the helper column
    if "card_round" in df.columns:
        df = df.drop(columns=["card_round"])

    log.info(f"  Final dataset: {len(df):,} rows, {len(df.columns)} columns")
    return df


# ────────────────────────────────────────────────────────────────────────
# Step 5: Save
# ────────────────────────────────────────────────────────────────────────

def save(df: pd.DataFrame, save_csv: bool = False):
    """Save to data/processed/."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    parquet_path = PROCESSED_DIR / "all_matches.parquet"
    df.to_parquet(parquet_path, index=False)
    size_mb = parquet_path.stat().st_size / 1_000_000
    log.info(f"  Saved: {parquet_path.relative_to(PROJECT_ROOT)}  ({size_mb:.1f} MB)")

    if save_csv:
        csv_path = PROCESSED_DIR / "all_matches.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        size_mb = csv_path.stat().st_size / 1_000_000
        log.info(f"  Saved: {csv_path.relative_to(PROJECT_ROOT)}  ({size_mb:.1f} MB)")


# ────────────────────────────────────────────────────────────────────────
# Summary report
# ────────────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame):
    """Print a human-readable summary of the final dataset."""
    log.info("=" * 55)
    log.info("DATASET SUMMARY")
    log.info("=" * 55)
    log.info(f"  Total rows (board-room records): {len(df):,}")
    log.info(f"  Columns:                         {len(df.columns)}")

    if "match_id" in df.columns:
        log.info(f"  Unique matches:                  {df['match_id'].nunique():,}")
    if "competition" in df.columns:
        log.info(f"  Competitions:                    {df['competition'].nunique()}")
        for comp, count in df.groupby("competition").size().items():
            log.info(f"    {comp}: {count:,} rows")
    if "category" in df.columns:
        log.info(f"  Categories:                      {sorted(df['category'].unique())}")
    if "has_cards" in df.columns:
        pct = 100 * df["has_cards"].mean()
        log.info(f"  Rows with card data:             {pct:.1f}%")
    if "has_bidding" in df.columns:
        pct = 100 * df["has_bidding"].mean()
        log.info(f"  Rows with bidding sequence:      {pct:.1f}%")
        log.info(f"  NOTE: rows WITHOUT bidding = Madeira 2022 (scraper fix needed)")
    log.info("=" * 55)
    log.info("  Output: data/processed/all_matches.parquet")
    log.info("")
    log.info("  To load:")
    log.info("    import pandas as pd")
    log.info('    df = pd.read_parquet("data/processed/all_matches.parquet")')
    log.info("=" * 55)


# ────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────

def run(dry_run: bool = False, save_csv: bool = False):
    log.info("=" * 55)
    log.info("BRIDGE DATA PIPELINE")
    log.info("=" * 55)

    log.info("Step 1: Loading matches...")
    df_matches = load_matches()

    log.info("Step 2: Loading cards...")
    df_cards = load_cards()

    log.info("Step 3: Joining matches + cards...")
    df = join_matches_and_cards(df_matches, df_cards)

    log.info("Step 4: Cleaning...")
    df = clean(df)

    print_summary(df)

    if dry_run:
        log.info("DRY RUN — not saving. Remove --dry-run to save.")
    else:
        log.info("Step 5: Saving...")
        save(df, save_csv=save_csv)
        log.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bridge Data Pipeline")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run without saving — just show what would be produced")
    parser.add_argument("--save-csv", action="store_true",
                        help="Also save a CSV copy (in addition to parquet)")
    args = parser.parse_args()

    run(dry_run=args.dry_run, save_csv=args.save_csv)
