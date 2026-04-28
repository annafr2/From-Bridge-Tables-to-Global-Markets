"""
Quick Data Explorer
===================
Run this to get a first look at the collected bridge dataset.

Usage:
    python explore_data.py
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
FILE = PROJECT_ROOT / "data" / "processed" / "all_matches.parquet"

print("Loading dataset...")
df = pd.read_parquet(FILE)

print("\n" + "=" * 60)
print("BASIC INFO")
print("=" * 60)
print(f"Total rows:        {len(df):,}")
print(f"Total columns:     {len(df.columns)}")
print(f"Unique matches:    {df['match_id'].nunique():,}")

print("\n" + "=" * 60)
print("COMPETITIONS")
print("=" * 60)
for comp, count in df.groupby("competition").size().items():
    print(f"  {comp}: {count:,} rows")

print("\n" + "=" * 60)
print("CATEGORIES")
print("=" * 60)
for cat, count in df.groupby("category").size().items():
    print(f"  {cat}: {count:,} rows")

print("\n" + "=" * 60)
print("DATA QUALITY")
print("=" * 60)
print(f"  Rows with bidding:  {df['has_bidding'].sum():,}  ({100*df['has_bidding'].mean():.1f}%)")
print(f"  Rows with cards:    {df['has_cards'].sum():,}  ({100*df['has_cards'].mean():.1f}%)")
print(f"  Rows with BOTH:     {(df['has_bidding'] & df['has_cards']).sum():,}")

print("\n" + "=" * 60)
print("SAMPLE — 5 ROWS (bidding + cards only)")
print("=" * 60)
sample = df[df["has_bidding"] & df["has_cards"]][
    ["competition", "home_team", "visiting_team", "board", "room",
     "contract", "declarer", "tricks", "bidding"]
].head(5)
pd.set_option("display.max_colwidth", 60)
pd.set_option("display.width", 120)
print(sample.to_string(index=False))

print("\n" + "=" * 60)
print("COLUMN NAMES")
print("=" * 60)
for col in df.columns:
    print(f"  {col}")

print("\n" + "=" * 60)
print("Saving 100-row sample to: sample_data.csv")
print("(Open this in Excel to browse the data)")
print("(To export ALL rows run:  python explore_data.py --full-csv)")
print("=" * 60)
df[df["has_bidding"] & df["has_cards"]].head(100).to_csv(
    PROJECT_ROOT / "sample_data.csv", index=False, encoding="utf-8-sig"
)
print("Done!")

# Full CSV export (optional — large file, takes a few seconds)
import sys
if "--full-csv" in sys.argv:
    print("\nExporting full dataset to: all_matches_full.csv ...")
    df.to_csv(PROJECT_ROOT / "all_matches_full.csv", index=False, encoding="utf-8-sig")
    print(f"Done! {len(df):,} rows exported.")
