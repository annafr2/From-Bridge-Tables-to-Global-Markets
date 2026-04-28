"""
Quick HTML Inspector — Player Names
====================================
Run this to see how player names appear in the EuroBridge HTML.
We need to know the exact HTML structure before writing the parser.

Usage:
    python src/tools/inspect_player_html.py
"""

import sys
import requests
from bs4 import BeautifulSoup
from pathlib import Path

# The match we know has player names
URL = "https://db.eurobridge.org/repository/competitions/25Poznan/microsite/Asp/BoardDetails.asp?qmatchid=138742"

print(f"Fetching: {URL}\n")
resp = requests.get(URL, timeout=15)
resp.raise_for_status()

soup = BeautifulSoup(resp.text, "html.parser")

# ── Strategy 1: look for <td> or <div> containing player names ──────────────
# Player names are typically in UPPERCASE (like "BRINK Sjoert")
# Save the full HTML so we can search manually
out_path = Path(__file__).resolve().parents[2] / "data" / "raw" / "debug_boarddetails.html"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(resp.text, encoding="utf-8")
print(f"Full HTML saved to: {out_path}")
print("Open this file in a browser or text editor to inspect.\n")

# ── Strategy 2: find all <a> tags — player names are often links ─────────────
print("=" * 60)
print("ALL <a> tags on the page (player names are often clickable):")
print("=" * 60)
for a in soup.find_all("a"):
    text = a.get_text(strip=True)
    href = a.get("href", "")
    if text:
        print(f"  [{text}]  →  {href[:80]}")

print()
print("=" * 60)
print("Looking for compass / player layout tables:")
print("=" * 60)

# ── Strategy 3: look for tables with N/W/E/S compass structure ──────────────
for i, table in enumerate(soup.find_all("table")):
    text = table.get_text(" ", strip=True)
    # Look for tables that mention compass directions AND have uppercase names
    if any(d in text for d in ["North", "South", "East", "West", "NORTH", " N ", " S "]):
        print(f"\n--- Table #{i} ---")
        print(table.prettify()[:2000])

print()
print("=" * 60)
print("Searching for any element with 'player' or 'name' in class/id:")
print("=" * 60)
for tag in soup.find_all(True):
    cls = " ".join(tag.get("class", []))
    tid = tag.get("id", "")
    if any(kw in (cls + tid).lower() for kw in ["player", "name", "pair", "north", "south", "east", "west"]):
        text = tag.get_text(" ", strip=True)[:120]
        print(f"  <{tag.name} class='{cls}' id='{tid}'> → {text}")
