"""
Test: Player Name Extraction
=============================
Run this to verify that the scraper now correctly extracts player names.
Tests against match 138742 (FERM vs MSV, Poznan 2025 Mixed).

Expected result based on the screenshot:
  Open Room:   N=BRINK Sjoert, W=LANTARON Luis, E=SAINZ DE VICUNA Maria, S=FERM Barbara
  Closed Room: N=MEDIERO Marina, W=GRONKVIST Ida, E=MANNO Andrea, S=WASIK Arturo

Usage:
    python src/tools/test_player_extraction.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from downloaders.eurobridge_scraper import EuroBridgeScraper

MATCH_ID = 138742
BASE_URL = "https://db.eurobridge.org/repository/competitions/25Poznan/microsite/Asp"

print("=" * 60)
print(f"Testing player extraction for match {MATCH_ID}")
print(f"URL: {BASE_URL}/BoardDetails.asp?qmatchid={MATCH_ID}")
print("=" * 60)

scraper = EuroBridgeScraper(delay=0.5, base_url=BASE_URL)
match = scraper.get_match_details(MATCH_ID)

print(f"\nMatch: {match.home_team} vs {match.visiting_team}")
print(f"Round: {match.round_num}  |  VP: {match.home_vp} – {match.visiting_vp}")
print(f"IMP:   {match.home_imp} – {match.visiting_imp}")
print(f"Boards scraped: {len(match.boards)}")

print("\n── Open Room Players ──────────────────────")
print(f"  North:  {match.open_north  or '(not found)'}")
print(f"  South:  {match.open_south  or '(not found)'}")
print(f"  East:   {match.open_east   or '(not found)'}")
print(f"  West:   {match.open_west   or '(not found)'}")

print("\n── Closed Room Players ────────────────────")
print(f"  North:  {match.closed_north or '(not found)'}")
print(f"  South:  {match.closed_south or '(not found)'}")
print(f"  East:   {match.closed_east  or '(not found)'}")
print(f"  West:   {match.closed_west  or '(not found)'}")

# ── Verify against expected ──────────────────────────────────────────────────
expected = {
    "open_north":   "BRINK Sjoert",
    "open_west":    "LANTARON Luis",
    "open_east":    "SAINZ DE VICUNA Maria",
    "open_south":   "FERM Barbara",
    "closed_north": "MEDIERO Marina",
    "closed_west":  "GRONKVIST Ida",
    "closed_east":  "MANNO Andrea",
    "closed_south": "WASIK Arturo",
}

print("\n── Verification vs. screenshot ────────────")
all_ok = True
for field, expected_name in expected.items():
    actual = getattr(match, field)
    ok = expected_name.lower() in actual.lower() if actual else False
    status = "✅" if ok else "❌"
    print(f"  {status} {field}: expected '{expected_name}', got '{actual}'")
    if not ok:
        all_ok = False

print()
if all_ok:
    print("✅ ALL PLAYER NAMES EXTRACTED CORRECTLY!")
    print("   The scraper is ready. You can now re-run the bulk scraper")
    print("   and all future matches will include player names.")
else:
    print("❌ Some names didn't match — need to inspect the HTML.")
    print("   Run: python src/tools/inspect_player_html.py")
    print("   Then paste the output into the chat so we can fix the parser.")

# ── Also show what the DataFrame looks like ─────────────────────────────────
df = scraper.match_to_dataframe(match)
player_cols = ["board", "room", "declarer", "contract",
               "open_north", "open_south", "open_east", "open_west",
               "closed_north", "closed_south", "closed_east", "closed_west"]
print("\n── First 4 rows of DataFrame (player columns) ─")
print(df[player_cols].head(4).to_string(index=False))
print(f"\nTotal rows: {len(df)}")
print("New columns added: open_north, open_south, open_east, open_west,")
print("                   closed_north, closed_south, closed_east, closed_west")
