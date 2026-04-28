"""
Dump the raw HTML around player name links.
Run this once — paste the output to Claude.

Usage:
    python src/tools/dump_player_html.py
"""
import re, requests
from bs4 import BeautifulSoup

URL = "https://db.eurobridge.org/repository/competitions/25Poznan/microsite/Asp/BoardDetails.asp?qmatchid=138742"

resp = requests.get(URL, timeout=15)
resp.encoding = "iso-8859-1"
soup = BeautifulSoup(resp.text, "html.parser")

# ── 1. Find all <a> tags with eurobridge.org/person in href ────────────────
pattern = re.compile(r"eurobridge\.org/person", re.IGNORECASE)
player_links = soup.find_all("a", href=pattern)

print(f"Found {len(player_links)} player links\n")
for i, a in enumerate(player_links):
    print(f"--- Player link #{i+1} ---")
    print(f"Text : {a.get_text(strip=True)}")
    print(f"Href : {a.get('href', '')}")
    # Show grandparent HTML (the cell containing this link)
    parent = a.parent          # immediate parent (<td> or <div>)
    grandparent = parent.parent if parent else None
    print(f"Parent tag      : <{parent.name}> class={parent.get('class','')}")
    if grandparent:
        print(f"Grandparent tag : <{grandparent.name}> class={grandparent.get('class','')}")
    print()

# ── 2. Show the raw HTML of the first table that contains a player link ────
print("\n" + "="*60)
print("RAW HTML of first table containing a player link:")
print("="*60)
for table in soup.find_all("table"):
    if table.find("a", href=pattern):
        print(table.prettify()[:3000])  # first 3000 chars
        print("... [truncated if long]")
        break

# ── 3. Show all table structures (row count) that have player links ─────────
print("\n" + "="*60)
print("All tables with player links (row/col counts):")
print("="*60)
for i, table in enumerate(soup.find_all("table")):
    if table.find("a", href=pattern):
        rows = table.find_all("tr", recursive=False)
        print(f"\nTable #{i}: {len(rows)} rows")
        for j, row in enumerate(rows):
            cells = row.find_all(["td","th"], recursive=False)
            cell_texts = [c.get_text(" ", strip=True)[:40] for c in cells]
            print(f"  Row {j} ({len(cells)} cells): {cell_texts}")
