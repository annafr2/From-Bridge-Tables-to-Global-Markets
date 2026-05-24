"""
Diagnostic tool: show raw HTML structure of player seating for older competitions.
Tests Ostend 2018 (match 68316) and Budapest 2016 (match 34787).
"""
import sys
import io
import re
import requests
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from pathlib import Path
from bs4 import BeautifulSoup, NavigableString

def dump_player_area(url: str, label: str):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  URL: {url}")
    print(f"{'='*60}")

    resp = requests.get(url, timeout=20,
                        headers={"User-Agent": "Mozilla/5.0 (research-scraper)"})
    soup = BeautifulSoup(resp.text, "html.parser")

    # Find all links with person.asp
    person_links = soup.find_all("a", href=re.compile(r"person\.asp", re.IGNORECASE))
    print(f"\n[1] Found {len(person_links)} person.asp links:")
    for a in person_links:
        print(f"    href={a['href']!r}  text={a.get_text(strip=True)!r}")

    # Find all tables containing person.asp links
    print("\n[2] Tables containing player links:")
    for i, table in enumerate(soup.find_all("table")):
        if not table.find("a", href=re.compile(r"person\.asp", re.IGNORECASE)):
            continue
        rows = table.find_all("tr", recursive=False)
        print(f"\n  Table #{i}: {len(rows)} rows (recursive=False)")
        for ri, row in enumerate(rows):
            tds = row.find_all(["td", "th"], recursive=False)
            print(f"    Row {ri}: {len(tds)} cells")
            for ci, td in enumerate(tds):
                links = td.find_all("a", href=re.compile(r"person\.asp", re.IGNORECASE))
                text = td.get_text(" ", strip=True)[:80]
                has_links = f"  << PLAYER LINKS: {[a.get_text(strip=True) for a in links]}" if links else ""
                print(f"      Cell[{ci}]: {text!r}{has_links}")

    # Also look for span classes like SpanStyleHomeTeam
    print("\n[3] Spans with compass-related classes:")
    for span in soup.find_all("span", class_=True):
        cls = " ".join(span.get("class", []))
        if any(k in cls.lower() for k in ["home", "visit", "north", "south", "east", "west", "open", "close"]):
            links = span.find_all("a", href=re.compile(r"person\.asp", re.IGNORECASE))
            if links:
                print(f"    class={cls!r}  players={[a.get_text(strip=True) for a in links]}")

    # Show raw HTML snippet around first person link
    if person_links:
        first = person_links[0]
        # Walk up to find the containing table row
        parent = first.parent
        for _ in range(8):
            if parent and parent.name == "tr":
                break
            parent = parent.parent if parent else None
        if parent and parent.name == "tr":
            print(f"\n[4] Raw HTML of row containing first player link:")
            print(f"    {str(parent)[:500]}")

if __name__ == "__main__":
    dump_player_area(
        "http://db.eurobridge.org/Repository/competitions/18Ostend/microSite/Asp/BoardDetails.asp?qmatchid=68316",
        "Ostend 2018 — match 68316"
    )
    dump_player_area(
        "http://db.eurobridge.org/Repository/competitions/16Budapest/microSite/Asp/BoardDetails.asp?qmatchid=34787",
        "Budapest 2016 — match 34787"
    )
    dump_player_area(
        "http://db.eurobridge.org/Repository/competitions/22Madeira/microsite/Asp/BoardDetails.asp?qmatchid=100820",
        "Madeira 2022 — match 100820 (for comparison)"
    )
