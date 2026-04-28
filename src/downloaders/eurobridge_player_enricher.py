"""
EuroBridge Player Name Enricher
================================
Adds player names (N/E/S/W for Open and Closed rooms) to an existing board-level
CSV that was produced by the bulk scraper + cards scraper pipeline.

Enrichment strategy
-------------------
Each match played on EuroBridge has a BoardDetails page
    BoardDetails.asp?qmatchid=<MATCH_ID>
that contains a visual bridge-table seating diagram for both rooms.

HTML structure of the player section (4-row table inside the page):

    Row 0 – headers:  "Open Room" (colspan=3) | separator(rowspan=4) | "Closed Room" (colspan=3)
    Row 1 – North:    Open_N link (colspan=3)                        | Closed_N link (colspan=3)
    Row 2 – W/E:      Open_W | <board3.gif> | Open_E                 | Closed_W | <board3.gif> | Closed_E
    Row 3 – South:    Open_S link (colspan=3)                        | Closed_S link (colspan=3)

The separator column has rowspan=4 so it appears only in Row 0 of the raw HTML;
BeautifulSoup sees 2 cells in Row 1 and Row 3, and 6 cells in Row 2.

Parsing algorithm:
  1. Locate the player table by finding <b>Open Room</b> inside it.
  2. Collect all <tr> rows that contain person links.
  3. North row = first row with person links and no board3.gif → links[0]=Open_N, links[1]=Closed_N
  4. W/E row  = row that contains board3.gif images
                → find board3.gif positions, use adjacent cells for W (before) and E (after)
  5. South row = last row with person links and no board3.gif → links[0]=Open_S, links[1]=Closed_S
  6. Fallback (8-link index): if structural parse fails, use fixed index ordering:
     [Open_N, Closed_N, Open_W, Open_E, Closed_W, Closed_E, Open_S, Closed_S]

Output columns added to the CSV
---------------------------------
north_player, north_id, east_player, east_id,
south_player, south_id, west_player, west_id,
ns_pair, ew_pair,
enrichment_source,       # "board_details" | "inferred" | "missing"
enrichment_confidence,   # "high" | "medium" | "low"
board_details_url

Integration into the bulk pipeline
------------------------------------
After running eurobridge_bulk_scraper.py and eurobridge_cards_scraper.py:

    python src/downloaders/eurobridge_player_enricher.py \\
        --input  data/raw/eurobridge/EBL_Poznan_2025/Open/matches.csv \\
        --output data/raw/eurobridge/EBL_Poznan_2025/Open/matches_with_players.csv \\
        --base-url https://db.eurobridge.org/repository/competitions/25Poznan/microsite/Asp \\
        --cache-dir data/cache/board_details

The output CSV preserves all original columns and adds the player columns above.
Pass it directly to src/pipeline.py (edit the input path) for the final joined dataset.
"""

import re
import sys
import time
import logging
import argparse
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup
import pandas as pd
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

sys.path.insert(0, str(Path(__file__).parent))
from eurobridge_scraper import clean_text

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Types ──────────────────────────────────────────────────────────────────────
# (name, eurobridge_person_id)
PlayerEntry = tuple[str, Optional[int]]
RoomPlayers = dict[str, PlayerEntry]   # keys: "N", "E", "S", "W"

EMPTY_PLAYER: PlayerEntry = ("", None)
EMPTY_ROOM: RoomPlayers = {"N": EMPTY_PLAYER, "E": EMPTY_PLAYER, "S": EMPTY_PLAYER, "W": EMPTY_PLAYER}


# ══════════════════════════════════════════════════════════════════════════════
# Core parser
# ══════════════════════════════════════════════════════════════════════════════

def _extract_player(tag) -> PlayerEntry:
    """Return (name, player_id) from a <a href='…person?qryid=N'> tag."""
    if tag is None:
        return EMPTY_PLAYER
    href = tag.get("href", "")
    m = re.search(r"qryid=(\d+)", href)
    player_id = int(m.group(1)) if m else None
    name = clean_text(tag.get_text())
    return (name, player_id)


def _person_links_in(tag) -> list:
    """All <a> tags linking to eurobridge person pages inside *tag*."""
    return tag.find_all("a", href=re.compile(r"person\?qryid=\d+"))


def _has_board3(tag) -> bool:
    return bool(tag.find("img", src=re.compile(r"board3\.gif", re.IGNORECASE)))


def parse_board_details_players(html: str) -> dict:
    """
    Parse the player seating diagram from a BoardDetails HTML page.

    Returns
    -------
    {
        "open_room":   {"N": (name, id), "E": ..., "S": ..., "W": ...},
        "closed_room": {"N": (name, id), "E": ..., "S": ..., "W": ...},
        "raw_snippet": str,      # first 3000 chars of the player table HTML
    }
    Empty strings / None are used when data is unavailable.
    """
    import copy
    soup = BeautifulSoup(html, "html.parser")

    result = {
        "open_room":   copy.deepcopy(EMPTY_ROOM),
        "closed_room": copy.deepcopy(EMPTY_ROOM),
        "raw_snippet": "",
    }

    # ── Step 1: find the player table ────────────────────────────────────────
    # "Closed Room" in the raw HTML has whitespace: "Closed \r\n\t... Room"
    # so we normalise before checking.
    player_table = None
    for bold in soup.find_all("b"):
        if "Open Room" in bold.get_text():
            td = bold.find_parent("td")
            if td:
                tr = td.find_parent("tr")
                if tr:
                    candidate = tr.find_parent("table")
                    if candidate:
                        normalised = " ".join(candidate.get_text().split())
                        if "Closed Room" in normalised:
                            player_table = candidate
                            break

    if player_table is None:
        log.debug("Player table not found in page")
        return result

    result["raw_snippet"] = str(player_table)[:3000]

    rows = player_table.find_all("tr", recursive=False)

    # ── Step 2: categorise rows by content ───────────────────────────────────
    north_row = None
    we_row = None
    south_row = None

    for row in rows:
        links = _person_links_in(row)
        if not links:
            continue
        if _has_board3(row):
            we_row = row
        elif north_row is None:
            north_row = row
        else:
            south_row = row

    # ── Step 3: parse North row ───────────────────────────────────────────────
    if north_row is not None:
        n_links = _person_links_in(north_row)
        if len(n_links) >= 1:
            result["open_room"]["N"] = _extract_player(n_links[0])
        if len(n_links) >= 2:
            result["closed_room"]["N"] = _extract_player(n_links[1])

    # ── Step 4: parse W/E row ─────────────────────────────────────────────────
    if we_row is not None:
        cells = we_row.find_all("td", recursive=False)
        # Find cells that contain board3.gif (these are the compass-image cells)
        board3_cell_indices = [
            i for i, cell in enumerate(cells)
            if cell.find("img", src=re.compile(r"board3\.gif", re.IGNORECASE))
        ]

        if len(board3_cell_indices) >= 2:
            open_gif_idx, closed_gif_idx = board3_cell_indices[0], board3_cell_indices[1]
            # Open West is in the cell just before the first board3 cell
            if open_gif_idx >= 1:
                result["open_room"]["W"] = _extract_player(
                    cells[open_gif_idx - 1].find("a", href=re.compile(r"person\?qryid="))
                )
            # Open East is in the cell just after the first board3 cell
            if open_gif_idx + 1 < len(cells):
                result["open_room"]["E"] = _extract_player(
                    cells[open_gif_idx + 1].find("a", href=re.compile(r"person\?qryid="))
                )
            # Closed West is in the cell just before the second board3 cell
            if closed_gif_idx >= 1:
                result["closed_room"]["W"] = _extract_player(
                    cells[closed_gif_idx - 1].find("a", href=re.compile(r"person\?qryid="))
                )
            # Closed East is in the cell just after the second board3 cell
            if closed_gif_idx + 1 < len(cells):
                result["closed_room"]["E"] = _extract_player(
                    cells[closed_gif_idx + 1].find("a", href=re.compile(r"person\?qryid="))
                )

        elif len(board3_cell_indices) == 1:
            # Only one room visible — apply W/E to open room
            gif_idx = board3_cell_indices[0]
            we_links = _person_links_in(we_row)
            if gif_idx >= 1:
                result["open_room"]["W"] = _extract_player(
                    cells[gif_idx - 1].find("a", href=re.compile(r"person\?qryid="))
                )
            if gif_idx + 1 < len(cells):
                result["open_room"]["E"] = _extract_player(
                    cells[gif_idx + 1].find("a", href=re.compile(r"person\?qryid="))
                )
        else:
            # No board3.gif found in the W/E row — fall back to link order
            we_links = _person_links_in(we_row)
            if len(we_links) >= 4:
                result["open_room"]["W"]   = _extract_player(we_links[0])
                result["open_room"]["E"]   = _extract_player(we_links[1])
                result["closed_room"]["W"] = _extract_player(we_links[2])
                result["closed_room"]["E"] = _extract_player(we_links[3])

    # ── Step 5: parse South row ───────────────────────────────────────────────
    if south_row is not None:
        s_links = _person_links_in(south_row)
        if len(s_links) >= 1:
            result["open_room"]["S"] = _extract_player(s_links[0])
        if len(s_links) >= 2:
            result["closed_room"]["S"] = _extract_player(s_links[1])

    # ── Step 6: 8-link index fallback if structural parse left gaps ───────────
    # Index order from live inspection: [Open_N, Closed_N, Open_W, Open_E,
    #                                    Closed_W, Closed_E, Open_S, Closed_S]
    open_complete = all(result["open_room"][p][0] for p in "NESW")
    closed_complete = all(result["closed_room"][p][0] for p in "NESW")

    if not (open_complete and closed_complete):
        all_links = _person_links_in(player_table)
        if len(all_links) == 8:
            idx_map = [
                ("open_room",   "N", 0),
                ("closed_room", "N", 1),
                ("open_room",   "W", 2),
                ("open_room",   "E", 3),
                ("closed_room", "W", 4),
                ("closed_room", "E", 5),
                ("open_room",   "S", 6),
                ("closed_room", "S", 7),
            ]
            for room, pos, idx in idx_map:
                if not result[room][pos][0]:  # only fill gaps
                    result[room][pos] = _extract_player(all_links[idx])

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Fetching & caching
# ══════════════════════════════════════════════════════════════════════════════

def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (research-scraper)"})
    return s


def _board_details_url(base_url: str, match_id: int) -> str:
    return f"{base_url}/BoardDetails.asp?qmatchid={match_id}"


def _fetch_or_cache(
    match_id: int,
    base_url: str,
    cache_dir: Path,
    session: requests.Session,
    delay: float,
) -> str:
    """Return HTML for the BoardDetails page, using disk cache when available."""
    cache_file = cache_dir / f"{match_id}.html"
    if cache_file.exists():
        return cache_file.read_text(encoding="iso-8859-1")

    url = _board_details_url(base_url, match_id)
    resp = session.get(url, timeout=30)
    resp.encoding = "iso-8859-1"
    html = resp.text

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(html, encoding="iso-8859-1")
    time.sleep(delay)
    return html


def _confidence(room: RoomPlayers) -> str:
    """Return 'high' if all 4 positions found, 'medium' if partial, 'low' if none."""
    filled = sum(1 for p in "NESW" if room[p][0])
    if filled == 4:
        return "high"
    if filled > 0:
        return "medium"
    return "low"


# ══════════════════════════════════════════════════════════════════════════════
# Main enrichment function
# ══════════════════════════════════════════════════════════════════════════════

def enrich_boards_with_players(
    input_csv: Path,
    match_ids: list[int],
    base_url: str,
    output_csv: Path,
    cache_dir: Path,
    delay: float = 0.5,
    dry_run: bool = False,
) -> None:
    """
    Read *input_csv*, add player name columns, write *output_csv*.

    Parameters
    ----------
    input_csv   : path to the board-level CSV (must have 'match_id' and 'room' columns)
    match_ids   : list of qmatchid values to enrich (must appear in the CSV)
    base_url    : competition microsite Asp folder, e.g.
                  https://db.eurobridge.org/repository/competitions/25Poznan/microsite/Asp
    output_csv  : destination path for enriched CSV
    cache_dir   : directory for caching BoardDetails HTML pages
    delay       : seconds to wait between live HTTP requests
    dry_run     : if True, report counts but do not fetch or write
    """
    df = pd.read_csv(input_csv, encoding="utf-8-sig")

    if "match_id" not in df.columns:
        log.error(
            "Input CSV has no 'match_id' column — cannot map rows to BoardDetails pages.\n"
            "Run the bulk scraper first; it adds match_id to every row."
        )
        sys.exit(1)

    if "room" not in df.columns:
        log.error("Input CSV has no 'room' column (expected 'Open' or 'Closed').")
        sys.exit(1)

    # Determine which match_ids actually appear in the CSV
    csv_match_ids = set(df["match_id"].dropna().astype(int).unique())
    target_ids = sorted(set(match_ids) & csv_match_ids)
    log.info(f"match_ids requested: {len(match_ids)} | found in CSV: {len(target_ids)}")

    if dry_run:
        already_cached = sum(1 for mid in target_ids if (cache_dir / f"{mid}.html").exists())
        log.info(
            f"DRY RUN — {len(target_ids)} matches to enrich, "
            f"{already_cached} already cached, "
            f"{len(target_ids) - already_cached} would require HTTP requests."
        )
        return

    # ── Fetch & parse all BoardDetails pages ─────────────────────────────────
    session = _make_session()
    match_players: dict[int, dict] = {}  # match_id → parse result

    ids_iter = target_ids
    if HAS_TQDM:
        ids_iter = tqdm(target_ids, desc="Fetching BoardDetails", unit="match")

    fetched = skipped = parse_errors = 0
    for mid in ids_iter:
        cache_file = cache_dir / f"{mid}.html"
        cached = cache_file.exists()
        try:
            html = _fetch_or_cache(mid, base_url, cache_dir, session, delay)
            parsed = parse_board_details_players(html)
            match_players[mid] = parsed
            if cached:
                skipped += 1
            else:
                fetched += 1
        except Exception as exc:
            log.warning(f"  match {mid}: fetch/parse error — {exc}")
            match_players[mid] = {
                "open_room": {k: EMPTY_PLAYER for k in "NESW"},
                "closed_room": {k: EMPTY_PLAYER for k in "NESW"},
                "raw_snippet": "",
            }
            parse_errors += 1

    log.info(
        f"BoardDetails pages: {fetched} fetched live, "
        f"{skipped} loaded from cache, {parse_errors} errors"
    )

    # ── Build player columns per row ──────────────────────────────────────────
    new_cols = [
        "north_player", "north_id",
        "east_player",  "east_id",
        "south_player", "south_id",
        "west_player",  "west_id",
        "ns_pair", "ew_pair",
        "enrichment_source",
        "enrichment_confidence",
        "board_details_url",
    ]
    for col in new_cols:
        if col not in df.columns:
            df[col] = ""

    rows_enriched = rows_missing = 0

    for idx, row in df.iterrows():
        mid = int(row["match_id"]) if pd.notna(row["match_id"]) else None
        if mid is None or mid not in match_players:
            df.at[idx, "enrichment_source"] = "missing"
            df.at[idx, "enrichment_confidence"] = "low"
            rows_missing += 1
            continue

        parsed = match_players[mid]
        room_key = "open_room" if str(row.get("room", "")).strip().lower() == "open" else "closed_room"
        room_data: RoomPlayers = parsed[room_key]

        n_name, n_id = room_data["N"]
        e_name, e_id = room_data["E"]
        s_name, s_id = room_data["S"]
        w_name, w_id = room_data["W"]

        df.at[idx, "north_player"] = n_name
        df.at[idx, "north_id"]     = str(n_id) if n_id is not None else ""
        df.at[idx, "east_player"]  = e_name
        df.at[idx, "east_id"]      = str(e_id) if e_id is not None else ""
        df.at[idx, "south_player"] = s_name
        df.at[idx, "south_id"]     = str(s_id) if s_id is not None else ""
        df.at[idx, "west_player"]  = w_name
        df.at[idx, "west_id"]      = str(w_id) if w_id is not None else ""

        if n_name and s_name:
            df.at[idx, "ns_pair"] = f"{n_name} / {s_name}"
        if e_name and w_name:
            df.at[idx, "ew_pair"] = f"{e_name} / {w_name}"

        confidence = _confidence(room_data)
        df.at[idx, "enrichment_source"]     = "board_details" if confidence != "low" else "missing"
        df.at[idx, "enrichment_confidence"] = confidence
        df.at[idx, "board_details_url"]     = _board_details_url(base_url, mid)

        if confidence != "low":
            rows_enriched += 1
        else:
            rows_missing += 1

    # ── Write output ──────────────────────────────────────────────────────────
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    log.info("=" * 60)
    log.info(f"DONE enrichment summary:")
    log.info(f"  Total rows       : {len(df)}")
    log.info(f"  Rows enriched    : {rows_enriched}")
    log.info(f"  Rows missing data: {rows_missing}")
    log.info(f"  Output written   : {output_csv}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Enrich a board-level CSV with player names scraped from "
            "EuroBridge BoardDetails pages."
        )
    )
    parser.add_argument(
        "--input", required=True, type=Path,
        help="Path to input matches CSV (must have match_id and room columns).",
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="Path to write the enriched output CSV.",
    )
    parser.add_argument(
        "--base-url", required=True,
        help=(
            "EuroBridge competition microsite Asp folder, e.g. "
            "https://db.eurobridge.org/repository/competitions/25Poznan/microsite/Asp"
        ),
    )
    parser.add_argument(
        "--cache-dir", required=True, type=Path,
        help="Directory for caching BoardDetails HTML pages.",
    )
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="Seconds between live HTTP requests (default: 0.5).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report how many matches need enrichment but do not fetch or write.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        log.error(f"Input file not found: {args.input}")
        sys.exit(1)

    # Derive match_ids from the CSV itself — no need to pass them separately
    df_preview = pd.read_csv(args.input, encoding="utf-8-sig", usecols=["match_id"])
    match_ids = sorted(df_preview["match_id"].dropna().astype(int).unique().tolist())
    log.info(f"Unique match_ids in CSV: {len(match_ids)}")

    enrich_boards_with_players(
        input_csv=args.input,
        match_ids=match_ids,
        base_url=args.base_url,
        output_csv=args.output,
        cache_dir=args.cache_dir,
        delay=args.delay,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
