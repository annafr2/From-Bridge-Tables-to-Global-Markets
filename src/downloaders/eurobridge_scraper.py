"""
EuroBridge Tournament Data Scraper
===================================
Scrapes match and board data from db.eurobridge.org for research purposes.

Two main page types:
1. BoardDetails.asp?qmatchid=XXXX  - A specific match between two teams (all boards)
2. BoardAcross.asp?qboard=XXX.RR..CCCC - A specific board across all tables in a round

Usage examples:
    scraper = EuroBridgeScraper()

    # Get one match
    match = scraper.get_match_details(138742)

    # Get one board across all tables
    board = scraper.get_board_across("001.01..2513")

    # Get all matches in a round (by scraping board-across pages to find match IDs)
    all_matches = scraper.get_all_round_matches(round_num=1, category_code=2513, num_boards=10)

    # Export to CSV
    scraper.export_matches_to_csv([match], "match_138742.csv")
    scraper.export_board_across_to_csv(board, "board_001.csv")
"""

import requests
from bs4 import BeautifulSoup, NavigableString
import csv
import re
import time
import html
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path


DEFAULT_BASE_URL = "https://db.eurobridge.org/repository/competitions/25Poznan/microsite/Asp"


def clean_text(text: str) -> str:
    """Clean whitespace and &nbsp; from extracted text."""
    if not text:
        return ""
    return text.replace("\xa0", "").strip()


def safe_int(text: str) -> int:
    """Parse int from text, returning 0 if empty or non-numeric."""
    text = clean_text(text) if text else ""
    m = re.search(r"-?\d+", text)
    return int(m.group()) if m else 0


def parse_suit_symbols(tag) -> str:
    """Convert HTML suit symbols (♠♥♦♣) to text characters (S/H/D/C)."""
    if tag is None:
        return ""
    result = []
    for child in tag.children:
        if isinstance(child, NavigableString):
            result.append(str(child))
        elif child.name == "font" or child.name == "span":
            inner = child.get_text()
            # Map unicode suit symbols
            inner = inner.replace("♠", "S").replace("♥", "H").replace("♦", "D").replace("♣", "C")
            result.append(inner)
        else:
            txt = child.get_text()
            txt = txt.replace("♠", "S").replace("♥", "H").replace("♦", "D").replace("♣", "C")
            result.append(txt)
    text = "".join(result).strip().replace("\xa0", "").strip()
    # Also handle entities that might have been decoded
    text = text.replace("♠", "S").replace("♥", "H").replace("♦", "D").replace("♣", "C")
    return text


def extract_contract_from_tooltip(tooltip_tag) -> str:
    """Extract just the contract text (e.g. '4S') from tooltip <a>, ignoring the <span>."""
    if tooltip_tag is None:
        return ""
    # Get direct children text nodes and font tags, skip <span>
    parts = []
    for child in tooltip_tag.children:
        if isinstance(child, NavigableString):
            parts.append(str(child))
        elif child.name == "span":
            continue  # skip bidding tooltip
        elif child.name == "font":
            txt = child.get_text()
            txt = txt.replace("♠", "S").replace("♥", "H").replace("♦", "D").replace("♣", "C")
            parts.append(txt)
    return "".join(parts).replace("\xa0", "").strip()


def extract_bidding_from_tooltip(tooltip_tag) -> list[list[str]]:
    """
    Extract the bidding sequence from the tooltip <span> inside an <a class="tooltip">.
    Returns list of rounds, each round = [W, N, E, S].
    """
    if tooltip_tag is None:
        return []
    span = tooltip_tag.find("span")
    if span is None:
        return []
    table = span.find("table")
    if table is None:
        return []
    rows = table.find_all("tr")
    bidding = []
    for row in rows[1:]:  # skip header row
        cells = row.find_all("td")
        round_bids = []
        for cell in cells:
            bid_text = parse_suit_symbols(cell)
            round_bids.append(bid_text)
        if round_bids:
            bidding.append(round_bids)
    return bidding


def bidding_to_string(bidding: list[list[str]]) -> str:
    """Convert bidding sequence to a flat string like 'W:- N:Pass E:Pass S:1C | W:1S N:x ...'"""
    if not bidding:
        return ""
    positions = ["W", "N", "E", "S"]
    parts = []
    for rnd in bidding:
        round_parts = []
        for i, bid in enumerate(rnd):
            if i < len(positions) and bid:
                round_parts.append(f"{positions[i]}:{bid}")
        parts.append(" ".join(round_parts))
    return " | ".join(parts)


def extract_contract_and_bidding_from_cell(cell_tag) -> tuple[str, str]:
    """
    Extract the contract and bidding from a cell.
    Handles newer formats (with tooltip) and older formats (no tooltip, e.g. Madeira 2022).
    """
    if cell_tag is None:
        return "", ""
    tooltip = cell_tag.find("a", class_="tooltip")
    if tooltip:
        contract = extract_contract_from_tooltip(tooltip)
        bidding = bidding_to_string(extract_bidding_from_tooltip(tooltip))
        return contract, bidding
    
    # Fallback: Contract is inside the cell directly; no bidding tooltip available
    contract = parse_suit_symbols(cell_tag)
    return contract, ""


@dataclass
class BoardResult:
    """Result of a single board in one room."""
    board_num: int = 0
    room: str = ""  # "Open" or "Closed"
    contract: str = ""
    declarer: str = ""  # N/S/E/W
    lead: str = ""
    tricks: int = 0
    ns_score: int = 0
    ew_score: int = 0
    bidding: str = ""  # flattened bidding string


@dataclass
class MatchData:
    """Data for a full match between two teams."""
    match_id: int = 0
    round_num: str = ""
    category: str = ""
    home_team: str = ""
    visiting_team: str = ""
    home_vp: float = 0.0
    visiting_vp: float = 0.0
    home_imp: int = 0
    visiting_imp: int = 0
    boards: list = field(default_factory=list)  # list of BoardResult pairs
    # Individual player names per position (N/S/E/W) per room
    # Open Room: home team plays N/S, visiting team plays E/W (standard EBL convention)
    open_north:  str = ""
    open_south:  str = ""
    open_east:   str = ""
    open_west:   str = ""
    # Closed Room: visiting team plays N/S, home team plays E/W
    closed_north: str = ""
    closed_south: str = ""
    closed_east:  str = ""
    closed_west:  str = ""


@dataclass
class BoardAcrossEntry:
    """One table's result for a board across all tables."""
    table_num: int = 0
    home_team: str = ""
    visiting_team: str = ""
    room: str = ""
    contract: str = ""
    declarer: str = ""
    lead: str = ""
    tricks: int = 0
    ns_score: int = 0
    ew_score: int = 0
    home_result: str = ""
    visiting_result: str = ""
    bidding: str = ""
    match_id: int = 0


@dataclass
class BoardAcrossData:
    """Full board-across data: the board itself + results from all tables."""
    board_num: int = 0
    round_num: str = ""
    category: str = ""
    dealer: str = ""
    vulnerability: str = ""
    # Card holdings per direction
    north_spades: str = ""
    north_hearts: str = ""
    north_diamonds: str = ""
    north_clubs: str = ""
    south_spades: str = ""
    south_hearts: str = ""
    south_diamonds: str = ""
    south_clubs: str = ""
    east_spades: str = ""
    east_hearts: str = ""
    east_diamonds: str = ""
    east_clubs: str = ""
    west_spades: str = ""
    west_hearts: str = ""
    west_diamonds: str = ""
    west_clubs: str = ""
    entries: list = field(default_factory=list)  # list of BoardAcrossEntry


def _player_name_from_cell(cell) -> str:
    """
    Extract a player name from a table cell.

    Two href patterns exist across EuroBridge generations:
      - New (Herning 2024, Poznan 2025):  http://www.eurobridge.org/person?qryid=XXXX
      - Old (Ostend 2018, Budapest 2016, Madeira 2022): /people/person.asp?qryid=XXXX

    Both patterns appear in the same 4-row seating table layout.
    """
    a = cell.find("a", href=re.compile(r"(eurobridge\.org/person|/people/person\.asp)", re.IGNORECASE))
    if a:
        return clean_text(a.get_text())
    return ""


def _extract_player_positions(soup: BeautifulSoup, md: "MatchData"):
    """
    Extract player names (N/S/E/W, Open/Closed Room) from a BoardDetails page.

    The page contains one 4-row seating table structured as:
      Row 0: [Open Room header] [spacer col] [Closed Room header]
      Row 1: [North Open]                    [North Closed]
      Row 2: [West Open] [compass] [East Open] [West Closed] [compass] [East Closed]
      Row 3: [South Open]                    [South Closed]

    Identification: 4 rows, Row 0 contains both "Open Room" and "Closed" in text.
    """
    seating_table = None
    for table in soup.find_all("table"):
        rows = table.find_all("tr", recursive=False)
        if len(rows) != 4:
            continue
        row0_text = rows[0].get_text(" ", strip=True)
        if "Open Room" in row0_text and "Closed" in row0_text:
            seating_table = table
            break

    if seating_table is None:
        return  # Page doesn't have the expected structure (older competitions)

    rows = seating_table.find_all("tr", recursive=False)

    # Row 1: North players (2 cells — Open, Closed)
    r1 = rows[1].find_all("td", recursive=False)
    if len(r1) >= 2:
        md.open_north   = _player_name_from_cell(r1[0])
        md.closed_north = _player_name_from_cell(r1[1])

    # Row 2: West / East players (6 cells — W_open, compass, E_open, W_closed, compass, E_closed)
    r2 = rows[2].find_all("td", recursive=False)
    if len(r2) >= 6:
        md.open_west    = _player_name_from_cell(r2[0])
        md.open_east    = _player_name_from_cell(r2[2])
        md.closed_west  = _player_name_from_cell(r2[3])
        md.closed_east  = _player_name_from_cell(r2[5])

    # Row 3: South players (2 cells — Open, Closed)
    r3 = rows[3].find_all("td", recursive=False)
    if len(r3) >= 2:
        md.open_south   = _player_name_from_cell(r3[0])
        md.closed_south = _player_name_from_cell(r3[1])


class EuroBridgeScraper:
    def __init__(self, delay: float = 0.5, base_url: str = DEFAULT_BASE_URL):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (research-scraper)"
        })
        self.delay = delay  # seconds between requests
        self.base_url = base_url  # competition microsite Asp folder

    def _get(self, url: str) -> BeautifulSoup:
        resp = self.session.get(url, timeout=30)
        resp.encoding = "iso-8859-1"
        time.sleep(self.delay)
        return BeautifulSoup(resp.text, "html.parser")

    # ─────────────────────────────────────────────
    # BoardDetails  (one match, all boards)
    # ─────────────────────────────────────────────
    def get_match_details(self, match_id: int) -> MatchData:
        url = f"{self.base_url}/BoardDetails.asp?qmatchid={match_id}"
        soup = self._get(url)
        md = MatchData(match_id=match_id)

        # Round and category from header
        header_cells = soup.find_all("table")[0].find_all("td")
        for cell in header_cells:
            txt = clean_text(cell.get_text())
            if "Round" in txt:
                m = re.search(r"Round\s+(\S+)", txt)
                if m:
                    md.round_num = m.group(1)
            if "Teams" in txt or "Pairs" in txt:
                md.category = txt.strip()

        # Team names
        team_links = soup.find_all("a", href=re.compile(r"TeamDetails\.asp\?qteamid="))
        if len(team_links) >= 2:
            md.home_team = clean_text(team_links[0].get_text())
            md.visiting_team = clean_text(team_links[1].get_text())

        # VP / IMP line
        page_text = soup.get_text()
        vp_match = re.search(r"([\d.]+)\s*-\s*([\d.]+)\s*VP\s*\(\s*(\d+)\s*-\s*(\d+)\s*IMP\s*\)", page_text)
        if vp_match:
            md.home_vp = float(vp_match.group(1))
            md.visiting_vp = float(vp_match.group(2))
            md.home_imp = int(vp_match.group(3))
            md.visiting_imp = int(vp_match.group(4))

        # ── Player names from the compass seating diagrams ──────────────────
        # EuroBridge shows two compass tables (Open Room / Closed Room).
        # Each has N at top, W at left, E at right, S at bottom.
        # Player names are inside <a href="...PlayerDetails.asp?qplayerid=..."> links.
        _extract_player_positions(soup, md)

        # Board results from the highlight table
        highlight_table = soup.find("table", id="highlight")
        if highlight_table is None:
            return md

        rows = highlight_table.find_all("tr", recursive=False)
        for row in rows[1:]:  # skip header
            cells = row.find_all("td", recursive=False)
            if len(cells) < 13:
                continue

            # Board number in first cell
            board_link = cells[0].find("a")
            if board_link is None:
                continue
            board_num = int(re.search(r"\d+", clean_text(board_link.get_text())).group())

            # Open Room data (cells 1-6)
            open_contract, open_bidding = extract_contract_and_bidding_from_cell(cells[1])
            open_result = BoardResult(
                board_num=board_num,
                room="Open",
                contract=open_contract,
                declarer=clean_text(cells[2].get_text()),
                lead=parse_suit_symbols(cells[3]),
                tricks=safe_int(cells[4].get_text()),
                ns_score=safe_int(cells[5].get_text()),
                ew_score=safe_int(cells[6].get_text()),
                bidding=open_bidding,
            )

            # Closed Room data (cells 7-12)
            closed_contract, closed_bidding = extract_contract_and_bidding_from_cell(cells[7])
            closed_result = BoardResult(
                board_num=board_num,
                room="Closed",
                contract=closed_contract,
                declarer=clean_text(cells[8].get_text()),
                lead=parse_suit_symbols(cells[9]),
                tricks=safe_int(cells[10].get_text()),
                ns_score=safe_int(cells[11].get_text()),
                ew_score=safe_int(cells[12].get_text()),
                bidding=closed_bidding,
            )

            # IMP columns (cells 13-14) - home/visiting IMP for this board
            md.boards.append({"open": open_result, "closed": closed_result})

        return md

    # ─────────────────────────────────────────────
    # BoardAcross  (one board, all tables)
    # ─────────────────────────────────────────────
    def get_board_across(self, board_code: str) -> BoardAcrossData:
        """
        board_code like '001.01..2513' means board 1, round 01, category 2513.
        """
        url = f"{self.base_url}/BoardAcross.asp?qboard={board_code}"
        soup = self._get(url)
        ba = BoardAcrossData()

        # Parse board number from code
        m = re.match(r"(\d+)\.(\d+)\.\.(\d+)", board_code)
        if m:
            ba.board_num = int(m.group(1))
            ba.round_num = m.group(2)

        # Header
        header_cells = soup.find_all("table")[0].find_all("td")
        for cell in header_cells:
            txt = clean_text(cell.get_text())
            if "Teams" in txt or "Pairs" in txt:
                ba.category = txt.strip()

        # Dealer / vulnerability from board heading
        board_heading = soup.find("b", string=re.compile(r"Board\s+\d+"))
        if board_heading:
            heading_text = clean_text(board_heading.get_text())
            dealer_m = re.search(r"Dealer\s+(\w+)", heading_text)
            if dealer_m:
                ba.dealer = dealer_m.group(1)
            vul_m = re.search(r"(None|All|N-S|E-W)\s+Vulnerable", heading_text, re.IGNORECASE)
            if vul_m:
                ba.vulnerability = vul_m.group(1)

        # Card diagram - extract holdings
        # Cell 0 = heading, cells 1-4 = North, West, East, South
        brd_cells = soup.find_all("td", class_="BrdDispl")
        if len(brd_cells) >= 5:
            for idx, direction in zip([1, 2, 3, 4], ["north", "west", "east", "south"]):
                text = brd_cells[idx].get_text(separator="|")
                # Match suit symbol followed by card values
                suits = re.findall(r"[♠♣]|[♥♦]", text)
                values = re.split(r"[♠♥♦♣]", text)
                # values[0] is before first suit (empty), values[1:] are the card values
                card_values = [v.replace("|", "").strip() for v in values[1:]]
                suit_names = ["spades", "hearts", "diamonds", "clubs"]
                for j, sname in enumerate(suit_names):
                    if j < len(card_values):
                        setattr(ba, f"{direction}_{sname}", card_values[j])

        # Results table - find rows with match data
        # The results table has headers: Table, Home Team, Visiting Team, Room, Cont., Decl., Lead, Tricks, NS, EW, Home Res., Vis Res.
        # Find the data table: first td of first row says "Table"
        results_table = None
        for tbl in soup.find_all("table"):
            first_tr = tbl.find("tr", recursive=False)
            if not first_tr:
                continue
            first_tds = first_tr.find_all("td", recursive=False)
            if first_tds and clean_text(first_tds[0].get_text()) == "Table":
                results_table = tbl
                break

        if results_table is None:
            return ba

        rows = results_table.find_all("tr", recursive=False)
        i = 1  # skip header
        while i < len(rows):
            cells = rows[i].find_all("td", recursive=False)
            if len(cells) < 10:
                i += 1
                continue

            # Check if this is a rowspan=2 row (table number spans 2 rows)
            table_cell = cells[0]
            if table_cell.get("rowspan") == "2":
                # This row has Open room data, next row has Closed room data
                table_link = table_cell.find("a")
                table_num = 0
                match_id = 0
                if table_link:
                    table_num = int(re.search(r"\d+", clean_text(table_link.get_text())).group())
                    mid = re.search(r"qmatchid=(\d+)", table_link.get("href", ""))
                    if mid:
                        match_id = int(mid.group(1))

                home_team = clean_text(cells[1].get_text())
                visiting_team = clean_text(cells[2].get_text())

                # Open room: cells[3]=Room, [4]=Cont, [5]=Decl, [6]=Lead, [7]=Tricks, [8]=NS, [9]=EW
                open_contract, open_bidding = extract_contract_and_bidding_from_cell(cells[4])
                open_entry = BoardAcrossEntry(
                    table_num=table_num,
                    home_team=home_team,
                    visiting_team=visiting_team,
                    room="Open",
                    contract=open_contract,
                    declarer=clean_text(cells[5].get_text()),
                    lead=parse_suit_symbols(cells[6]),
                    tricks=safe_int(cells[7].get_text()),
                    ns_score=safe_int(cells[8].get_text()),
                    ew_score=safe_int(cells[9].get_text()),
                    bidding=open_bidding,
                    match_id=match_id,
                )
                # Home/Vis result in cells[10], [11]
                if len(cells) > 10:
                    open_entry.home_result = clean_text(cells[10].get_text())
                if len(cells) > 11:
                    open_entry.visiting_result = clean_text(cells[11].get_text())
                ba.entries.append(open_entry)

                # Next row = Closed room
                if i + 1 < len(rows):
                    closed_cells = rows[i + 1].find_all("td", recursive=False)
                    if len(closed_cells) >= 7:
                        # cells: [0]=Room, [1]=Cont, [2]=Decl, [3]=Lead, [4]=Tricks, [5]=NS, [6]=EW
                        closed_contract, closed_bidding = extract_contract_and_bidding_from_cell(closed_cells[1])
                        closed_entry = BoardAcrossEntry(
                            table_num=table_num,
                            home_team=home_team,
                            visiting_team=visiting_team,
                            room="Closed",
                            contract=closed_contract,
                            declarer=clean_text(closed_cells[2].get_text()),
                            lead=parse_suit_symbols(closed_cells[3]),
                            tricks=safe_int(closed_cells[4].get_text()),
                            ns_score=safe_int(closed_cells[5].get_text()),
                            ew_score=safe_int(closed_cells[6].get_text()),
                            bidding=closed_bidding,
                            match_id=match_id,
                        )
                        closed_entry.home_result = open_entry.home_result
                        closed_entry.visiting_result = open_entry.visiting_result
                        ba.entries.append(closed_entry)
                    i += 2
                    continue
            i += 1

        return ba

    # ─────────────────────────────────────────────
    # Bulk scraping helpers
    # ─────────────────────────────────────────────
    def get_all_board_across_in_round(
        self, round_num: int, category_code: int, num_boards: int
    ) -> list[BoardAcrossData]:
        """Scrape all boards in a round. Returns list of BoardAcrossData."""
        results = []
        for board in range(1, num_boards + 1):
            board_code = f"{board:03d}.{round_num:02d}..{category_code}"
            print(f"Fetching board {board_code}...")
            ba = self.get_board_across(board_code)
            results.append(ba)
        return results

    def get_match_ids_from_board_across(self, board_across: BoardAcrossData) -> list[int]:
        """Extract all unique match IDs from a board-across page."""
        return list({e.match_id for e in board_across.entries if e.match_id > 0})

    def get_all_matches_in_round(
        self, round_num: int, category_code: int, num_boards: int = 10
    ) -> list[MatchData]:
        """Get all matches in a round by first finding match IDs from board 1."""
        board1_code = f"001.{round_num:02d}..{category_code}"
        print(f"Fetching board 1 to discover match IDs: {board1_code}")
        ba = self.get_board_across(board1_code)
        match_ids = self.get_match_ids_from_board_across(ba)
        print(f"Found {len(match_ids)} matches: {match_ids}")

        matches = []
        for mid in sorted(match_ids):
            print(f"Fetching match {mid}...")
            md = self.get_match_details(mid)
            matches.append(md)
        return matches

    # ─────────────────────────────────────────────
    # Export to CSV / DataFrame
    # ─────────────────────────────────────────────
    def match_to_dataframe(self, match: MatchData) -> pd.DataFrame:
        """
        Convert a match to a flat DataFrame (one row per board-room).

        Player name columns:
          open_north/south/east/west   — who played each position in Open Room
          closed_north/south/east/west — who played each position in Closed Room

        These are the same for every board in the match (players don't change mid-match).
        From these + the 'declarer' column you can tell which player declared each board.
        """
        rows = []
        for board_pair in match.boards:
            for room_key in ["open", "closed"]:
                br: BoardResult = board_pair[room_key]
                row = {
                    "match_id":       match.match_id,
                    "round":          match.round_num,
                    "category":       match.category,
                    "home_team":      match.home_team,
                    "visiting_team":  match.visiting_team,
                    "home_vp":        match.home_vp,
                    "visiting_vp":    match.visiting_vp,
                    "home_imp":       match.home_imp,
                    "visiting_imp":   match.visiting_imp,
                    "board":          br.board_num,
                    "room":           br.room,
                    "contract":       br.contract,
                    "declarer":       br.declarer,
                    "lead":           br.lead,
                    "tricks":         br.tricks,
                    "ns_score":       br.ns_score,
                    "ew_score":       br.ew_score,
                    "bidding":        br.bidding,
                    # ── Player names (same for all boards in this match) ──
                    "open_north":     match.open_north,
                    "open_south":     match.open_south,
                    "open_east":      match.open_east,
                    "open_west":      match.open_west,
                    "closed_north":   match.closed_north,
                    "closed_south":   match.closed_south,
                    "closed_east":    match.closed_east,
                    "closed_west":    match.closed_west,
                }
                rows.append(row)
        return pd.DataFrame(rows)

    def board_across_to_dataframe(self, ba: BoardAcrossData) -> pd.DataFrame:
        """Convert a board-across to a flat DataFrame."""
        rows = []
        for entry in ba.entries:
            rows.append({
                "board": ba.board_num,
                "round": ba.round_num,
                "category": ba.category,
                "dealer": ba.dealer,
                "vulnerability": ba.vulnerability,
                "table": entry.table_num,
                "home_team": entry.home_team,
                "visiting_team": entry.visiting_team,
                "room": entry.room,
                "contract": entry.contract,
                "declarer": entry.declarer,
                "lead": entry.lead,
                "tricks": entry.tricks,
                "ns_score": entry.ns_score,
                "ew_score": entry.ew_score,
                "home_result": entry.home_result,
                "visiting_result": entry.visiting_result,
                "bidding": entry.bidding,
                "match_id": entry.match_id,
                # Include card holdings
                "north_spades": ba.north_spades,
                "north_hearts": ba.north_hearts,
                "north_diamonds": ba.north_diamonds,
                "north_clubs": ba.north_clubs,
                "south_spades": ba.south_spades,
                "south_hearts": ba.south_hearts,
                "south_diamonds": ba.south_diamonds,
                "south_clubs": ba.south_clubs,
                "east_spades": ba.east_spades,
                "east_hearts": ba.east_hearts,
                "east_diamonds": ba.east_diamonds,
                "east_clubs": ba.east_clubs,
                "west_spades": ba.west_spades,
                "west_hearts": ba.west_hearts,
                "west_diamonds": ba.west_diamonds,
                "west_clubs": ba.west_clubs,
            })
        return pd.DataFrame(rows)

    def export_matches_to_csv(self, matches: list[MatchData], filepath: str):
        dfs = [self.match_to_dataframe(m) for m in matches]
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
            df.to_csv(filepath, index=False, encoding="utf-8-sig")
            print(f"Exported {len(df)} rows to {filepath}")

    def export_boards_across_to_csv(self, boards: list[BoardAcrossData], filepath: str):
        dfs = [self.board_across_to_dataframe(b) for b in boards]
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
            df.to_csv(filepath, index=False, encoding="utf-8-sig")
            print(f"Exported {len(df)} rows to {filepath}")


# ═══════════════════════════════════════════════
# Quick test / demo
# ═══════════════════════════════════════════════
if __name__ == "__main__":
    scraper = EuroBridgeScraper(delay=0.3)

    # --- Example 1: Single match ---
    print("=" * 60)
    print("Fetching match 138742 (FERM vs MSV, Round 1, Mixed Teams)")
    print("=" * 60)
    match = scraper.get_match_details(138742)
    print(f"  {match.home_team} vs {match.visiting_team}")
    print(f"  VP: {match.home_vp} - {match.visiting_vp}")
    print(f"  IMP: {match.home_imp} - {match.visiting_imp}")
    print(f"  Boards scraped: {len(match.boards)}")
    for bp in match.boards:
        o = bp["open"]
        c = bp["closed"]
        print(f"    Board {o.board_num}: Open={o.contract} by {o.declarer} ({o.tricks} tricks, NS:{o.ns_score} EW:{o.ew_score}) | Closed={c.contract} by {c.declarer} ({c.tricks} tricks, NS:{c.ns_score} EW:{c.ew_score})")
        if o.bidding:
            print(f"      Open bidding: {o.bidding}")

    df_match = scraper.match_to_dataframe(match)
    df_match.to_csv("match_138742.csv", index=False, encoding="utf-8-sig")
    print(f"\nSaved match_138742.csv ({len(df_match)} rows)")

    # --- Example 2: Board across all tables ---
    print("\n" + "=" * 60)
    print("Fetching Board 1, Round 1, Mixed Teams across all tables")
    print("=" * 60)
    ba = scraper.get_board_across("001.01..2513")
    print(f"  Board {ba.board_num}, Dealer: {ba.dealer}, Vul: {ba.vulnerability}")
    print(f"  Tables: {len(ba.entries)}")
    for entry in ba.entries[:6]:  # show first 3 tables (6 entries = 3 open + 3 closed)
        print(f"    Table {entry.table_num} {entry.room}: {entry.home_team} vs {entry.visiting_team} => {entry.contract} by {entry.declarer}, {entry.tricks} tricks")

    df_board = scraper.board_across_to_dataframe(ba)
    df_board.to_csv("board_001_round01.csv", index=False, encoding="utf-8-sig")
    print(f"\nSaved board_001_round01.csv ({len(df_board)} rows)")

    # --- Example 3: All boards in a round ---
    # Uncomment to scrape all 10 boards of round 1:
    # print("\n" + "=" * 60)
    # print("Fetching all boards in Round 1, Mixed Teams")
    # print("=" * 60)
    # all_boards = scraper.get_all_board_across_in_round(round_num=1, category_code=2513, num_boards=10)
    # scraper.export_boards_across_to_csv(all_boards, "round01_all_boards.csv")

    # --- Example 4: All matches in a round ---
    # Uncomment to scrape all matches:
    # all_matches = scraper.get_all_matches_in_round(round_num=1, category_code=2513, num_boards=10)
    # scraper.export_matches_to_csv(all_matches, "round01_all_matches.csv")
