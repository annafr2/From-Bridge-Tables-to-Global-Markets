# Bridge Data Collection — Project README

**Researcher:** Anna Ben Shushan
**PhD Program:** LUT University, Finland
**Thesis:** "From Bridge Tables to Global Markets: AI and Optimization Models of Cooperation, Competition, and Negotiation on Digital Platforms"
**Last Updated:** April 2026

---

## What Is This Project?

This project collects and organizes **real bridge game data** from elite European tournaments.

The data is used to train AI models that study how people make strategic decisions under uncertainty — and then apply those insights to international business, negotiations, and digital platforms.

Bridge is used as a **controlled laboratory** for studying strategy, because:
- Players only see their own 13 cards (incomplete information — just like in real business)
- Two partners must cooperate while competing against two opponents
- The bidding system is a structured communication language with rules and signals
- Every decision is recorded and can be analyzed

---

## What Data We Collected

### Source: EuroBridge (db.eurobridge.org)

We scraped data from **5 major European Bridge Championships**:

| Competition | Year | Categories |
|-------------|------|------------|
| European Championships — Herning | 2024 | Open, Women, Senior, Mixed |
| European Championships — Madeira | 2022 | Open, Women, Senior, Mixed |
| European Championships — Ostend | 2018 | Open, Women, Senior, Mixed |
| European Championships — Budapest | 2016 | Open, Women, Senior, Mixed |
| European Championships — Poznan | 2025 | Open, Women, Senior, Mixed |

**These are top-level elite players** — the best bridge players in Europe.

### What Each Row Contains

Every row in our dataset represents **one board played in one room** by one pair of teams:

| Field | Example | What It Means |
|-------|---------|---------------|
| `match_id` | 120186 | Unique ID for this match |
| `home_team` | SWEDEN | Team playing at home |
| `visiting_team` | NORWAY | Opposing team |
| `board` | 1 | Board number (1–16) |
| `room` | Open | Open room or Closed room |
| `contract` | 4H | Final contract (4 Hearts) |
| `declarer` | N | Who plays the hand (North) |
| `lead` | SK | Opening lead card (King of Spades) |
| `tricks` | 10 | How many tricks declarer won |
| `ns_score` | 420 | Score for North-South |
| `ew_score` | 0 | Score for East-West |
| `bidding` | `W:- N:1H E:Pass S:2H \| W:Pass N:4H...` | Full bidding sequence |
| `open_north` | BRINK Sjoert | Player in North seat, Open Room |
| `open_south` | FERM Barbara | Player in South seat, Open Room |
| `open_east` | SAINZ DE VICUNA Maria | Player in East seat, Open Room |
| `open_west` | LANTARON Luis | Player in West seat, Open Room |
| `closed_north` | MEDIERO Marina | Player in North seat, Closed Room |
| `closed_south` | WASIK Arturo | Player in South seat, Closed Room |
| `closed_east` | MANNO Andrea | Player in East seat, Closed Room |
| `closed_west` | GRONKVIST Ida | Player in West seat, Closed Room |

**Key derived insight:** `declarer=N` + `open_north=BRINK Sjoert` → BRINK declared this board.
Combined with `contract`, `tricks`, and `bidding` → we know exactly who made which decision and whether it paid off.

### Key Feature: Same Hand, Multiple Tables

In team tournaments, **every board is played twice** — once in the Open room and once in the Closed room, by different pairs. This means:

```
Board 1, Round 1:
  Table 1 Open:   SWEDEN vs NORWAY     → bid 4H, made 10 tricks, score +420
  Table 1 Closed: NORWAY vs SWEDEN     → bid 3NT, made 9 tricks, score +400
  Table 2 Open:   FRANCE vs ITALY      → bid 4H, went down 1, score -50
  Table 2 Closed: ITALY vs FRANCE      → bid 5H, made 11 tricks, score +450
  ...
```

**The same 52 cards, 8–20 different pairs, all different bidding decisions.** This is gold for research.

---

## How We Collected It — The Scraper

### File: `src/downloaders/eurobridge_scraper.py`

The base scraper. Given a match ID, it visits the EuroBridge website and extracts:
- Team names
- All boards with contracts, leads, tricks, scores
- Full bidding sequences (parsed from HTML tooltips)

### File: `src/downloaders/eurobridge_bulk_scraper.py`

The automation layer. It:
1. Reads the competition list from `configs/competitions.yaml`
2. For each competition → each category → each round: discovers all match IDs
3. Downloads every match using the base scraper
4. Saves results to `data/raw/eurobridge/<competition>/<category>/matches.csv`
5. Logs every download to `logs/scrape_log.csv` (so it never re-downloads what it already has)

### To run the scraper:

```bash
# From the collectBridgeData folder:

# Download everything (all competitions, all categories):
python src/downloaders/eurobridge_bulk_scraper.py --delay 0.8

# Download only one competition:
python src/downloaders/eurobridge_bulk_scraper.py --competitions EBL_Herning_2024

# Download only Mixed and Open categories:
python src/downloaders/eurobridge_bulk_scraper.py --categories Mixed Open

# Test without downloading (just discover what exists):
python src/downloaders/eurobridge_bulk_scraper.py --dry-run
```

---

## Why This Data Specifically? (Not Kaggle)

A generic bridge dataset from Kaggle would only give you cards + results.
**That is enough to teach an AI to play bridge. It is not enough for this PhD.**

Here is why our dataset is different:

### Research Question 1: Bidding Dialects
> *"Can AI learn a player's unique bidding dialect and predict their future decisions?"*

This requires tracking **the same player across many games**.
Our data includes **team names** in every row — so we can group all games by a specific team and study their bidding style.
Kaggle datasets have no player identity.

### Research Question 2: Partner Matching and Decision Styles
> *"Does modeling personal decision-making style improve partner matching?"*

This requires seeing **how different pairs bid the same hand**.
Our data has the same board played by 8–20 different pairs (Open + Closed rooms across multiple tables).
This lets us ask: "Given these 52 cards, which pairs made the best decision? What was different about their bidding?"

### Research Question 3: Cooperation and Competition Dynamics
> *"Do cooperation-competition patterns in bridge reflect alliance dynamics in global business?"*

This requires seeing **how partnerships evolve over many rounds together**.
Our data covers entire tournaments — the same teams playing round after round.
We can track how a partnership's performance changes, when they miscommunicate, and when they excel.

### Research Question 4: AI-Enhanced Game Theory
> *"Can AI models trained on bridge explain business negotiation outcomes?"*

This requires **high-quality, expert-level data** — not casual online games.
Our data is from European Championships — the strongest players in Europe.
Elite-level play shows the most sophisticated strategic decisions.

---

## Data Volume (Target)

| Status | What |
|--------|------|
| ✅ Done | Herning 2024, Madeira 2022, Poznan 2025 (all categories): **78,584 rows** |
| 🔄 In progress | Ostend 2018, Budapest 2016 |
| 🎯 Target | **~150,000 rows** total |

---

## Folder Structure

```
collectBridgeData/
├── README.md                          ← This file
├── CLAUDE.md                          ← Context for AI assistant
├── PRD.md                             ← Full requirements document
├── TASKS.md                           ← Implementation task list
├── configs/
│   └── competitions.yaml              ← List of competitions to scrape
├── src/
│   └── downloaders/
│       ├── eurobridge_scraper.py      ← Base scraper (one match)
│       └── eurobridge_bulk_scraper.py ← Bulk scraper (all competitions)
├── data/
│   └── raw/
│       └── eurobridge/
│           ├── EBL_Herning_2024/      ← ✅ Done
│           ├── EBL_Madeira_2022/      ← ✅ Done
│           ├── EBL_Poznan_2025/       ← ✅ Done
│           └── ...
├── logs/
│   └── scrape_log.csv                 ← Audit trail of all downloads
└── (future folders: data/processed, data/features, notebooks, tests)
```

---

## Research Questions This Dataset Can Answer

See `RESEARCH_INSIGHTS.md` for the full list. Three examples:

**1. Individual risk profiles** *(Paper 1 target: AAMAS 2027)*
> "Is BRINK Sjoert a calculated risk-taker? How many slams did he bid, and how many succeeded?"
> → Use `open_north` / `declarer` / `is_slam_contract` / `made_contract`

**2. VP-scale game theory** *(Paper 2 target: Games and Economic Behavior)*
> "Do players actually take more risks when losing? Does the VP scale incentive work?"
> → Use running IMP score per board + risk metrics

**3. Partnership chemistry** *(Paper 3 target: IJCAI 2027)*
> "Do established pairs bid better than new pairings on the same hand?"
> → Use pair identity across tournaments + consistency metrics

---

## Next Steps

1. ✅ ~~Finish data collection~~ — Herning, Madeira, Poznan done; Ostend + Budapest running
2. ✅ ~~Combine all CSVs~~ — `data/processed/all_matches.parquet` (78K rows)
3. ✅ ~~Add card holdings~~ — N/S/E/W cards in ~50% of rows
4. ✅ ~~Add individual player names~~ — 8 player columns added (April 2026)
5. 🔲 **Build risk metrics** — `src/features/risk_metrics.py`
6. 🔲 **Compute running match scores** — `src/features/running_score.py`
7. 🔲 **First analysis notebook** — VP scale test, risk profiles

---

## Data Ethics

- All data is from **publicly available** tournament records on EuroBridge
- Player names are **team names** (national or club teams), not individual personal data
- The scraper uses **rate limiting** (0.8 second delay between requests) to be respectful
- Every download is **logged** for academic citation purposes
- This data is used exclusively for **non-commercial academic research**
