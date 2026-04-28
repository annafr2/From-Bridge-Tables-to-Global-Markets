# PRD — Bridge Data Collection Pipeline
## Product Requirements Document

**Project:** PhD Data Collection — Bridge Game Data
**Researcher:** Anna Ben Shushan, LUT University
**Phase:** Year 1, Months 1–12
**Last Updated:** April 2026

> **Status update (April 2026):** Primary data source changed from BBO to EuroBridge.
> A working scraper (`eurobridge_scraper.py` + `eurobridge_bulk_scraper.py`) is collecting data and was adapted for older layouts (e.g., Madeira 2022).
> Currently collected: Herning 2024, Madeira 2022, Poznan 2025 (~78.5K rows). Full collection in progress.

---

## 1. Problem Statement

The PhD research requires a large, structured dataset of **Contract Bridge games** to train AI models that study strategic decision-making, bidding dialects, cooperation/competition dynamics, and negotiation patterns.

Currently there is **no data** and **no pipeline**. This document defines exactly what data is needed, where to get it, how to process it, and how to store it so it is ready for machine learning.

---

## 2. Goals

### Primary Goals
- Collect at minimum **100,000 bridge deals** with complete bidding and play records
- Parse and normalize raw data into a clean, consistent format
- Engineer features directly relevant to the PhD research questions
- Store data in a format ready for model training (CSV, Parquet, JSON)

### Secondary Goals
- Capture **timing data** where available (how long each decision took)
- Capture **player-level metadata** (anonymized player IDs, skill levels)
- Enable **reproducibility**: every download is logged with date, source, version
- Design the pipeline so it can grow — more sources can be added later

---

## 3. What Data Do We Need?

### 3.1 Core Deal Data (Required)

Every record must have:

| Field | Description | Example |
|-------|-------------|---------|
| `deal_id` | Unique identifier for the deal | `bbo_2024_001234` |
| `source` | Where the data came from | `BBO`, `WBF`, `PBN_archive` |
| `date` | When the game was played | `2024-03-15` |
| `north_hand` | 13 cards held by North player | `SA KH QD JC 9C ...` |
| `south_hand` | 13 cards held by South | ... |
| `east_hand` | 13 cards held by East | ... |
| `west_hand` | 13 cards held by West | ... |
| `dealer` | Who dealt first | `N`, `S`, `E`, `W` |
| `vulnerability` | Who is vulnerable | `None`, `NS`, `EW`, `Both` |
| `bidding_sequence` | Full auction in order | `["1C","1H","2H","4H","Pass","Pass","Pass"]` |
| `contract` | Final contract | `4H` |
| `declarer` | Who plays the hand | `S` |
| `doubled` | Was contract doubled/redoubled | `None`, `X`, `XX` |
| `opening_lead` | First card played | `SK` |
| `play_sequence` | All 52 cards in order | `["SK","S3","S5","SA","..."]` |
| `tricks_made` | How many tricks declarer won | `10` |
| `result` | Contract result | `+1`, `=`, `-2` |
| `score` | Points scored | `450` |

### 3.2 Player Data (Important for Research)

| Field | Description |
|-------|-------------|
| `north_player_id` | Anonymized player ID |
| `south_player_id` | Anonymized player ID |
| `east_player_id` | Anonymized player ID |
| `west_player_id` | Anonymized player ID |
| `north_player_rank` | Skill level / master points if available |
| `convention_system` | Bidding system used (SAYC, 2/1, Precision, Acol...) |

### 3.3 Timing Data (Bonus — if available)

| Field | Description |
|-------|-------------|
| `bid_timing` | Time taken per bid (seconds) |
| `play_timing` | Time taken per card play (seconds) |

---

## 4. Where Does the Data Come From?

### Source 1: EuroBridge (db.eurobridge.org) — **PRIMARY** ✅ ACTIVE

**Why:** European Bridge League database. Contains all major European Championships with full bidding, team names, and match structure. Crucially: every board is played by multiple pairs (Open + Closed rooms), which is essential for the PhD research questions.

**What is available:**
- All European Championship tournaments (2000–2025)
- Each board played twice per round (Open room + Closed room)
- Full bidding sequences, contracts, leads, tricks, scores
- Team names per match

**How to access:**
- `https://db.eurobridge.org/repository/competitions/`
- Automated via `src/downloaders/eurobridge_bulk_scraper.py`
- Competition list configured in `configs/competitions.yaml`

**Format:** HTML pages → parsed to CSV

**Estimated volume:** ~150,000 rows from 5 competitions × 4 categories

---

### Source 2: Bridge Base Online (BBO) — **FUTURE**

**Why:** Largest online platform — millions of hands including non-elite players. Useful in Year 2 for studying a wider range of bidding styles and dialects.

**Format:** `.lin` (BBO proprietary format — needs custom parser, not yet built)

---

### Source 3: World Bridge Federation (WBF) — **SUPPLEMENTARY**

**Why:** High-quality tournament data from elite players (world championships, European championships). Useful for studying expert-level play.

**What is available:**
- Tournament results and hand records in `.pbn` format
- World championship bulletins with annotated hands

**How to access:**
- `https://www.worldbridge.org/` — public archives
- Download tournament `.pbn` files directly

**Format:** `.pbn` (Portable Bridge Notation — well-documented standard)

---

### Source 3: PBN Archives (Public Collections)

**Why:** Large existing collections of hands already in standard format. Easy to start with.

**What is available:**
- Thousands of `.pbn` files from various sources
- Historic tournament records
- Problem collections (educational hands with annotations)

**How to access:**
- Various public repositories and bridge club websites
- GitHub repositories with PBN datasets

**Format:** `.pbn`

---

### Source 4: Funbridge (Future)

- Popular European platform
- May require web scraping or API investigation
- Lower priority — add in Month 9–12 if needed

---

## 5. Data Processing Pipeline

### Step 1: Download / Fetch Raw Data
- Download `.lin` and `.pbn` files from sources above
- Store originals unchanged in `data/raw/` — **never modify raw files**
- Log every download: source URL, date downloaded, file size, number of deals

### Step 2: Parse Raw Files
- Parse `.lin` files → Python dict / JSON
- Parse `.pbn` files → Python dict / JSON
- Handle errors gracefully: corrupted/incomplete records go to `data/rejected/`

### Step 3: Normalize to Common Schema
- Convert both formats to a single unified schema (defined in Section 3)
- Standardize card notation: `SA` = Ace of Spades, `2H` = 2 of Hearts
- Standardize bid notation: `1NT`, `Pass`, `Dbl`, `Rdbl`
- Standardize vulnerability: always `None` / `NS` / `EW` / `Both`

### Step 4: Validate & Clean
- Remove duplicate deals (same deal_id or same cards + same bids)
- Remove incomplete records (missing bidding sequence or card play)
- Validate card deals: each hand must have exactly 13 cards, no repeats
- Validate bidding: sequence must be valid (proper turn order, legal bids)
- Flag suspicious records (e.g., impossible scores)

### Step 5: Feature Engineering
Extract additional columns useful for ML:

| Feature | How Computed |
|---------|-------------|
| `hcp_north` | High Card Points for North (A=4, K=3, Q=2, J=1) |
| `hcp_south` | Same for South |
| `hcp_ew` | Combined HCP for East-West |
| `distribution_north` | Shape of North's hand (e.g., 4-4-3-2) |
| `auction_length` | Number of bids in auction |
| `has_double` | Was there a double/redouble? |
| `is_competitive` | Did both sides bid? |
| `final_level` | Contract level (1-7) |
| `final_suit` | Contract suit (C, D, H, S, NT) |
| `bid_dialects` | Encoded bidding sequence for NLP |
| `par_score` | Theoretical optimal score for the deal |
| `result_vs_par` | How far from optimal the players played |
| `partnership_ns` | Combined NS player ID (for tracking partnerships) |
| `partnership_ew` | Combined EW player ID |

### Step 6: Store Final Dataset
- Primary storage: **Parquet** format (efficient, typed, fast for pandas/spark)
- Also export: **CSV** for easy inspection
- Index by: `deal_id`, `date`, `source`, `north_player_id`

---

## 6. Folder Structure

```
collectBridgeData/
├── CLAUDE.md              # Context for Claude
├── PRD.md                 # This file
├── TASKS.md               # Implementation tasks
├── requirements.txt       # Python dependencies
├── configs/
│   └── sources.yaml       # Source URLs and settings
├── src/
│   ├── downloaders/
│   │   ├── bbo_downloader.py      # Download from BBO
│   │   └── wbf_downloader.py      # Download from WBF
│   ├── parsers/
│   │   ├── lin_parser.py          # Parse .lin files
│   │   └── pbn_parser.py          # Parse .pbn files
│   ├── normalizer.py              # Normalize to common schema
│   ├── validator.py               # Validate and clean
│   ├── feature_engineer.py        # Compute ML features
│   └── pipeline.py                # Run full pipeline end-to-end
├── data/
│   ├── raw/                       # Raw downloaded files (read-only)
│   │   ├── bbo/
│   │   └── wbf/
│   ├── rejected/                  # Bad records with reason logged
│   ├── processed/                 # Parsed + normalized JSON/CSV
│   └── features/                  # Final ML-ready Parquet files
├── notebooks/
│   ├── 01_explore_raw_data.ipynb
│   ├── 02_validate_pipeline.ipynb
│   └── 03_feature_analysis.ipynb
├── tests/
│   ├── test_lin_parser.py
│   ├── test_pbn_parser.py
│   └── test_validator.py
└── logs/
    └── download_log.csv           # Audit trail of all downloads
```

---

## 7. Success Criteria

| Metric | Target |
|--------|--------|
| Total deals collected | >= 100,000 |
| Deals with full bidding + play | >= 80% |
| Deals with timing data | >= 10,000 |
| Duplicate rate after dedup | < 1% |
| Validation pass rate | >= 95% |
| Pipeline runtime for 10k deals | < 30 minutes |
| Data ready for ML (Parquet output) | Yes |

---

## 8. What NOT to Collect (Out of Scope for Now)

- Business negotiation data (Phase 2)
- Synthetic/simulated deals (Phase 3 — only real human games for now)
- Video/audio of bridge tournaments
- Private player data (emails, personal info) — use only anonymized IDs

---

## 9. Ethical and Legal Considerations

- Only use **publicly available** data or data explicitly authorized for research
- **Anonymize** all player identifiers — no real names stored
- **Rate-limit** all web requests — be respectful, no aggressive scraping
- **Log all sources** — required for academic citations
- Store data on a **secure, local machine** — not on public cloud without encryption
- Comply with each platform's **Terms of Service** — check before scraping

---

## 10. Technology Stack

| Purpose | Tool |
|---------|------|
| Language | Python 3.10+ |
| Data manipulation | pandas, polars |
| Storage | Parquet (pyarrow), CSV |
| Web requests | requests, httpx |
| HTML parsing | BeautifulSoup4 |
| Configuration | YAML (PyYAML) |
| Logging | Python logging module |
| Testing | pytest |
| Notebooks | Jupyter |
| Version control | Git |
