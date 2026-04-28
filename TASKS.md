# TASKS.md — PhD Implementation Plan (Advisor-Organized)

**Status Legend:** `[ ]` Not started  `[~]` In progress  `[x]` Done

Each task is tagged with:
- **RQ#** — which Research Question(s) it serves (see CLAUDE.md)
- **Model:** — recommended model/tool (when applicable)

---

# YEAR 1 — Data Foundation + First Models

Goal by end of Year 1: A clean dataset of ≥200K labelled deals, feature-engineered, plus a working ENN baseline and bidding-dialect clustering.

---

## PHASE 0 — Project Setup ✅ COMPLETE

```
[x] T0.1  Environment, folder structure, CLAUDE.md/PRD.md/TASKS.md
[x] T0.2  Explored PBN + EuroBridge HTML formats
```

---

## PHASE 1 — EuroBridge Scraper ✅ COMPLETE

```
[x] T1.1  Base scraper (one match)
[x] T1.2  Bulk scraper (all competitions / categories / rounds)
[x] T1.3  configs/competitions.yaml with 5 competitions
[x] T1.4  First run: EBL Herning 2024 Mixed → 6,656 rows
[x] T1.5  Cards scraper (N/S/E/W holdings via BoardAcross)
[x] T1.6  Discovered even/odd round pairing (Open/Closed room structure)
```

---

## PHASE 2 — Pipeline & Master Dataset ✅ COMPLETE

```
[x] T2.1  src/pipeline.py — joins matches + cards
[x] T2.2  Handles even/odd round pairing at join
[x] T2.3  Quality flags has_bidding / has_cards (don't drop partial rows)
[x] T2.4  Produces data/processed/all_matches.parquet (78,584 × 40)
[x] T2.5  explore_data.py + sample_data.csv
```

---

## PHASE 3 — Fix Remaining Data Gaps + Risk Research Data  ← **YOU ARE HERE**

> **RQ:** all — we need clean, complete data before any modelling

### T3.1 — Fix Madeira 2022 scraper
```
[x] Inspect Madeira 2022 HTML structure (diff vs. Herning 2024)
[x] Update eurobridge_scraper.py parsers to handle both layouts
[x] Re-scrape Madeira 2022
[x] Re-run pipeline → expect ~95K rows total
```

### T3.2 — Finish remaining competitions
```
[x] Run bulk scraper for Poznan 2025
[ ] Run bulk scraper for Ostend 2018
[ ] Run bulk scraper for Budapest 2016
[ ] Target: ~150K board-room rows across all 5 competitions
```

### T3.2b — Run Ostend 2018 + Budapest 2016 (PARALLEL with T3.3)
```
[ ] python src/downloaders/eurobridge_bulk_scraper.py --competitions EBL_Ostend_2018 --delay 0.8
[ ] python src/downloaders/eurobridge_bulk_scraper.py --competitions EBL_Budapest_2016 --delay 0.8
[ ] python src/downloaders/eurobridge_cards_scraper.py --competitions EBL_Ostend_2018 EBL_Budapest_2016 --delay 0.8
[ ] Re-run pipeline → target >140K rows
```
> **RQ:** all — volume needed for statistical power

### T3.3 — Data validation pass
```
[ ] Check bidding format is consistent across competitions
    (spot: hearts as "1H" vs "1♥", Dbl vs X)
[ ] Check score ranges are sane (bridge scoring limits)
[ ] Log which rows have which fields (produce a "data coverage report")
[ ] Document known issues in data/DATA_ISSUES.md
```

### T3.4 — Normalize bidding sequences
```
[ ] Parse bidding string → structured Python list
    Input:  "W:- N:1H E:Pass S:2H | W:Pass N:4H ..."
    Output: [{"player":"N","bid":"1H"},{"player":"E","bid":"Pass"}, ...]
[ ] Standardize notation:
    Suits: "1H" "1S" "3NT" (never ♥/♠)
    Calls: "Pass" "Dbl" "Rdbl" (pick one spelling globally)
[ ] Save as a new parquet column `bidding_parsed` (list of dicts)
```

---

## PHASE 3.5 — Risk Research Data (NEW — from Nezer's research direction)

> **RQ1, RQ4** — individual risk-taking behavior and game-state driven decisions

### T3.5a — Download VP scale tables
```
[ ] Go to: https://www.ebu.co.uk/regulation-and-conduct/vp-scales
[ ] Download VP tables for: 8, 10, 12, 14, 16, 20 boards
[ ] Save as: data/raw/vp_tables/vp_scale_NNboards.csv
[ ] Format: two columns — imp_diff, home_vp, visiting_vp
```

### T3.5b — Build VP converter utility
```
[ ] File: src/features/vp_converter.py
[ ] Function: imp_to_vp(imp_diff: int, boards: int) → tuple[float, float]
[ ] Function: vp_to_imp_margin(vp: float, boards: int) → int
[ ] Unit test: verify against Nezer's example (+15 → 14.46–5.54 for 12 boards)
```

### T3.5c — Compute running match score per board
```
[ ] For every match, compute CUMULATIVE IMP after each board:
    - board 1: score = ns_score of board 1
    - board 2: score = board1_score + ns_score of board 2
    - etc.
[ ] Add column: running_ns_imp_after_board  (running NS total up to this board)
[ ] Add column: running_ew_imp_after_board
[ ] Add column: ns_leading_by              (positive = NS ahead, negative = EW ahead)
[ ] Save back to features parquet
```
> **Key insight:** we can already do this from existing data. No new scraping needed.

### T3.5d — Compute tournament standing per team per round
```
[ ] For each round in each competition+category, rank teams by cumulative VP
[ ] Add column: team_standing_before_round  (1st, 2nd, 3rd... place)
[ ] Add column: vp_gap_to_leader            (how many VPs behind 1st place)
[ ] This tells us: is this team trying to catch up, or protecting a lead?
```

### T3.5e — Parse individual bid attributions from bidding string
```
[ ] The bidding column already contains position info: "W:- N:1NT E:Pass S:2H | ..."
[ ] Write parser: src/features/bid_parser.py
[ ] For each board, produce a list of dicts:
    [
      {"turn": 1, "position": "W", "bid": "Pass", "is_opening": True},
      {"turn": 2, "position": "N", "bid": "1NT",  "is_opening": False},
      ...
    ]
[ ] Derive: who made the first aggressive bid? Who initiated slam?
[ ] Add column: opening_position  (who opened: N/S/E/W)
[ ] Add column: first_slam_bid_position  (who first bid slam level)
[ ] Add column: double_position  (who doubled)
```
> **This is FREE data** — it's already in the bidding column. No new scraping.

### T3.5f — Investigate EuroBridge player roster pages  ✅ DONE
```
[x] EuroBridge BoardDetails page DOES contain individual player names
[x] HTML structure: 4-row seating table with compass layout
    Row 1: North (Open) | North (Closed)
    Row 2: West (Open) | compass | East (Open) | West (Closed) | compass | East (Closed)
    Row 3: South (Open) | South (Closed)
[x] Links pattern: <a href="http://www.eurobridge.org/person?qryid=XXXX">NAME</a>
[x] Written: src/downloaders/eurobridge_players_backfill.py
[x] Written: src/tools/test_player_extraction.py (all 8 names verified ✅)
[x] 8 new columns added to DataFrame: open_north/south/east/west + closed_north/south/east/west
[x] Verified on match 138742: BRINK Sjoert, FERM Barbara, LANTARON Luis, SAINZ DE VICUNA...
```
> **UNLOCK:** We can now ask "who made which decision" for every board in the dataset.

### T3.5f2 — Run backfill on ALL competitions
```
[~] python src/downloaders/eurobridge_players_backfill.py --delay 0.5
    (running now — ~30-40 minutes)
[ ] After completion: python src/pipeline.py to rebuild parquet
[ ] Verify: df["open_north"].notna().mean() > 0.8 (>80% of rows have names)
```

### T3.5g — Download WBF PBN files (have player names!)
```
[ ] Go to: https://www.worldbridgefed.com/news-media/download-centre/
[ ] Download PBN files for (priority order):
    1. Bermuda Bowl 2023  (Open world championship, ~500 deals)
    2. Venice Cup 2023     (Women world championship)
    3. World Teams Olympiad 2022
    4. Bermuda Bowl 2022
    5. Bermuda Bowl 2019  (you already have a sample of this)
[ ] Save to: data/raw/wbf/
[ ] Check: do PBN files contain [West "name"] [North "name"] tags?
[ ] If yes: WBF PBN is our source for player-level analysis
```
> **WBF PBN has player names in every deal.** This is what EuroBridge lacks.

### T3.5h — Build risk metrics module
```
[ ] File: src/features/risk_metrics.py
[ ] Per board, compute:
    is_slam_attempt   — any bid at 5+ level in auction
    is_slam_contract  — final contract at 6 or 7 level
    is_preempt_open   — opening bid at 2/3/4 level
    is_double         — Dbl appears in bidding
    is_redouble       — Rdbl appears in bidding
    is_sacrifice      — team took a loss deliberately (heuristic: doubled contract, went down 1-2, would have cost less than opponent's contract)
    is_insurance      — stopped at part-score when game was available (heuristic)
    risk_score        — composite: slam_attempt×3 + preempt×2 + double×1
[ ] Per team per competition: aggregate risk_score → team_risk_profile
[ ] Per match situation: correlate risk_score with running_imp (is risk higher when losing?)
```

---

## PHASE 4 — Feature Engineering

> **RQ1, RQ2, RQ3, RQ4** — features are the input to every model we'll train

### T4.1 — Hand features (per player)
```
[ ] hcp_north, hcp_south, hcp_east, hcp_west  (Ace=4 King=3 Queen=2 Jack=1)
[ ] distribution_north = "5431" (sorted suit lengths)
[ ] longest_suit_north = "H"
[ ] is_balanced_north  (4333, 4432, 5332)
[ ] is_semi_balanced_north  (5422, 6322)
[ ] same for S, E, W
```

### T4.2 — Auction features
```
[ ] auction_length              (total calls including passes)
[ ] auction_length_non_pass     (bids only)
[ ] competitive                 (both NS and EW bid a suit?)
[ ] has_double                  (any Dbl or Rdbl?)
[ ] has_redouble
[ ] opening_side                (NS or EW opened?)
[ ] opening_bid                 (e.g. "1H")
[ ] jump_count                  (non-forcing jumps)
[ ] preempt_opened              (opened at 2 or 3 level)
[ ] bid_encoding                (list[int] — vocabulary of unique bids)
```

### T4.3 — Contract features
```
[ ] contract_level              (1-7)
[ ] contract_suit               (C/D/H/S/NT)
[ ] contract_doubled            (0=undoubled, 1=X, 2=XX)
[ ] is_game                     (3NT, 4H, 4S, 5C, 5D or higher)
[ ] is_slam                     (6/7 level)
[ ] tricks_needed               (level + 6)
[ ] made_contract               (tricks >= tricks_needed)
[ ] overtricks / undertricks
```

### T4.4 — Partnership features
```
[ ] partnership_id              (stable hash of team names)
[ ] partnership_board_count     (how many boards this pair has played across dataset)
[ ] same_board_score_rank       (on each board, where did this pair rank vs others?)
[ ] consistency_score           (std dev of their results across shared boards)
```
**Serves:** RQ2 (partner matching), RQ4 (coopetition)

### T4.5 — Save to `data/features/`
```
[ ] src/features/build_features.py
[ ] data/features/deals_features.parquet
[ ] Document every feature in data/features/FEATURE_CARD.md
```

---

## PHASE 5 — First Exploratory Analysis

> **RQ1, RQ2** — sanity check + first signals for the research narrative

### T5.1 — notebooks/01_data_overview.ipynb
```
[ ] Row count, column count, unique matches/teams
[ ] Competition × category matrix with row counts
[ ] Distribution of contracts (bar chart: 1NT, 3NT, 4H, 4S, ...)
[ ] Distribution of tricks made (histogram)
[ ] Distribution of HCP per hand (should be ~normal around 10)
[ ] % deals where both rooms played same contract (agreement rate)
```

### T5.2 — notebooks/02_bidding_dialects_preview.ipynb
> **Serves RQ1** — the central thesis claim
```
[ ] For 20 "same hand across many pairs" boards, show:
    - The hand
    - Every pair's contract + bidding sequence
    - Visually demonstrate: same cards → different decisions
[ ] Compute simple dialect metric:
    avg_auction_length per team, % competitive per team
[ ] First scatter plot: teams in (aggression × length) space
```

### T5.3 — notebooks/03_partnership_consistency.ipynb
> **Serves RQ2**
```
[ ] For top-20 partnerships (most boards played):
    - Score variance on matching boards
    - Contract agreement rate with partner
[ ] Rank partnerships by consistency — is there a "chemistry" signal?
```

---

## PHASE 6 — Bidding Dialect Model (first real ML)

> **RQ1, RQ2**
> **Model:** BERT-style Transformer encoder on bidding sequences + HDBSCAN clustering
> **Why BERT:** Bidding is a short, structured "language" — masked-language-modelling on bid tokens learns contextual embeddings without needing labels.

### T6.1 — Bid tokenizer
```
[ ] Build vocabulary of all bid tokens (1C ... 7NT, Pass, Dbl, Rdbl, + special [CLS][SEP])
[ ] src/models/bid_tokenizer.py
[ ] Unit tests: round-trip tokenize/decode
```

### T6.2 — Pre-train BiddingBERT
```
[ ] Small model first: 4 layers, 128 hidden, 4 heads (≈2M params)
[ ] Masked-bid prediction (mask 15% of non-pass bids, predict them)
[ ] Train on all `bidding_parsed` sequences (~200K expected after Phase 3)
[ ] MLflow tracking: loss curves, perplexity
[ ] src/models/bidding_bert.py
```

### T6.3 — Dialect clustering
```
[ ] Extract BiddingBERT [CLS] embedding per auction
[ ] Aggregate per team: mean embedding → team vector
[ ] HDBSCAN cluster team vectors
[ ] Label clusters by looking at representative boards
    (e.g., "aggressive preempt school", "conservative natural", "strong club")
[ ] notebooks/04_dialect_clusters.ipynb
```

---

## PHASE 7 — ENN Baseline (Estimation Neural Network)

> **RQ1, RQ3**
> **Model:** Transformer encoder (following Rong et al. 2019). Input: own 13 cards + bidding so far. Output: probability of each remaining card being in each opponent's hand.

### T7.1 — Training data prep
```
[ ] For each deal, generate N training examples:
    - At step k of the auction, inputs = (hand, bids[:k])
    - Target = full 39-card distribution in the other 3 hands
[ ] Split 80/10/10 train/val/test; no deals overlap between splits
```

### T7.2 — ENN architecture
```
[ ] src/models/enn.py
[ ] 6-layer Transformer encoder, 256 hidden, 8 heads
[ ] Card-embedding: 52-card vocabulary, learned embeddings
[ ] Output head: 3 × 52 × softmax-over-players
[ ] PyTorch or JAX (choose once, stick with it)
```

### T7.3 — Train + evaluate
```
[ ] Train on all labelled deals
[ ] Metric: top-k accuracy for card ownership
[ ] Compare to simple baseline (uniform random + obvious bid inferences)
[ ] Publish loss curves to MLflow
```

---

# YEAR 2 — Scale Up, PNN, LLM, Publications

Goal by end of Year 2: A full ENN+PNN with DRL self-play, bidding dialect model published, first business-negotiation study, and data scaled 10× via BBO.

---

## PHASE 8 — Data Scale-Up

### T8.1 — WBF PBN parser & downloader
```
[ ] src/parsers/pbn_parser.py  — generic PBN parser
[ ] src/downloaders/wbf_downloader.py — finds all WBF year archives
[ ] Target: add all World Championships 2010–2025 (~500K deals)
```

### T8.2 — BBO LIN parser & Vugraph downloader
```
[ ] src/parsers/lin_parser.py
[ ] src/downloaders/bbo_downloader.py
[ ] Target: add Vugraph archives 2015–2024 (millions of deals)
[ ] Critical: BBO has player IDs → enables real dialect analysis per individual
```

### T8.3 — Normalizer across sources
```
[ ] src/normalizer.py — unify EBL + WBF + BBO into one schema
[ ] Deduplicate (same physical deal across sources)
[ ] Final master: data/processed/all_deals_multisource.parquet
```

---

## PHASE 9 — PNN + DRL Self-Play

> **RQ1, RQ4**
> **Model:** PPO over a Transformer-decoder policy (Rong et al. 2019 architecture)

### T9.1 — PNN architecture
```
[ ] src/models/pnn.py
[ ] Consumes ENN embedding + bidding state
[ ] Outputs distribution over next legal bid
```

### T9.2 — Self-play environment
```
[ ] src/env/bridge_auction_env.py  (gym-style)
[ ] Actions = legal bids; reward = final IMP score vs par
[ ] 4 agents (2 partnerships) share weights
```

### T9.3 — PPO training loop
```
[ ] Stable-Baselines3 or CleanRL
[ ] MLflow tracking
[ ] Benchmark: beat random & beat a heuristic Natural-system bot
```

### T9.4 — MCCFR equilibrium analysis
> **Serves RQ4**
```
[ ] Implement MCCFR on a reduced auction subset (e.g., 1-level openings)
[ ] Compare PNN policy to MCCFR equilibrium → measure exploitability
```

---

## PHASE 10 — Coopetition & Partnership Analysis

> **RQ2, RQ4**
> **Models:** Graph Neural Network (PyG) + Shapley values

### T10.1 — Partnership graph
```
[ ] Nodes = players; edges = played-together count
[ ] GNN embedding per player (unsupervised via GraphSAGE)
```

### T10.2 — Shapley attribution
```
[ ] For each deal, compute each player's Shapley contribution to the IMP result
[ ] Aggregate per partnership → chemistry score
[ ] Aggregate per individual → player skill
```

### T10.3 — Coopetition metric
```
[ ] Define: ratio of info-giving-to-partner vs. info-hiding-from-opponent
[ ] Measure per team — correlate with success
[ ] First paper draft: "Coopetition Signatures in Bridge Bidding"
```

---

## PHASE 11 — XAI Layer

> **RQ5** — explainability is one of your central contributions

### T11.1 — SHAP on ENN
```
[ ] Per-bid SHAP values: which bid moved our hand-estimate the most?
[ ] notebooks/11_shap_enn.ipynb
```

### T11.2 — LIME on PNN
```
[ ] For a given auction, generate counterfactual auctions
    → which bid would've changed the final contract?
```

### T11.3 — Attention heat-map visualizer
```
[ ] Render bidding attention matrix as heat-map
[ ] Include in every model card
```

---

## PHASE 12 — LLM for Negotiation Transcripts

> **RQ3, RQ6**
> **Model:** Llama 3 (or GPT-4 via API) for open-ended transcript analysis

### T12.1 — Collect business negotiation transcripts
```
[ ] 100+ from published M&A negotiations, WTO, labour disputes
[ ] Consent & anonymization protocol
```

### T12.2 — Map bridge bids → negotiation moves
```
[ ] Taxonomy: offer, counter-offer, signal, bluff, pass-to-partner
[ ] Llama 3 fine-tuned to tag negotiation transcripts with this taxonomy
[ ] Compare tag distributions between bridge and business
```

---

# YEAR 3 — Integration, Transfer, Dissertation

### PHASE 13 — Cross-domain transfer study
```
[ ] Train coopetition model on bridge → test on business case studies
[ ] Publish: "From Bridge Tables to Boardrooms"
```

### PHASE 14 — Digital-platform simulation
> **Serves RQ6**
```
[ ] Simulated Uber-style marketplace with coopetition agents
[ ] Compare to bridge-trained strategies
```

### PHASE 15 — Dissertation writing
```
[ ] Chapter drafts (one per research question)
[ ] Final defence prep
```

---

## Ongoing (throughout all years)

```
[ ] Literature review — add 5 papers / month to refs.bib
[ ] MLflow experiment tracking for every model run
[ ] Monthly progress update to supervisor (use PhD_Progress_Update template)
[ ] Backup of data/ to external drive weekly
[ ] Git commits with meaningful messages
```

---

## Quick Reference: Which RQ Uses Which Data Field?

| Field | RQ1 | RQ2 | RQ3 | RQ4 | RQ5 | RQ6 |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| bidding sequence | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| 52 cards | ✅ |  | ✅ | ✅ | ✅ |  |
| team / player IDs |  | ✅ | ✅ | ✅ |  | ✅ |
| IMP / score | ✅ | ✅ |  | ✅ |  |  |
| vulnerability | ✅ |  | ✅ |  |  |  |
| negotiation transcripts |  |  | ✅ |  |  | ✅ |

---

## Reminder to self: **why we scraped what we scraped**

- `matches.csv` → bidding, contract, score (RQ1, RQ3)
- `cards.csv` → 52-card deal (RQ1 ENN target, RQ4 Shapley)
- Open + Closed room structure → same hand, different pairs (RQ1 dialects, RQ2 chemistry)
- Multiple championships → dialect diversity (RQ1)
- Multiple years → longitudinal partnerships (RQ2)
