# CLAUDE.md — Project Context for AI Assistant

## Who is Anna?

Anna Ben Shushan — PhD student at LUT University (Finland), self-funded, 2025–2026 cohort.

**Background:** M.Sc. Software Engineering, full-stack developer, academic lecturer in programming, AI Developer Expert course at Tel Aviv University.

**Thesis (working title):** *"From Bridge Tables to Global Markets: AI and Optimization Models of Cooperation, Competition, and Negotiation on Digital Platforms"*

**Timeline:** 3 years + administrative period → Graduation target: Spring 2030
**Current position:** Year 1, Month 1–2 (active data collection)

---

## The Six Research Questions (from Study Plan)

Every task in this project must map to at least one of these. When in doubt, ask: *"Which RQ does this serve?"*

| # | Research Question | What we must measure |
|---|-------------------|----------------------|
| **RQ1** | Can AI model decision-making styles ("bidding dialects") under incomplete information? | Bidding sequences, team/player IDs, contracts, results |
| **RQ2** | How does partner matching (chemistry) affect performance? | Same partnerships over time, consistency metrics |
| **RQ3** | Can bridge bidding be formalized as a negotiation protocol transferable to business? | Bid-by-bid state, information gain per bid, offer/counter-offer structure |
| **RQ4** | How do cooperation + competition (coopetition) dynamics emerge in partnerships? | Shapley values per player, partnership vs. opposition trade-offs |
| **RQ5** | Can XAI make strategic decisions transparent? | Model outputs + SHAP/LIME/attention explanations |
| **RQ6** | How do bridge-derived insights transfer to digital platforms & alliances? | Business case studies, negotiation transcripts (Year 2+) |

---

## Why Bridge? (The Research Case)

- **Imperfect information** — each player sees only 13 of 52 cards (unlike Chess/Go)
- **Partnership dynamics** — cooperate with partner, compete against opponents simultaneously (coopetition)
- **Structured negotiation language** — bidding has ~10⁴⁷ possible sequences, each bid both informs partner and contests opponents
- **Four-player structure** — maps better to multi-stakeholder business dynamics than 2-player games (Poker, Chess)
- **Large labelled datasets** available from tournament archives (EuroBridge, WBF, BBO)

---

## Technical Architecture (Planned)

Aligned with the Methods section of the Study Plan.

| Component | Purpose | Primary Model | Serves RQ |
|-----------|---------|---------------|-----------|
| **ENN** (Estimation NN) | Infer partner's hidden hand from bids observed so far | **Transformer encoder** (multi-head attention over bidding sequence) — following Rong et al. 2019 | RQ1, RQ3 |
| **PNN** (Policy NN) | Choose the next bid given ENN output + own hand | **Transformer decoder + PPO** head | RQ1, RQ4 |
| **DRL self-play** | Train ENN+PNN against themselves | **PPO** (proven in Rong et al.) with **MCCFR** as backup for equilibrium analysis | RQ1, RQ4 |
| **Dialect analysis** | Cluster players/teams by bidding style | **BERT / RoBERTa** fine-tuned on bidding "sentences" + K-means / HDBSCAN clustering | RQ1, RQ2 |
| **Partnership model** | Score partnership chemistry over time | **Graph Neural Network** over player-pair graph + **Shapley values** | RQ2, RQ4 |
| **XAI layer** | Explain every model decision | **SHAP, LIME, attention heat-maps** on ENN/PNN | RQ5 |
| **Negotiation-to-business mapping** | Transfer bridge insights to business negotiation | **LLM (Llama 3 / GPT-4)** for transcript analysis — Year 2 | RQ3, RQ6 |
| **Coopetition game theory** | Quantify cooperative vs. competitive value | **Shapley Value, Core stability** (cooperative game theory) | RQ4 |
| **Experiment tracking** | Reproducibility of every run | **MLflow** (primary) / **Weights & Biases** (optional) | all |

---

## Thesis Framing (Clarified with Supervisors)

Bridge is a **testbed**, not the final subject. The architecture:

```
Empirical work (all on bridge data):          → RQ1, RQ2, RQ4, RQ5
Theoretical discussion (transfer framing):    → RQ3, RQ6
```

**RQ3 and RQ6 are discussion/implication chapters, NOT separate empirical studies.**
The business transfer is argued theoretically + supported by bridge empirics — exactly like AlphaZero (testbed: chess → claim: general game-playing AI).

### Key Research Direction: Risk-Taking Behavior (from Supervisor Nezer, April 2026)

This is now a **central empirical thread** of the thesis, not just a side question.

#### The Core Research Questions (Nezer's call):

1. **Individual risk profile** — Is there a player who *consistently* takes risks? Did they gain or lose from it?
2. **Risk style taxonomy** — Impulsive? Risk-loving? Insurance-preferring? Classify per player.
3. **Game-state driven risk** — As match deficit grows, do players take more risks? (Game theory prediction: yes)
4. **Demographics** — Young/old, women/men — who takes more risks? (aspirational — needs external data)
5. **Tournament standing** — Do 2nd-place teams take more risks trying to close the gap? Do leaders play conservative ("insurance")?
6. **Timing** — Early vs. late in tournament — does risk-taking increase at the end when trailing?
7. **Do decisions even change?** — Maybe some players are rigid regardless of score. That itself is a finding.
8. **Who made the risky bid** — Need player name + position + the specific bid that was the risk moment.

#### Unit of analysis — Individual vs. Pairs (advisor recommendation):
- **Primary unit: PAIRS** (bridge is a partnership game; bids signal between partners)
- **Secondary unit: INDIVIDUAL** — within each pair, the opener vs. responder can be identified from bidding sequence + dealer position
- **DO NOT** try to track both simultaneously in Year 1 — start with pairs, then drill to individual

#### Risk Metric Definitions (to be implemented in src/features/risk_metrics.py):
| Metric | Definition | Counts as |
|--------|-----------|-----------|
| `is_slam_attempt` | Any 5-level+ bid in auction | High risk |
| `is_slam_made` | Final contract at 6/7 level | High risk |
| `is_preempt` | Opening bid at 2/3/4 level | High risk |
| `is_double` | Any Dbl or Rdbl in auction | Medium risk |
| `is_sacrifice` | Took penalty to block opponent | High risk |
| `is_insurance` | Stopped below game despite strength | Low risk / conservative |
| `is_optimistic_game` | Bid 4H/4S/3NT with borderline values | Medium risk |
| `risk_score` | Composite: slam×3 + preempt×2 + double×1 | 0–10 scale |

#### Data Gaps for This Research Direction:

| Data Needed | Status | Source |
|-------------|--------|--------|
| Running IMP within match | ✅ Computable from existing boards | Derive from ns_score per board |
| Tournament standing per round | ✅ Computable from match IMP totals | Derive from matches.csv |
| Who bid what (position) | ✅ Already in bidding string (`N:1H E:Pass...`) | Parse bidding column |
| **Player names per position** | ❌ **CRITICAL GAP** | EuroBridge roster page / WBF PBN |
| Player gender | ❌ Not in EuroBridge | WBF world rankings (external) |
| Player age | ❌ Not in EuroBridge | WBF world rankings (external) |

#### Key insight from Nezer: the bidding string already tells us WHO did WHAT
The existing `bidding` column (`W:- N:1NT E:Pass S:2H | W:Pass ...`) explicitly labels each bid by position.
**We know who made each risky bid** — N/S/E/W — without any new scraping.
**What we're missing:** mapping N/S/E/W to actual player names.

#### VP Scale and Game-Theory Risk Incentive:
- Non-linear IMP→VP conversion creates rational incentive: take risks when losing, be conservative when winning
- This mirrors **Prospect Theory** (Kahneman-Tversky 1979)
- VP tables: https://www.ebu.co.uk/regulation-and-conduct/vp-scales (download 8/10/12/14/16/20 board versions)

---

## Data Volume Targets (Advisor Assessment)

Based on the literature Anna cites (Rong et al. 2019 trained on ~1M deals; Kramár et al. 2022 Diplomacy used millions of games):

| Dataset stage | Rows / deals needed | Status today | Gap |
|---------------|---------------------|--------------|-----|
| **Minimum viable (baseline ML)** | ~50K rows with full bidding + cards | ~39K rows w/ both (78K board-room records) | ⚠️ **Close but thin** |
| **Solid (deep learning ENN/PNN)** | 200K–500K rows | 78K | ❌ **Need 3–6× more** |
| **Competitive (publishable DRL)** | 1M+ deals (via self-play augmentation) | — | Self-play must generate the rest |
| **Dialect analysis (RQ1/RQ2)** | 500+ distinct teams, 5K+ partnerships | ~100 teams in current data | ❌ **Need BBO + more championships** |
| **Negotiation transcripts (RQ3)** | 100+ transcripts | 0 | Year 2 task |

**Bottom line:** The EuroBridge scraper alone is **not enough** for deep RL. We must:
1. Finish scraping remaining EuroBridge competitions (Ostend 2018, Budapest 2016)
2. Add **BBO Vugraph** (adds millions of deals — critical for Year 2)
3. Add **WBF PBN** tournaments (adds world championships, different dialects)
4. **Generate more via self-play** once ENN/PNN are trained (standard DRL practice)

---

## Current Working Directory

```
C:\Users\annaf\OneDrive\Desktop\דוקטורט\דוקטורט אנה\דוקטורט - שילוב בינה מלאכותית\collectBridgeData\
```

Data collection project folder — all scraping, parsing, normalizing, and storage code lives here.

---

## Data Sources (Prioritized)

| Source | Format | Type | Volume potential | Status |
|--------|--------|------|------------------|--------|
| **EuroBridge** (db.eurobridge.org) | HTML → CSV | European Championships (2016–2025) | ~150K rows | ✅ **Scraper working, pipeline working** |
| **WBF** (worldbridgefed.com) | `.pbn` | World Championships | ~500K deals if we scrape all years | Sample (504 deals) downloaded — parser not built |
| **BBO Vugraph** (bridgebase.com) | `.lin` | Massive online tournaments | Millions of deals | Year 2 — parser not built |
| **Self-play generated** | internal | For DRL scale-up | Unlimited | Year 2 |

**Primary source:** EuroBridge (for Year 1). Every board is played by multiple pairs (Open + Closed rooms) — this is **essential** for RQ1 (comparing decision styles on the same hand).

---

## What Is Already Built

| File | What It Does | Status |
|------|-------------|--------|
| `src/downloaders/eurobridge_scraper.py` | Scrapes one match from EuroBridge | ✅ Done |
| `src/downloaders/eurobridge_bulk_scraper.py` | Scrapes all competitions automatically | ✅ Done |
| `src/downloaders/eurobridge_cards_scraper.py` | Adds card holdings (N/S/E/W) via BoardAcross | ✅ Done |
| `src/pipeline.py` | Joins matches + cards, handles even/odd round pairing | ✅ Done |
| `explore_data.py` | Quick dataset explorer + CSV sample | ✅ Done |
| `configs/competitions.yaml` | 5 competitions, 4 categories each | ✅ Done |
| `data/processed/all_matches.parquet` | Master dataset | ✅ 78,584 rows × 40 cols |
| `logs/scrape_log.csv`, `logs/cards_scrape_log.csv` | Audit trails | ✅ Active |
| `src/downloaders/eurobridge_players_backfill.py` | Adds 8 player-name columns to existing CSVs | ✅ Done Apr 2026 |
| `RESEARCH_INSIGHTS.md` | Empirical questions + paper roadmap | ✅ Done Apr 2026 |

**Player name columns added (April 2026):**
`open_north, open_south, open_east, open_west, closed_north, closed_south, closed_east, closed_west`
Source: `<a href="http://www.eurobridge.org/person?qryid=...">` links in the seating diagram table.
Table structure: 4 rows — header / North / (W-compass-E) / South — parsed exactly by position.

---

## Key Data Fields (what each row in the master dataset should have)

**Identity:**
`competition, category, tournament_id, match_id, round, board, room, year`

**Players / Teams:**
`home_team, visiting_team, home_imp, visiting_imp` (player IDs when scrapable — BBO will give us these)

**Deal (the 52 cards):**
`dealer, vulnerability, north_spades, north_hearts, north_diamonds, north_clubs` (and same for S/E/W)

**Auction:**
`bidding` (full sequence string), `contract, declarer, doubled`

**Play & Result:**
`opening_lead, tricks, ns_score, ew_score`

**Quality flags:**
`has_bidding, has_cards` (keep rows with partial data — use flags to filter per analysis)

**Missing (to add later):**
- Full play sequence (trick-by-trick) — BBO only
- Per-bid timing — BBO only
- Convention system (SAYC, Precision, Natural) — sometimes on team page, often absent

---

## Anna's Preferences (How I Should Work With Her)

- Technically capable (full-stack, Python, ML background) — **no hand-holding on code**
- New to **research-grade pipelines** — explain research rationale clearly
- **Clean, modular Python** — no one-off scripts; everything reproducible
- Comments where logic is non-obvious (especially scraping edge cases)
- **Seeds + logging** for every training run (MLflow from day one)
- File hierarchy: `raw → processed → features → models → experiments`
- **All user-facing text and documentation in English**
- Hebrew OK in conversation, but code/docs/commits in English

---

## Files in This Project

| File | Purpose |
|------|---------|
| `README.md` | Plain-English project explanation (start here) |
| `CLAUDE.md` | Context for Claude (this file) — advisor-level overview |
| `ADVISOR_ROADMAP.md` | Month-by-month recommended order of operations |
| `PRD.md` | Product Requirements — what we are building |
| `TASKS.md` | Ordered task list, linked to Research Questions |
| `configs/competitions.yaml` | Competition list for bulk scraper |
| `src/downloaders/` | All scrapers |
| `src/parsers/` | PBN + LIN parsers (to be built) |
| `src/features/` | Feature engineering modules (to be built) |
| `src/models/` | ENN / PNN / dialect models (to be built, Year 1 end) |
| `data/raw/` | Raw scraped data (never modify) |
| `data/processed/` | Cleaned, combined data |
| `data/features/` | Feature-engineered ML-ready data |
| `data/models/` | Trained model checkpoints |
| `logs/` | Scrape + training audit trails |
| `notebooks/` | Exploratory analysis |
| `experiments/` | MLflow runs |
