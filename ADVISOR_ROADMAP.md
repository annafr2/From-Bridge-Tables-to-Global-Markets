# ADVISOR_ROADMAP.md — Recommended Order of Operations

**From:** Your AI PhD advisor
**To:** Anna
**Date:** April 2026 (PhD Month 1–2)
**Purpose:** The exact sequence I'd recommend you follow, month by month, so that:
1. Every piece of work directly serves one of your 6 Research Questions
2. You hit publishable results by end of Year 2
3. You have a defendable dissertation by Spring 2030

> If in doubt, work top-to-bottom. Don't skip ahead.

---

## GUIDING PRINCIPLE

> **"Data → Features → Simple Models → Deep Models → XAI → Transfer"**

Do **not** start training a Transformer in Month 3. Do **not** try to solve all 6 RQs simultaneously. Build the stack one layer at a time. Every deep model is a stretch goal — a working baseline + good data is *always* more valuable at this stage than a half-trained neural network.

---

## MONTH-BY-MONTH PLAN

### 🟦 Month 1–2 (NOW — April–May 2026) — FOUNDATION

**Goal:** A validated dataset of ≥150K rows from EuroBridge, ready for feature engineering.

1. **[THIS WEEK]** Fix the Madeira 2022 scraper (T3.1)
   - The HTML is laid out differently — compare to Herning 2024
   - Add a parser variant; don't break the existing one
2. **Run the bulk scraper on any missing competitions** (T3.2)
   - Poznan 2025, Ostend 2018, Budapest 2016
3. **Re-run the pipeline** → expect ≥150K rows
4. **Write a data-coverage report** (T3.3)
   - Per competition: rows, % with bidding, % with cards, % with contract
   - Save as `data/DATA_COVERAGE_REPORT.md`
5. **Normalize bidding sequences** (T3.4)
   - Parse the raw bidding string into a structured `list[dict]`
   - This is a one-time investment — everything downstream depends on it
6. **Monthly supervisor update** — attach data-coverage report

> 🚩 **Deliverable at end of Month 2:** `data/processed/all_matches_v2.parquet` with ≥150K rows, fully normalized bidding, all quality flags populated.

---

### 🟦 Month 3 — FEATURE ENGINEERING

**Goal:** Every deal has ~40 engineered features ready for ML.

1. **Build `src/features/build_features.py`** (T4.1 – T4.4)
   - Hand features first (HCP, distribution)
   - Then auction features (length, competitive, jumps)
   - Then contract features (level, suit, game/slam)
   - Then partnership features (IDs, consistency)
2. **Save `data/features/deals_features.parquet`** (T4.5)
3. **Write the `FEATURE_CARD.md`** — document every column
4. **Notebook 01: data overview** (T5.1) — sanity check every feature

> 🚩 **Deliverable:** `deals_features.parquet` + `FEATURE_CARD.md`. You should be able to run `df.describe()` and nothing looks weird.

---

### 🟦 Month 4 — FIRST EXPLORATORY SIGNALS

**Goal:** First evidence that bridge-bidding-as-data tells a real research story.

1. **Notebook 02: bidding dialects preview** (T5.2) — **THIS IS RQ1 DEMO DAY**
   - Pick 20 "same-hand-many-pairs" boards
   - Show visually: same cards → different auctions
   - Compute first dialect metrics (avg auction length, % competitive)
2. **Notebook 03: partnership consistency** (T5.3) — **RQ2 DEMO DAY**
   - Top-20 partnerships by board count
   - Variance in their scores on identical boards
3. **Show these notebooks to your supervisor** — this is your first real research artefact

> 🚩 **Deliverable:** Three notebooks, printed as PDF, in a folder you email to your supervisor. Title of the email: *"First exploratory signals — bidding dialects and partnership consistency"*.

---

### 🟦 Month 5 — SIMPLE BASELINES (not deep models yet!)

**Goal:** Before the Transformer, prove the features predict *something*.

**Baselines to build (all using scikit-learn, takes 1 week each):**

1. **Contract-level classifier** — given hand HCP + distribution → predict contract level
   - *Model:* Random Forest (fast, interpretable)
2. **Made-contract predictor** — given contract + hands → will it make?
   - *Model:* Gradient Boosting (XGBoost / LightGBM)
3. **Bidding aggression cluster** — unsupervised clustering of teams on basic features
   - *Model:* K-means on (avg_auction_length, % competitive, % jumps)

All in `notebooks/05_simple_baselines.ipynb`. Log everything to MLflow.

> 🚩 **Deliverable:** MLflow tracked runs, baseline scores written up in `notebooks/05_simple_baselines_results.md`. These are your **numerical reference points** for all future deep models.

---

### 🟦 Month 6 — LITERATURE MILESTONE + YEAR-1 MIDPOINT REVIEW

**Goal:** Make sure your research narrative still holds up.

1. **Re-read your Study Plan** — are RQ1–RQ6 still the right questions?
2. **Literature review** — read & annotate:
   - Rong, J. et al. (2019) — ENN/PNN bridge paper (the foundation)
   - Silver et al. (2018) — general game-playing with MCTS
   - Kramár et al. (2022) — DeepMind Diplomacy (closest analog)
   - Vaccaro et al. (2025) — partnership & coopetition
   - 5 more papers on negotiation / coopetition
3. **Write Year-1 midpoint report** — 5 pages:
   - Research questions (unchanged from Study Plan? Refined?)
   - Data collected (volumes, sources, quality)
   - Exploratory signals (what notebooks show)
   - Baselines (what features predict what)
   - Plan for Months 7–12

> 🚩 **Deliverable:** A 5-page PDF. Share with supervisor. This is the checkpoint that decides whether Year 1 went well.

---

### 🟦 Month 7–8 — BIDDING DIALECT MODEL (first real ML contribution)

**Goal:** First publishable result — **bidding dialects clustered via BiddingBERT**.

**This is Phase 6 in TASKS.md.**

1. Build bid tokenizer (T6.1)
2. Pre-train BiddingBERT on ~150K auctions (T6.2)
3. Extract team-level embeddings
4. HDBSCAN cluster + label clusters qualitatively (T6.3)
5. Notebook 04: `dialect_clusters.ipynb`

**Why BERT and not a big custom Transformer?**
- Bidding sequences are short (≤30 tokens) — BERT-style masked LM is the right tool
- You can start small (4 layers, 2M params) and scale up later
- HuggingFace gives you 90% of the scaffolding — no need to write a Transformer from scratch yet

> 🚩 **Deliverable:** Figure: 2D UMAP of team vectors coloured by cluster. Table: which teams are in each cluster. Write-up: **draft of first workshop paper**.

---

### 🟦 Month 9–10 — ENN BASELINE

**Goal:** First replication of Rong et al. 2019 on your own data.

**This is Phase 7 in TASKS.md.**

1. Generate (partial-auction, full-hands) training examples (T7.1)
2. Build 6-layer Transformer encoder ENN (T7.2)
3. Train + evaluate — target: top-5 card-ownership accuracy > 70%
4. Compare to a naïve baseline (uniform + obvious inferences)

**Advisor warning:** This is the *hardest technical task of Year 1*. If your data volume is still low (<100K full-bidding+cards rows), the ENN will not train well. **Don't push it** — if you hit trouble, document the failure mode and defer to Year 2 after BBO data is added.

> 🚩 **Deliverable:** ENN checkpoint in `data/models/enn_v1.pt` + evaluation report. If it doesn't train well, **that is still a valid result** — document *why*.

---

### 🟦 Month 11 — SECOND SUPERVISOR CHECK-IN

**Goal:** Confirm direction before committing Year 2 to PNN + DRL.

1. Update `TASKS.md` with what's done, what slipped, what's added
2. Re-assess data volume — will EuroBridge alone carry you, or is BBO urgent?
3. Draft: *"Year 1 Results and Year 2 Plan"* — 8 pages

---

### 🟦 Month 12 — YEAR-1 CLOSE

**Goal:** End Year 1 with:
- ✅ 150K+ clean deals
- ✅ Engineered features
- ✅ Dialect model (working, with clusters)
- ✅ ENN baseline (working or known-failing-with-reason)
- ✅ 2 notebooks demoing RQ1 and RQ2 signals
- ✅ First draft workshop paper
- ✅ MLflow history of every run

**Celebrate. You earned it.**

---

## YEAR 2 — SCALE + DEEP MODELS

(High-level — detailed tasks live in TASKS.md Phases 8–12.)

| Quarter | Focus |
|---------|-------|
| **Y2 Q1 (M13–15)** | BBO LIN parser + WBF PBN parser → scale to 1M+ deals |
| **Y2 Q2 (M16–18)** | Re-train ENN on 10× more data + build PNN + self-play scaffold |
| **Y2 Q3 (M19–21)** | PPO self-play + coopetition metrics (Shapley, GNN partnerships) |
| **Y2 Q4 (M22–24)** | XAI layer (SHAP / LIME) + submit first journal paper |

**Year 2 milestone:** A full bridge AI (ENN + PNN + DRL) with quantified dialects, partnership chemistry, and XAI explanations. One submitted paper minimum.

---

## YEAR 3 — TRANSFER + DISSERTATION

| Quarter | Focus |
|---------|-------|
| **Y3 Q1 (M25–27)** | LLM-based negotiation-transcript tagger (RQ3) |
| **Y3 Q2 (M28–30)** | Digital-platform coopetition simulation (RQ6) |
| **Y3 Q3 (M31–33)** | Cross-domain transfer study: Bridge → Business |
| **Y3 Q4 (M34–36)** | Dissertation writing, defence prep |

---

## CRITICAL HABITS (from Day 1)

1. **MLflow every run.** No untracked training. Ever.
2. **One Jupyter notebook = one research question.** Don't let them become drawers.
3. **Weekly commits.** Even if just `notes.md`. It tracks your thinking.
4. **Data backups.** `OneDrive` is not enough. External drive, monthly.
5. **Read one paper per week.** 20 minutes is fine. Add it to `references/refs.bib`.
6. **Monthly supervisor email.** Even if nothing new. Keep the relationship warm.
7. **Experiments/ folder.** Every failed attempt documented. Failure is data.

---

## ANSWERS TO THE QUESTIONS YOU ASKED

### *"Is the data I have enough?"*

**No, but you're close for Year 1 and need more for Year 2.**

| Stage | Need | Have | Verdict |
|-------|------|------|---------|
| Year 1 dialect model | 50–100K auctions | 78K (39K with both cards+bidding) | ⚠️ Thin but workable after Madeira fix |
| Year 1 ENN baseline | 100K+ with cards | 39K | ❌ Will train poorly — fix Madeira + add BBO |
| Year 2 full DRL | 1M+ | — | ❌ Must add BBO Vugraph |
| Year 3 negotiation transfer | 100+ transcripts | 0 | Year 2 collection task |

### *"Which models should I use?"*

(Summary; full table in CLAUDE.md.)

| Job | Model | Why |
|-----|-------|-----|
| Hidden-hand inference | **Transformer encoder** (ENN) | Rong et al. 2019 showed it works; attention is perfect for variable-length bids |
| Bid selection | **Transformer decoder + PPO** | Policy learning with self-play; Rong et al. architecture |
| Dialect clustering | **BERT** + HDBSCAN | Short "sentences", MLM pre-training, unsupervised clusters |
| Partnership chemistry | **GNN + Shapley** | Graph = player-pair network; Shapley = coopetition theory |
| Negotiation transcripts | **Llama 3 / GPT-4** | Long-form text, open domain |
| Explainability | **SHAP + LIME + attention maps** | Standard XAI stack |

### *"What's the recommended order?"*

**Top of this file.** Month-by-month. Trust the sequence; don't skip to the deep models.

---

## WHEN TO COME BACK TO ME (your AI advisor)

- ✅ When a scraper breaks
- ✅ When a model doesn't train
- ✅ When you need to decide between two architectures
- ✅ When you want a code review on a new module
- ✅ When writing the monthly supervisor email
- ❌ When you just want affirmation — do the work first, then show results

---

**Good luck, Anna. This is a real thesis. Stick to the plan and you'll graduate on schedule.**
