# TASKS.md

> **NegoPlay Task Backlog**
> Detailed breakdown of work across 10 sessions (5 weeks, 2 sessions/week)
>
> **Session 1 starts: TODAY (May 19, 2026)**

---

## 📊 Status Legend

- 🔲 **TODO** — Not started
- 🟡 **WIP** — Work in progress
- ✅ **DONE** — Completed
- ⚠️ **BLOCKED** — Cannot proceed (dependency)
- ❌ **CANCELLED** — Decided not to do

---

## 🎯 Sprint Overview

| Sprint | Sessions | Dates (approximate) | Theme | Status |
|--------|----------|---------------------|-------|--------|
| Sprint 1 | 1-2 | Week of May 19 | Research & Setup | ✅ DONE |
| Sprint 2 | 3-4 | Week of May 26 | ML Clustering + Skills | 🟡 WIP |
| Sprint 3 | 5-6 | Week of Jun 2 | Agent Construction | 🔲 |
| Sprint 4 | 7-8 | Week of Jun 9 | Dual Simulation | 🔲 |
| Sprint 5 | 9-10 | Week of Jun 16 | Polish & Defense | 🔲 |

---

# 🚀 BEFORE SESSION 1 (Pre-class checklist)

> **Goal:** Arrive at Session 1 (today!) with foundation ready.

### Immediate setup (do TODAY before class)

- [x] 🟡 **Install Google Antigravity IDE**
  - Download from https://antigravity.google/
  - Sign in with personal Gmail
  - Verify Gemini 3 Pro access in Manager View
  - **Deliverable:** Working Antigravity install

- [x] 🟡 **Generate Google Gemini API key**
  - Go to https://aistudio.google.com/apikey
  - Create new API key
  - Save in password manager
  - **Deliverable:** Active API key

- [x] 🟡 **Verify Python 3.11 in WSL2**
  ```bash
  python3.11 --version
  # Should show: Python 3.11.x
  ```

- [x] ✅ **Create GitHub repos**
  - Parent repo: `From-Bridge-Tables-to-Global-Markets` (https://github.com/annafr2/From-Bridge-Tables-to-Global-Markets)
  - Subdirectory: `NegoPlay/`
  - Files are already in the `NegoPlay/` folder
  - **Deliverable:** Live GitHub repo

### Optional (start before, finish after Session 1)

- [ ] 🔲 **Process the 11 papers from NotebookLM** (~15 min)
  - Open `docs/literature.md`, copy the prompt at the bottom into NotebookLM
  - Paste NotebookLM's full response into the "Paper-by-paper notes" section
  - Fill the index table at the top of the file (1 row per paper)
  - **Deliverable:** `docs/literature.md` filled in

---

# 📅 SESSION 1 — Research Foundation & Repo Init
**Date:** May 19, 2026 (TODAY)
**Duration:** ~2 hours
**Goal:** Repo initialized, literature foundation started.

### Tasks

- [x] ✅ **Confirm project scope with Dr. Yoram Segal**
  - Presented NegoPlay overview in meeting (May 26, 2026)
  - Research question validated
  - PRD reviewed and approved
  - **Deliverable:** Sign-off ✅

- [x] ✅ **Initialize repo structure**
  - src/{stage1_clustering,stage2_skills,stage3_agents,stage4_simulate,shared}
  - notebooks/, tests/, results/llm_logs/, data/processed/
  - src/__init__.py, src/sdk.py, all sub-package __init__.py files
  - **Deliverable:** Full folder skeleton ✅

- [x] ✅ **Create `.gitignore`**
  - Covers: .env, .venv, __pycache__, .pytest_cache, LaTeX build artifacts, data/processed CSVs
  - **Deliverable:** .gitignore committed ✅

- [x] ✅ **Create `.env.example`**
  - Template with GOOGLE_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY, BUDGET_CAP_USD
  - **Deliverable:** .env.example committed ✅ (.env itself is gitignored ✅)

- [x] ✅ **Create `requirements.txt`**
  - pandas, numpy, scikit-learn, gensim, shap, google-generativeai, anthropic, openai, pytest, ruff...
  - **Deliverable:** requirements.txt committed ✅

- [ ] 🔲 **Set up venv & install**
  ```bash
  python -m venv .venv
  .venv\Scripts\activate   # Windows
  pip install -r requirements.txt
  ```
  - **Deliverable:** Working venv (local only, not committed)

- [x] ✅ **Test Gemini connection**
  - `tests/test_connection.py` created with API key check + minimal Gemini call
  - **Deliverable:** test_connection.py committed ✅ (run manually to verify)

- [x] ✅ **First commit + push**
  - Multiple commits pushed to https://github.com/annafr2/From-Bridge-Tables-to-Global-Markets
  - **Deliverable:** Live GitHub repo ✅

### End-of-Session Deliverables

- ✅ Approved scope (Dr. Yoram Segal — May 26, 2026)
- ✅ GitHub repo with full folder structure
- ✅ .gitignore, .env.example, requirements.txt
- ✅ tests/test_connection.py ready
- ✅ Commits pushed
- 🔲 venv install (do locally — not tracked in git)

---

# 📅 SESSION 2 — PRD Finalization & Architecture
**Date:** _[fill in — likely May 21 or 22]_
**Duration:** ~2 hours
**Goal:** Complete technical specification, start literature review.

### Tasks

- [ ] 🔲 **Finalize PRD.md**
  - Review current PRD draft
  - Update success metrics with Dr. Rami's input
  - **Deliverable:** Locked PRD v1.0

- [ ] 🔲 **Architecture diagram**
  - Create visual diagram (mermaid or draw.io)
  - Show 4 stages with data flow
  - **Deliverable:** `docs/architecture.md` (with mermaid embedded)

- [ ] 🔲 **Literature review session 1**
  - Use Antigravity's browser agent for parallel search:
    - Agent 1: "AI in Bridge" (DeepStack, AlphaBridge papers)
    - Agent 2: "Multi-Agent LLM Debate" (Du et al. 2023)
    - Agent 3: "LLM negotiation simulation"
  - Find 8 academic papers
  - **Deliverable:** `docs/literature_review.md` v1

- [ ] 🔲 **GitHub project survey**
  - Use Antigravity browser to search GitHub
  - Topics: "bridge AI", "multi-agent simulation", "LLM negotiation"
  - Rank by stars + recency
  - **Deliverable:** `docs/related_projects.md`

- [ ] 🔲 **Write `pyproject.toml`**
  ```toml
  [project]
  name = "negoplay"
  version = "0.1.0"
  description = "Strategic Profiles from Bridge to Business"
  requires-python = ">=3.11"

  [tool.ruff]
  line-length = 100

  [tool.pytest.ini_options]
  testpaths = ["tests"]
  addopts = "--cov=src --cov-report=term-missing"
  ```

- [ ] 🔲 **Create LICENSE file**
  - Standard MIT License
  - Add Anna Ben-Shushan, 2026

### End-of-Session Deliverables

- ✅ PRD.md locked
- ✅ Architecture diagram in markdown
- ✅ 8 papers identified and summarized
- ✅ 5+ related GitHub projects documented
- ✅ Modern Python project config (`pyproject.toml`)

---

# 📅 SESSION 3 — Stage 1: Feature Engineering & Clustering
**Date:** _[fill in]_
**Duration:** ~3 hours (longer — heavy code)
**Goal:** First analytical results: clusters discovered.

### Tasks

- [ ] 🔲 **`src/shared/data_loader.py`**
  - Function: `load_matches(path) → pd.DataFrame`
  - Validation: schema check, encoding (`utf-8-sig`), NaN handling
  - **Tests:** `tests/test_data_loader.py`

- [ ] 🔲 **`src/stage1_clustering/features.py`**
  - Function: `compute_player_features(df) → pd.DataFrame`
  - Compute: `slam_rate`, `success_rate`, `preempt_rate`, `double_rate`,
    `avg_risk_score`
  - Filter to players with ≥20 boards
  - **Tests:** Verify exact values for known player (e.g., McGOWAN Elizabeth)

- [ ] 🔲 **`src/stage1_clustering/clustering.py`**
  - Function: `cluster_players(features, k_range=(2,6))`
  - Implement K-Means with multiple k values
  - Implement HDBSCAN with default params
  - Return: cluster labels + silhouette scores per k
  - **Tests:** Deterministic with `random_state=42`

- [ ] 🔲 **`src/stage1_clustering/validation.py`**
  - Function: `validate_clusters(features, labels) → dict`
  - Compute: silhouette, Davies-Bouldin, p-values vs random
  - Output: structured validation report
  - **Tests:** Sanity check with known synthetic data

- [ ] 🔲 **`notebooks/01_clustering_exploration.ipynb`**
  - EDA on features
  - Visualize cluster separation (PCA, t-SNE)
  - Compare K-Means vs HDBSCAN results
  - Document chosen k and reasoning

### End-of-Session Deliverables

- ✅ `data/processed/player_features.csv` (1,421 rows × 5 features)
- ✅ `data/processed/player_clusters.csv` (with cluster labels)
- ✅ `results/stage1_silhouette.png`
- ✅ `results/stage1_validation.md`
- ✅ Test coverage ≥ 80% for Stage 1 code

### Acceptance Criteria

- [ ] Silhouette ≥ 0.4 for chosen k
- [ ] HDBSCAN finds same k ± 1
- [ ] All tests pass
- [ ] Notebook runnable end-to-end

---

# 📅 SESSION 4 — Stage 2: LLM Skill Extraction
**Date:** _[fill in]_
**Duration:** ~3 hours
**Goal:** 4 named profiles with 5-7 skills each.

### Tasks

- [ ] 🔲 **`src/shared/llm_client.py`**
  - Class: `LLMClient` — unified wrapper with `provider=` arg (default `"gemini"`)
  - Supports: `google-generativeai`, `anthropic`, `openai`
  - Features: retry with exponential backoff, cost tracking per provider, structured logging
  - Output: standardized response format regardless of provider
  - **Tests:** Mocked API tests + 1 smoke test with real key per provider available

- [ ] 🔲 **`src/stage2_skills/chunker.py`**
  - Function: `chunk_games_by_cluster(df, clusters, chunk_size=25)`
  - Smart sampling: diverse games per chunk
  - Format: structured text representation of bridge games
  - **Tests:** Chunk size and diversity validation

- [ ] 🔲 **`src/stage2_skills/extractor.py`**
  - Function: `extract_skills(chunk, profile_id) → dict`
  - Use Gemini Flash 2.0 with `response_mime_type="application/json"`
  - JSON schema enforced
  - Logged calls in `results/llm_logs/`
  - **Tests:** Smoke test with 1 real API call

- [ ] 🔲 **`src/stage2_skills/aggregator.py`**
  - Function: `aggregate_skills(extractions) → list[Skill]`
  - Aggregate: skills appearing in 3+ chunks
  - Deduplicate similar skills (semantic similarity check)
  - Rank by frequency
  - **Tests:** Mock data with known patterns

- [ ] 🔲 **Manual profile naming**
  - Anna reviews top skills per cluster
  - Assigns human-readable names
  - Updates `data/processed/skill_profiles.json`
  - **Deliverable:** 4 named profiles (e.g., Slam Hunter, Insurance Player,
    Bluffer, Doubler)

- [ ] 🔲 **Document profiles**
  - For each profile: name, skills, example player, narrative description
  - **Deliverable:** `docs/profile_descriptions.md`

### End-of-Session Deliverables

- ✅ `data/processed/skill_profiles.json` (4 profiles, fully specified)
- ✅ `docs/profile_descriptions.md`
- ✅ `results/llm_logs/stage2_calls.jsonl` (with costs)
- ✅ Gemini cost so far: < $1

### Acceptance Criteria

- [ ] Each profile has 5-7 interpretable skills
- [ ] Profiles are distinct (low skill overlap)
- [ ] Anna can write a narrative description per profile

---

# 📅 SESSION 5 — Stage 3: Agent Construction
**Date:** _[fill in]_
**Duration:** ~3 hours
**Goal:** 4 working LLM agents with valid behavior.

### Tasks

- [ ] 🔲 **`src/stage3_agents/base_agent.py`**
  - Abstract class: `BaseAgent`
  - Methods: `respond(context) → Response`
  - Properties: `profile_name`, `skills`, `system_prompt`
  - Uses `GeminiClient` from shared
  - **Tests:** Interface compliance

- [ ] 🔲 **`src/stage3_agents/bridge_agent.py`**
  - Class: `BridgeAgent(BaseAgent)`
  - Method: `make_bid(hand, auction_so_far) → str`
  - Handles bidding rules (valid bids only)
  - **Tests:** Verify legal bid generation

- [ ] 🔲 **`src/stage3_agents/nego_agent.py`**
  - Class: `NegotiationAgent(BaseAgent)`
  - Method: `respond_to_offer(scenario, history, current_offer) → Response`
  - Response includes: counter_offer, justification, willing_to_close
  - **Tests:** Output format validation

- [ ] 🔲 **`src/shared/prompts.py`**
  - Centralized prompt templates
  - One template per (profile × domain) = 8 prompts
  - Few-shot examples embedded
  - **Deliverable:** `docs/prompts.md` showing all 8

- [ ] 🔲 **Sanity testing**
  - Manual test: 1 bidding round with each agent
  - Manual test: 1 negotiation turn with each agent
  - Verify: agents behave *differently* in same context
  - **Deliverable:** `experiments/manual_sanity/`

### End-of-Session Deliverables

- ✅ 4 `BridgeAgent` instances ready
- ✅ 4 `NegotiationAgent` instances ready
- ✅ `docs/prompts.md` with full prompt book
- ✅ Sanity test confirms behavioral variance

### Acceptance Criteria

- [ ] All 8 prompts (4 profiles × 2 domains) documented
- [ ] Each agent produces valid JSON output 95%+
- [ ] Behavioral variance > 30% in identical scenarios

---

# 📅 SESSION 6 — Stage 4a: Bridge Simulation (Manual MVP)
**Date:** _[fill in]_
**Duration:** ~3 hours
**Goal:** Understand LLM behavior in bridge before automation.

### Tasks

- [ ] 🔲 **`src/shared/ben_client.py`** ← *new, from related-work analysis*
  - Thin adapter wrapping the `lorserker/ben` engine as an external process
  - BEN runs in its **own venv** (GPL boundary — do NOT import its code)
  - Interface: `BenClient.get_bid(hand, auction) → str`
  - Allows Stage 4 to use BEN as a real, non-LLM opponent/referee
  - **Tests:** Mock subprocess; 1 integration smoke test if BEN is installed

- [ ] 🔲 **`src/stage4_simulate/bridge_game.py`**
  - Class: `BridgeGame`
  - Method: `run(opponent="ben"|"llm") → GameResult`
  - Handles: 4-agent turn-taking, end condition (3 Pass)
  - When `opponent="ben"`: profile-agents bid at one table, BEN bids at the other → IMP score is objective
  - When `opponent="llm"`: all 4 seats are profile-agents (fallback if BEN not installed)
  - Logs: full bidding sequence + reasoning
  - **Tests:** Single game runs to completion in both modes

- [ ] 🔲 **Manual 2-agent runs**
  - Run 5 games: Slam Hunter vs Insurance Player
  - Document observations:
    - Did Slam Hunter bid higher?
    - Did Insurance Player pass earlier?
    - Any hallucinations or rule violations?
  - **Deliverable:** `experiments/manual_simulation_v1/observations.md`

- [ ] 🔲 **Identify LLM behavior issues**
  - Document patterns observed:
    - Sycophancy (agent agreeing too much)
    - Hallucination (invalid bids)
    - Inconsistency (same context, different bid)
  - Adjust prompts to mitigate
  - **Deliverable:** `docs/llm_behavior_notes.md`

- [ ] 🔲 **Refine bidding rules enforcement**
  - Add validator: `is_valid_bid(bid, auction)`
  - Add retry logic: if invalid bid, re-prompt
  - **Tests:** Reject illegal bids correctly

### End-of-Session Deliverables

- ✅ 5 manually-run games documented
- ✅ Behavior notes with mitigations
- ✅ Refined prompts (v2)

### Acceptance Criteria

- [ ] Each game completes (no infinite loops)
- [ ] Bidding rules enforced
- [ ] Profile differences observable to human reader

---

# 📅 SESSION 7 — Stage 4a Automation: 50 Bridge Games
**Date:** _[fill in]_
**Duration:** ~3 hours
**Goal:** Full bridge simulation dataset.

### Tasks

- [ ] 🔲 **`src/stage4_simulate/bridge_runner.py`**
  - Function: `run_bridge_tournament(n_games=50)`
  - Generate diverse hands (realistic distributions)
  - Run all 6 unique pairings × ~9 games each
  - Parallel execution where possible (Antigravity Manager View!)
  - **Tests:** Tournament runs to completion

- [ ] 🔲 **Outcome scoring**
  - Implement: `score_game(result) → dict`
  - Metrics: declarer success, IMP score, contract value
  - **Tests:** Known outcomes give expected scores

- [ ] 🔲 **Win-rate analysis**
  - Per profile: % games won, average score
  - Identify clear winners and losers
  - **Deliverable:** `results/bridge_winrates.csv`

- [ ] 🔲 **Run the tournament**
  - Execute 50 games (~$0.25 budget)
  - Monitor cost in real-time
  - Save full logs
  - **Deliverable:** `results/bridge_simulations.jsonl`

### End-of-Session Deliverables

- ✅ 50 bridge games simulated
- ✅ Win-rate data per profile
- ✅ Visualization of profile performance
- ✅ Cost so far: < $2

### Acceptance Criteria

- [ ] 50 games completed
- [ ] Each profile played at least 25 games
- [ ] Win rates statistically distinguishable

---

# 📅 SESSION 8 — Stage 4b+c: Negotiation Simulation + Alignment
**Date:** _[fill in]_
**Duration:** ~3 hours (CRITICAL SESSION)
**Goal:** The "Holy Grail" — answer the research question.

### Tasks

- [ ] 🔲 **Negotiation scenario design**
  - Write 4 scenarios in detail (Anna):
    1. M&A acquisition
    2. JV equity split
    3. B2B SaaS pricing
    4. Government tender bidding
  - Each: context, opening positions, win conditions
  - **Deliverable:** `data/scenarios/negotiation_scenarios.json`

- [ ] 🔲 **`src/stage4_simulate/negotiation.py`**
  - Class: `NegotiationSession`
  - Method: `run(scenario, agent_a, agent_b) → SessionResult`
  - End conditions: deal closed, walkout, max rounds
  - **Tests:** Session terminates properly

- [ ] 🔲 **Run 48 negotiations**
  - 4 scenarios × 6 pair combinations × 2 runs = 48
  - ~$0.25 budget
  - **Deliverable:** `results/negotiation_simulations.jsonl`

- [ ] 🔲 **`src/stage4_simulate/alignment.py`**
  - Function: `compute_alignment(bridge_results, nego_results)`
  - Compute: per-profile win rates in both domains
  - Spearman correlation across profiles
  - Statistical significance test
  - **Deliverable:** `results/alignment_analysis.png`

- [ ] 🔲 **Write alignment report**
  - Present numbers honestly
  - If ≥70%: celebrate, frame as evidence
  - If 50-69%: nuanced discussion
  - If <50%: frame as valid null result
  - **Deliverable:** `results/alignment_report.md` ⭐ **THE OUTPUT**

### End-of-Session Deliverables

- ✅ 48 negotiation sessions simulated
- ✅ Cross-domain alignment computed
- ✅ Final research answer documented

### Acceptance Criteria

- [ ] All 48 sessions run successfully
- [ ] Alignment score computed with p-value
- [ ] Honest framing regardless of result

---

# 📅 SESSION 9 — Polish, Business Plan, Video
**Date:** _[fill in]_
**Duration:** ~3 hours
**Goal:** Project becomes presentable.

### Tasks

- [ ] 🔲 **Business plan section**
  - TAM/SAM/SOM for negotiation intelligence market
  - Token economics analysis
  - Competitive landscape
  - **Deliverable:** `docs/business_plan.md`

- [ ] 🔲 **Final presentation deck**
  - 12-15 slides covering:
    - Problem & motivation
    - Methodology
    - Results
    - Discussion & limitations
    - Future work
  - **Deliverable:** `docs/presentation.pptx`

- [ ] 🔲 **Promotional video**
  - Script: 60-90 second hook
  - Use Nano Banana / Veo or similar tool
  - **Deliverable:** `assets/promo_video.mp4`

- [ ] 🔲 **README polish**
  - Add final result numbers
  - Add result visualizations
  - Update status badges
  - **Deliverable:** Updated `README.md`

- [ ] 🔲 **Code quality pass**
  - Run `ruff check --fix`
  - Run `pytest --cov`
  - Fix any failing tests
  - Achieve 60%+ overall coverage
  - **Deliverable:** Green CI

### End-of-Session Deliverables

- ✅ Business plan document
- ✅ Presentation deck
- ✅ Promo video
- ✅ Polished README
- ✅ Clean test suite

---

# 📅 SESSION 10 — Defense Rehearsal
**Date:** _[fill in]_
**Duration:** ~2-3 hours
**Goal:** Ready for final defense.

### Tasks

- [ ] 🔲 **Full presentation dry-run**
  - Time it (target: 15 minutes)
  - Record yourself (optional)
  - Identify weak transitions

- [ ] 🔲 **Q&A preparation**
  - Anticipate 10 hard questions
  - Write 2-sentence answers for each
  - **Deliverable:** `docs/defense_qa.md`

- [ ] 🔲 **Backup plan**
  - Record demo video (in case live demo fails)
  - Create backup branch with stable code
  - Test full pipeline on clean clone
  - **Deliverable:** Backup video + branch

- [ ] 🔲 **Documentation final pass**
  - Update all "Last updated" dates
  - Verify all internal links
  - Spell-check
  - **Deliverable:** Clean docs

- [ ] 🔲 **Repository hygiene**
  - Delete temporary branches
  - Tag final version: `v1.0`
  - Update GitHub repo description
  - Add topics/tags for discoverability
  - **Deliverable:** Production-ready repo

### End-of-Session Deliverables

- ✅ Presentation rehearsed
- ✅ Q&A document ready
- ✅ Backup materials prepared
- ✅ Repository tagged v1.0

---

# 🎁 STRETCH GOALS (If Time Permits)

These are nice-to-haves, not commitments.

### Stretch 1: Word2Vec on Bidding Sequences
- Train Word2Vec on 46K bidding sequences
- Visualize bid embeddings (similar bids cluster)
- **Why:** Adds NLP unit + multimodal evidence for PhD

### Stretch 2: LSTM Risk Predictor
- Train LSTM on bidding sequences → risk score
- Compare to feature-based predictions
- **Why:** Adds Deep Learning unit

### Stretch 3: 4-Agent Bridge Simulation
- Instead of 2v2, run all 4 profiles simultaneously
- More realistic but more complex
- **Why:** Closer to real bridge

### Stretch 4: SUNO Theme Song
- Generate musical theme for presentation
- **Why:** Per Dr. Segal's framework

### Stretch 5: Antigravity Manager View workflows
- Document multi-agent dev workflows used
- **Why:** Showcase IDE for portfolio/demo

---

# 📋 BLOCKED / PARKING LOT

Items that came up but aren't actionable now.

- 🅿️ **Real negotiation dataset** — Year 2 of PhD
- 🅿️ **Multi-cultural validation** — Future research
- 🅿️ **Real-time API for users** — Post-research, productization
- 🅿️ **WBridge5 as secondary opponent** — Via OpenSpiel PR #766, when BEN benchmark is insufficient (PhD Year 2)

---

# 📊 Progress Tracker

> Update this table after each session.

| Sprint | Sessions | Status | % Complete | Notes |
|--------|----------|--------|------------|-------|
| Sprint 1 | 1-2 | 🟡 WIP | 0% | Starting today (May 19) |
| Sprint 2 | 3-4 | 🔲 TODO | 0% | — |
| Sprint 3 | 5-6 | 🔲 TODO | 0% | — |
| Sprint 4 | 7-8 | 🔲 TODO | 0% | — |
| Sprint 5 | 9-10 | 🔲 TODO | 0% | — |

**Last updated:** May 19, 2026 (project kickoff)
**Next checkpoint:** End of Session 1 (today)

---

# 💡 Working Conventions

### Daily routine (suggested)
1. **Open Antigravity** in `negoplay/` workspace
2. **Read CLAUDE.md** if starting fresh
3. **Check TASKS.md** for current session
4. **Use Manager View** for parallel work where possible
5. **Make one small commit** before moving on
6. **Update task status** before closing for the day

### Session retrospective (recommended)
After each session, write 3 lines in `docs/retrospective.md`:
- ✅ What worked
- ⚠️ What was hard
- 🎯 Adjustment for next time

### When in doubt
- Re-read **PRD.md** for *what*
- Re-read **CLAUDE.md** for *how*
- Re-read **TASKS.md** for *when*
- Ask the AI agent for help (it knows the structure)

### Cost monitoring
After each Stage 2-4 task, check `results/llm_logs/`:
- If cumulative > $5: review which calls are expensive
- If cumulative > $10: pause and optimize prompts
- Hard cap: $15 (set in `.env`)
