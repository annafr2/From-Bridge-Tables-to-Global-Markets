# PRD: NegoPlay

> **Product Requirements Document**
> Version 2.0 · Anna Ben-Shushan · 2026
> *Updated May 2026 after Stage 1 results — see Section 4 (H1 Revision) and Section 6 (Stage 1)*

---

## 1. Executive Summary

**NegoPlay** is a research system that:
1. Discovers decision-making profiles from 149,208 elite bridge tournament hands
2. Builds LLM agents instantiated with these profiles
3. Tests whether the agents exhibit consistent behavior across bridge games and business negotiation simulations

**Primary deliverable:** Empirical answer to whether ≥70% of "winners" in bridge simulations are also "winners" in parallel negotiation simulations using the same agents.

**Status:** Course final project (AI Development Expert) + PhD baseline (LUT University)

**Tech stack:** Python 3.11 · scikit-learn · gensim · Google Gemini Flash 2.0 (default LLM) · Anthropic Claude + OpenAI (optional, for cross-model validation) · built in Google Antigravity IDE

---

## 2. Problem Statement

### The Research Gap
Business negotiation suffers from a fundamental data scarcity problem:
- Real negotiation transcripts are proprietary and protected
- Outcomes are rarely measurable in standardized form
- No public datasets allow training of behavioral models

Bridge, by contrast, offers:
- 149K+ structured decision points with measurable outcomes
- Clear win/loss signals (contract success, IMP scores)
- Tournament-grade decision-making by experts

**The question:** Can we train AI behavioral models in the clean bridge environment and transfer them to the noisier domain of negotiation?

### The User Need
Three audiences:
1. **Academic researchers** investigating cross-domain behavioral transfer
2. **Business consultants** seeking AI-driven negotiation intelligence (future product)
3. **AI/ML practitioners** exploring multi-agent LLM systems

---

## 3. Solution Overview

### Hybrid ML + LLM Pipeline

```
INPUT: 149K bridge boards (CSV)
   │
   ├─→ Stage 1: Profile Discovery (scikit-learn + extreme percentile) ✅ DONE
   │   Output: 4 extreme-percentile profiles (563 players, May 2026)
   │
   ├─→ Stage 2: LLM Skill Extraction (Gemini Flash by default)
   │   Output: Named profiles with 5-7 skills each
   │
   ├─→ Stage 3: Agent Construction (Gemini Flash by default; Claude/OpenAI for cross-model validation)
   │   Output: 5 LLM agents, one per profile
   │
   └─→ Stage 4: Dual Simulation (LLM agents + LLM negotiation)
       Output: Alignment score (target ≥70%); cross-provider robustness check
```

### Why Hybrid?
- **ML alone (clustering):** Statistically rigorous, but cannot simulate behavior
- **LLM alone (agents):** Behaviorally rich, but lacks empirical grounding
- **Hybrid:** Empirical baseline (from ML) + behavioral simulation (from LLM)

### LLM Provider Policy
**Gemini Flash 2.0 is the default**, for cost reasons:
- **Cost:** Cheapest production-grade LLM ($0.30/M output)
- **Sufficient quality:** Flash 2.0 handles structured JSON output and reasoning at MVP scale
- **IDE integration:** Antigravity uses Gemini natively, simplifying workflow

**Claude and OpenAI are available** when justified:
- **Cross-model validation runs** — running the same profile on Gemini + Claude + OpenAI strengthens the alignment finding and makes it harder to dismiss as a single-provider artifact (key for PhD paper credibility)
- **Quality-critical synthesis** — final paper writing, hard reasoning checks
- **Capability gaps** — if Gemini measurably underperforms on a specific task

All calls route through `src/shared/llm_client.py` with a `provider=` argument. Costs are logged per-provider with a $20 alert and $50 hard cap across all providers combined.

---

## 4. Research Question & Hypotheses

### Primary Research Question

> **RQ:** *Can LLM agents built from automatically-discovered bridge decision-making profiles exhibit behavioral consistency (≥70% alignment) between winning in bridge simulations and winning in parallel business negotiation simulations?*

### Hypotheses

**H1 (Statistical) — ORIGINAL:** K-Means clustering on 5 decision features will produce 3-5 stable profiles with Silhouette ≥ 0.4 and p < 0.05 versus random clustering.

> **⚠️ Protocol Deviation (May 2026) — Stage 1 Results:**
> K-Means, HDBSCAN, and GMM all failed to produce stable clusters (best Silhouette = 0.15, well below the 0.4 target). This is not a pipeline error — it is a finding: **elite tournament players form a statistical continuum, not discrete clusters.** This parallels known results in expertise research (experts converge toward optimal behavior, reducing inter-player variance).
>
> **H1 (Revised):** Elite bridge players do not form discrete behavioral clusters. Instead, they occupy a statistical continuum. Meaningful profiles can be identified via **extreme-percentile profiling**: players in the top 10% on a defining behavioral axis (slam_rate, partscore_rate, penalty_double_rate, nt_rate) represent the behavioral "tails" of the expert distribution.
>
> **Method used:** For each of 4 defining axes, compute z-scores across all 563 qualifying players (≥50 declared boards AND ≥50 bidding boards). Assign a player to a profile if their z-score on that axis is (a) above the 90th percentile AND (b) passes a one-sided binomial test at p < 0.05 vs population baseline. All remaining players = Generalist (baseline).
>
> **Result:** 4 extreme profiles identified — Slam Hunter (n=20), Insurance Player (n=21), Fighter (n=37), NT Specialist (n=17), Generalist (n=468). Key ratios: Slam Hunter 1.84× on slam_rate; Fighter 1.55× on penalty_double_rate.
> *(v2.0 numbers of n=64/60/66/53 were based on min_boards=20 — statistically unreliable for rare-event rates; revised to min_boards=50 + binomial test)*
>
> **This continuum finding is itself a publishable contribution** and will be reported as a primary result in the PhD paper.

**H2 (Behavioral):** LLM agents instantiated with these profiles will exhibit measurably different behaviors in bridge bidding (per-profile win rate variance > 15%).

**H3 (Transfer — Main):** The win-rate ranking across profiles will correlate ≥ 0.7 (Spearman) between the bridge and negotiation domains.

**Null Hypothesis:** No correlation > 0.5 between domains — this would also be a valid research finding.

### Anchor Papers (per Dr. Segal's framework)

Two anchor papers ground NegoPlay's methodology:

- **⚓ Talwadker et al., 2022 — CognitionNet** — methodological anchor for Stage 1 (sequence-based player-style clustering with a Seq2Seq + classifier).
- **⚓ Rong et al., 2019 — Competitive Bridge Bidding with DNNs** — methodological anchor for Stage 3 (ENN + PNN inference-then-decision split; benchmark target — beats Wbridge5).
- **💡 Lockett et al., 2007 — Evolving Explicit Opponent Models** — theoretical foundation (model unknown counterparts as a mixture of cardinal profiles — NegoPlay's core premise).

Full literature: [`docs/literature.md`](docs/literature.md) (17 papers, NegoPlay-focused) + [`../PHD_LITERATURE.md`](../PHD_LITERATURE.md) (9 papers, broader PhD background).

### Baselines (what we measure against)

| Domain | Baseline | Purpose |
|--------|----------|---------|
| Bridge | Random bidder | Sanity floor |
| Bridge | **Generic LLM agent, no profile prompt** | Main comparison — does profile-conditioning matter? |
| Bridge | `lorserker/ben` engine | Objective non-LLM benchmark (Stage 4 v2 / PhD Year 2) |
| Negotiation | Generic agent, no profile prompt | Same scenario, no behavioral conditioning |
| Cross-domain | Spearman ρ across profile rankings | Headline metric — target ρ ≥ 0.7 |

---

## 5. Target Users

### Primary: **Anna (Researcher)**
- **Need:** Empirical foundation for Paper 1 of PhD
- **Success criterion:** Reproducible results, clear methodology, defendable in academic settings
- **Pain points:** Limited time (5 weeks), limited budget, simultaneous course requirements

### Secondary: **Dr. Rami (Course Examiner)**
- **Need:** Demonstration of course mastery
- **Success criterion:** Clear use of 5+ course units, working agents, measurable outcomes
- **Pain points:** Must evaluate technical quality, not just academic value

### Tertiary: **Prof. Jari (PhD Supervisor)**
- **Need:** Evidence of research progress aligned with study plan
- **Success criterion:** Project advances Year 1 objectives, methodologically sound
- **Pain points:** Limited time for review, expects scientific rigor

---

## 6. Features (Detailed)

### Stage 1: Profile Discovery ✅ COMPLETE (May 2026)

**Description:** Statistical profiling of 563 elite bridge players into 4 extreme behavioral profiles + 1 Generalist baseline via extreme-percentile method with binomial significance test. Original plan was K-Means clustering — see H1 revision above for why the method changed.

**Inputs:**
- `data/raw/all_matches_full.csv` (149,208 rows)

**Processing (as executed — v2.1 after Dr. Rami preprocessing audit):**
1. Filter to players with ≥50 declared boards AND ≥50 boards with bidding data → **563 players**
2. Compute **8 features** per player (selected after variance + correlation filter):
   - `slam_rate`, `partscore_rate`, `nt_rate`, `double_rate`, `opening_rate`, `preempt_rate`, `intervention_rate`, `penalty_double_rate`
3. **Full preprocessing pipeline** (`src/stage1_clustering/preprocessing.py`): variance filter (CV ≥ 0.10), correlation filter (|r| ≤ 0.70), Mahalanobis outlier detection (α=0.01), RobustScaler, auto-PCA
4. **Attempted K-Means** (k=2–8): best Silhouette = 0.24 → below 0.40 target → rejected
5. **Attempted HDBSCAN**: 0 clusters found in every configuration → rejected
6. **Attempted GMM + BIC**: max silhouette = 0.22 → rejected
7. **5-pipeline audit** (`notebooks/preprocessing_comparison.py`) confirmed continuum across all configurations
8. **Extreme Percentile Profiling** (adopted method): top 10% per defining axis + binomial test (p < 0.05) → 4 extreme profiles

**Outputs:**
- `data/processed/player_features.csv`
- `data/processed/player_profiles.csv` ← replaces `player_clusters.csv`
- `docs/images/pca_scatter.png`
- `docs/images/radar_profiles.png`
- `docs/images/feature_bars.png`

**Results (v2.1 — after expert review by Nezer, May 2026):**

| Profile | n | % | Defining feature | Profile mean | Generalist mean | Ratio |
|---------|---|---|-----------------|--------------|----------------|-------|
| Slam Hunter | 20 | 3.6% | slam_rate | 0.101 | 0.055 | 1.84× |
| Insurance Player | 21 | 3.7% | partscore_rate | 0.684 | 0.570 | 1.20× |
| Fighter | 37 | 6.6% | penalty_double_rate | 0.131 | 0.085 | 1.55× |
| NT Specialist | 17 | 3.0% | nt_rate | 0.385 | 0.282 | 1.36× |
| Generalist | 468 | 83.1% | — | — | — | baseline |

> **⚠️ Sample-size revision (May 2026):** The v2.0 pipeline used `min_boards=20`
> and reported 64 Slam Hunters with a 2.8× ratio. PhD supervisor Nezer (an
> expert bridge player) noted that 20 declared boards is too few to estimate
> rare-event rates like slam: a player with 20 boards and 2 slams has
> slam_rate = 10% with 95% CI [1.2%, 31.7%] — overlapping the baseline.
>
> The revised pipeline (v2.1) requires:
> - `min_boards ≥ 50` AND `min_bidding_boards ≥ 50`
> - One-sided binomial test against the population baseline at p < 0.05
>
> The Slam Hunter cohort fell from 64 to 20 players, but those 20 are
> statistically robust: median `n_declared` = 216, minimum = 69, all
> p-values < 0.05. This is a healthier basis for Stages 2-4.

**Success criteria (revised v2.1):**
- ✅ 4 extreme profiles with statistically distinct defining features (ratios 1.20×–1.84×)
- ✅ Binomial significance test confirms every assignment (p < 0.05)
- ✅ Continuum finding confirmed across **5 pipeline configurations** (Dr. Rami preprocessing audit, May 2026)
- ✅ Generalist baseline identified (83% of population, n=468)
- ✅ 81 pytest tests passing (52 Stage 1 + 29 bridge_validator)
- ✅ Bridge Expert Validation Skill added (`src/shared/bridge_validator.py` + `.claude/commands/bridge-expert.md`)
- ⚠️ Original silhouette target (≥0.4) NOT met — explained by continuum structure, not pipeline error

---

### Stage 2: Skill Extraction

**Description:** Gemini Flash 2.0 analyzes game samples from each profile to identify characteristic skills.

**Inputs:**
- 5 profiles from Stage 1 (4 extreme + 1 Generalist)
- 20-30 game samples per chunk
- 5-10 chunks per profile

**Processing:**
1. For each cluster, extract diverse game samples
2. Chunk into batches of 20-30 games
3. Send to Gemini Flash 2.0 with structured JSON output:
   ```
   "Analyze these tournament bridge games of players from one
    behavioral cluster. Identify 5-7 characteristic skills that
    distinguish their decision-making style.
    Output JSON: {skills: [...], confidence: float}"
   ```
4. Aggregate skills across chunks (skills appearing in 3+ chunks)
5. Assign human-readable profile name per cluster

**Outputs:** ✅ COMPLETE (May 2026)
- `results/stage2_sample_v2_focused_prompt/skill_profiles_reagg_t040.json`
- Empirically validated vs Generalist baseline (Cohen's d 2.13–4.62, p < 0.05)

**Success criteria:**
- ✅ Each profile has 5-7 distinct skills
- ✅ Skills are interpretable in natural language
- ✅ Cross-validator (Anna) agrees with at least 4/5 profile names

**Profile names (confirmed from Stage 1):**
- 🎯 Slam Hunter — aggressive risk-taker, bids for jackpot contracts
- 🛡️ Insurance Player — loss-averse, stops at safe partial scores
- 💥 Fighter — punishes opponents with penalty doubles
- ♠️ NT Specialist — analytical, prefers balanced no-trump contracts
- 👥 Generalist — baseline, statistically average across all features

---

### Stage 3: Agent Construction

**Description:** Build 5 LLM agents, each with a unique profile-based system prompt.

**Inputs:**
- Skill profiles from Stage 2
- Bridge bidding rules (rule-based context)
- Negotiation scenario templates

**Processing:**
1. For each profile, construct system prompt:
   - Identity (profile name)
   - Skills (5-7 traits)
   - Domain context (bridge OR negotiation)
   - Output format specification (JSON schema)
2. Implement `BridgeAgent.make_bid(context)` method
3. Implement `NegotiationAgent.respond(offer, context)` method
4. Sanity test: each agent should behave differently in same context

**Outputs:** ✅ COMPLETE (May 2026)
- `src/stage3_agents/base_agent.py` (shared skeleton)
- `src/stage3_agents/bridge_agent.py`
- `src/stage3_agents/nego_agent.py`
- `src/shared/prompts.py` (centralised prompt/"character card" library)

**Success criteria:**
- ✅ All 5 agents produce valid JSON output 95%+ of the time
- ✅ Behavioral variance > 30% in identical scenarios (different agents → different choices)
- ✅ Decisions traceable to profile skills (explainable)

---

### Stage 4: Dual Simulation

**Status:** ✅ COMPLETE (June 2026)

**Result:** Full pipeline ran end-to-end. The final, strongest metric measures
**bridge skill directly from the REAL competitive data** (duplicate IMP vs the
field, defence included) and correlates it with the simulated negotiation
surplus: **Spearman ρ = +0.50** (IMP-scaled; +0.60 on raw points, same ranking).
The aggressive elite profiles (Slam Hunter, Fighter) top BOTH domains. This needs
no simulation and no hand-chosen weight, and counts the Fighter's defence
automatically. A random "monkey" baseline (a metric a monkey beats is broken) and
a double-dummy evaluator were added to validate the bridge metric. Earlier
simulated metrics (par-only +0.20, fight-aware +0.80, double-dummy single-bid
−0.90 artifact) are documented as the journey; the real-data +0.50 is the
defensible, expert-reviewed result. n=5 keeps power low (ρ is an indication).
Outputs: `notebooks/alignment_real_bridge.py`,
`results/stage4/alignment_real_bridge_report.md`,
`docs/images/{alignment_real_bridge,real_skill_spectrum}.png`. See
RESEARCH_INSIGHTS Q7.10–Q7.12.

**Description:** Run agents in both bridge auctions and business negotiations; measure alignment.

#### Sub-stage 4a: Bridge Simulation

**Inputs:**
- 4 agents from Stage 3
- 50 randomly generated bridge hand contexts

**Processing:**
1. For each game:
   - Deal 4 hands (using realistic distribution)
   - Each agent (NSEW positions) bids in turn
   - Continue until 3 consecutive Pass
   - Compute outcome (declarer success, IMP score)
2. Track per-profile win rate

**Outputs:**
- `results/bridge_simulations.jsonl` (full bid logs)
- `results/bridge_winrates.csv`

#### Sub-stage 4b: Negotiation Simulation

**Inputs:**
- Same 4 agents (different domain instantiation)
- 4 scenarios × ~12 pair matchups = ~48 simulations

**Scenarios (designed by Anna):**
1. **M&A:** Acquirer ($40M offer) vs. Target ($50M asking)
2. **Joint Venture:** 60-40 split request vs. 50-50 demand
3. **B2B SaaS:** Vendor pricing vs. enterprise procurement
4. **Government Tender:** Bid pricing vs. evaluation criteria

**Processing:**
1. For each scenario × agent pair:
   - Initialize context
   - Agents exchange up to 10 rounds of offers/responses
   - Determine outcome (deal closed? At what terms?)
2. Track per-profile success rate

**Outputs:**
- `results/negotiation_simulations.jsonl`
- `results/negotiation_winrates.csv`

#### Sub-stage 4c: Alignment Analysis

**Processing:**
1. Compute per-profile win rate in both domains
2. Compute Spearman correlation across profile rankings
3. Visualize: scatter plot of bridge win rate vs. negotiation win rate
4. Statistical test: correlation significance (p-value)

**Outputs:**
- `results/alignment_analysis.png`
- `results/alignment_report.md`
- **The final answer:** Did we hit ≥70% alignment?

**Success criteria:**
- ✅ ≥70% Spearman correlation → Hypothesis supported
- ⚠️ 50-69% → Partial support, discussion required
- ❌ <50% → Null hypothesis: profiles don't transfer (still valid finding)

---

## 7. Out of Scope (Explicit)

The following are **NOT** included in this MVP:

| Excluded Item | Reason | Future Plan |
|---------------|--------|-------------|
| Real negotiation data validation | No public dataset exists | PhD Year 2-3 |
| Fine-tuning LLMs | 5-week timeline too short | Out of scope |
| Multi-modal data (timing, voice) | Not available in current dataset | PhD Year 2 |
| Real-time API for business users | Research prototype, not product | Post-PhD |
| Cross-cultural validation | Single-region (European) data | Future work |
| Fine-tuning any LLM | Out of timeline scope | Post-MVP |
| Diffusion models | Irrelevant to research question | Never |
| LSTM/Transformer for bid sequences | Stretch goal only | Maybe Sprint 4 |

---

## 8. Technical Architecture

### Stack Selection Rationale

| Component | Choice | Rationale |
|-----------|--------|-----------|
| ML | scikit-learn | Standard, well-documented, course-taught |
| LLM (default) | Gemini Flash 2.0 | $0.30/M output, sufficient quality |
| LLM (validation) | Anthropic Claude + OpenAI | Cross-provider robustness check (key for PhD) |
| LLM (synthesis, optional) | Gemini 2.5 Pro / Claude Opus | Higher quality for final report |
| LLM routing | `src/shared/llm_client.py` | Single contract, swap providers via `provider=` |
| Bridge opponent (Stage 4) | `lorserker/ben` engine | Real, non-LLM opponent for objective IMP scoring |
| Visualization | matplotlib + seaborn | Course standard |
| NLP | gensim (Word2Vec) | Course standard |
| Testing | pytest | Industry standard |
| IDE | Google Antigravity | Agent-first, free preview, Gemini-native |

### Data Flow

```
data/raw/all_matches_full.csv (read-only)
  │
  ↓ src/stage1_clustering/features.py
data/processed/player_features.csv
  │
  ↓ src/stage1_clustering/extreme_profiles.py
data/processed/player_profiles.csv
  │
  ↓ src/stage2_skills/extractor.py (calls Gemini Flash)
data/processed/skill_profiles.json
  │
  ↓ src/stage3_agents/*.py (loads profiles)
[agents instantiated in memory]
  │
  ↓ src/stage4_simulate/*.py (calls Gemini Flash)
results/bridge_simulations.jsonl
results/negotiation_simulations.jsonl
  │
  ↓ src/stage4_simulate/alignment.py
results/alignment_report.md ← FINAL OUTPUT
```

---

## 9. Key Performance Indicators (KPIs)

### Research KPIs (Primary)
| Metric | Target | Measured Where |
|--------|--------|----------------|
| Profile separation (Stage 1) | Defining feature ratio ≥ 1.5× | `docs/images/feature_bars.png` |
| Cross-domain alignment (Stage 4c) | ≥ 70% | `results/alignment_report.md` |
| Statistical significance (all stages) | p < 0.05 | Various |

### Engineering KPIs (Secondary)
| Metric | Target | Measured Where |
|--------|--------|----------------|
| Test coverage (Stage 1) | ≥ 80% | `pytest --cov` |
| Test coverage (overall) | ≥ 60% | `pytest --cov` |
| End-to-end pipeline runtime | < 30 min | CI logs |
| Total Gemini cost | < $15 | `results/cost_log.csv` |

### Process KPIs (Reporting)
| Metric | Target | Measured Where |
|--------|--------|----------------|
| GitHub commits/week | ≥ 10 | GitHub stats |
| Documented decisions | 100% | `docs/decisions.md` |
| Reproducibility from clean clone | Must work | CI test |

---

## 10. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Gemini Flash insufficient quality for agents | Medium | High | Upgrade Gemini → 2.5 Pro, or switch the affected stage to Claude/OpenAI via `llm_client.py` |
| Gemini API rate limits hit | Medium | Medium | Free tier = 15 req/min; upgrade to paid if needed |
| Clusters not statistically significant | **Occurred** | High | **Resolved:** switched to extreme-percentile profiling. Continuum finding documented as a research result. |
| Agent prompts produce inconsistent behavior | Medium | High | Manual sanity check at Stage 3 |
| Negotiation simulations too unrealistic | High | Medium | Explicit caveat in PRD: "proof-of-concept" |
| Alignment is <50% (null result) | Medium | Medium | Frame as valid research finding, not failure |
| 5 weeks insufficient | Medium | High | Scope creep discipline; cut LSTM if needed |
| Antigravity IDE bugs / preview issues | Low | Medium | Fall back to VS Code with Python extension |
| Data quality issues | Low | Medium | Already validated via existing exploration |

---

## 11. Deliverables Checklist

### For Course Grading (Dr. Rami)
- [ ] Working code in GitHub repo
- [ ] README with run instructions
- [ ] PRD (this document)
- [ ] Prompt book (`docs/prompts.md`)
- [ ] Test suite with coverage report
- [ ] Final presentation (12-15 slides)
- [ ] Promotional video (60-90 seconds)
- [ ] Business plan section (TAM/SAM/SOM)
- [ ] Token economics analysis

### For PhD Progress (Prof. Jari)
- [ ] Literature review (target met: 17 NegoPlay-focused + 9 PhD background = 26 sources; 2 anchors identified)
- [ ] Statistical validation of profiles
- [ ] Alignment analysis with significance tests
- [ ] Clear documentation of limitations
- [ ] Roadmap connecting to Paper 1

### For Personal Portfolio
- [ ] Polished README with results
- [ ] Public-friendly summary
- [ ] Academic citation format
- [ ] Open-source license (MIT)

---

## 12. Timeline Snapshot

| Week | Sprint | Key Deliverable |
|------|--------|----------------|
| 1 | Research & Setup | Literature review, PRD, repo setup |
| 2 | Stage 1 + 2 | Clusters discovered, skills extracted |
| 3 | Stage 3 | 4 agents constructed and validated |
| 4 | Stage 4 | Simulations run, alignment analyzed |
| 5 | Polish & Defense | Presentation, video, defense rehearsal |

See [TASKS.md](./TASKS.md) for detailed task breakdown.

---

## 13. Approval & Sign-off

This PRD is the authoritative specification for NegoPlay v1.0.

**Author:** Anna Ben-Shushan
**Reviewers:**
- Dr. Rami (Course Supervisor) — Pending
- Prof. Jari Hämäläinen (PhD Supervisor) — Pending

**Version history:**
- v1.0 — Initial PRD (project kickoff, Session 1)
- v2.0 — Stage 1 results incorporated (May 2026): K-Means/HDBSCAN/GMM failed → Extreme Percentile Profiling adopted; continuum finding documented; 5 profiles confirmed
- v2.1 — Dr. Rami preprocessing audit (May 2026): full preprocessing pipeline added; continuum confirmed across 5 configurations (max silhouette 0.24); player count revised 807→563 (min_boards 20→50 + binomial test); profiles revised 5→4 extreme; Bridge Expert Validation Skill added; 81 tests passing
- v2.1 — Sample-size revision (May 2026, after Nezer review): raised min_boards 20→50, added binomial significance test (p < 0.05). Slam Hunters: 64 → 20 (each statistically robust). Tests: 52 → 56.

Changes to this document require re-validation against:
1. Course requirements (Dr. Segal's methodology)
2. PhD study plan (LUT University)
3. Budget and timeline constraints

---

## 14. References

- **Course materials:** Dr. Yoram Segal's project methodology framework
- **Project literature (NegoPlay):** [`docs/literature.md`](docs/literature.md) — 17 papers, anchors marked
- **Project literature (PhD broader):** [`../PHD_LITERATURE.md`](../PHD_LITERATURE.md) — 9 papers
- **Related GitHub repos:** [`RELATED_WORK_AND_PLAN.md`](RELATED_WORK_AND_PLAN.md) — 8 repos reviewed
- **PhD plan:** `../../docs/study_plan.md`
- **Research insights:** `../../docs/research_insights.md`
- **Dataset documentation:** `../../data/README.md`
- **IDE:** [Google Antigravity](https://antigravity.google/) — agent-first development
