# PRD: NegoPlay

> **Product Requirements Document**
> Version 1.0 · Anna Ben-Shushan · 2026

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
   ├─→ Stage 1: ML Profile Discovery (scikit-learn)
   │   Output: 3-5 statistical clusters of players
   │
   ├─→ Stage 2: LLM Skill Extraction (Gemini Flash by default)
   │   Output: Named profiles with 5-7 skills each
   │
   ├─→ Stage 3: Agent Construction (Gemini Flash by default; Claude/OpenAI for cross-model validation)
   │   Output: 4 LLM agents, one per profile
   │
   └─→ Stage 4: Dual Simulation (LLM agents vs. BEN bridge engine + LLM negotiation)
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

**H1 (Statistical):** K-Means clustering on 5 decision features will produce 3-5 stable profiles with Silhouette ≥ 0.4 and p < 0.05 versus random clustering.

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

### Stage 1: Profile Discovery

**Description:** Statistical clustering of 1,421 bridge players into 3-5 decision profiles.

**Inputs:**
- `data/raw/all_matches_full.csv` (149,208 rows)

**Processing:**
1. Filter to players with ≥20 declared boards → 1,421 players
2. Compute 5 features per player:
   - `slam_rate`: % of declarations at slam level (6+)
   - `success_rate`: % of contracts made
   - `preempt_rate`: % of opening bids at level 2+
   - `double_rate`: % of double calls (penalty + takeout)
   - `avg_risk_score`: Composite risk index
3. Standardize features (StandardScaler)
4. Run K-Means for k=2,3,4,5,6 → choose by Silhouette
5. Validate with HDBSCAN → must produce similar clusters
6. Statistical test: cluster means vs. random → p < 0.05

**Outputs:**
- `data/processed/player_features.csv`
- `data/processed/player_clusters.csv`
- `results/stage1_silhouette.png`
- `results/stage1_validation.md`

**Success criteria:**
- ✅ Silhouette score ≥ 0.4 for chosen k
- ✅ HDBSCAN finds same number of clusters ± 1
- ✅ At least 3 features show statistically significant cluster differences

---

### Stage 2: Skill Extraction

**Description:** Gemini Flash 2.0 analyzes game samples from each cluster to identify characteristic skills.

**Inputs:**
- 4 cluster groups from Stage 1
- 20-30 game samples per chunk
- 5-10 chunks per cluster

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

**Outputs:**
- `data/processed/skill_profiles.json`
- `docs/profile_descriptions.md`

**Success criteria:**
- ✅ Each profile has 5-7 distinct skills
- ✅ Skills are interpretable in natural language
- ✅ Cross-validator (Anna) agrees with at least 3/4 profile names

**Expected profile names (hypothesis, will refine):**
- Slam Hunter (aggressive risk-taker)
- Insurance Player (risk-averse)
- Bluffer (aggressive but inconsistent)
- Doubler (defensive-aggressive)

---

### Stage 3: Agent Construction

**Description:** Build 4 LLM agents, each with a unique profile-based system prompt.

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

**Outputs:**
- `src/stage3_agents/bridge_agent.py`
- `src/stage3_agents/nego_agent.py`
- `docs/prompts.md` (full prompt book)

**Success criteria:**
- ✅ All 4 agents produce valid JSON output 95%+ of the time
- ✅ Behavioral variance > 30% in identical scenarios (different agents → different choices)
- ✅ Decisions traceable to profile skills (explainable)

---

### Stage 4: Dual Simulation

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
  ↓ src/stage1_clustering/clustering.py
data/processed/player_clusters.csv
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
| Silhouette score (Stage 1) | ≥ 0.4 | `results/stage1_silhouette.png` |
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
| Clusters not statistically significant | Low | High | Pre-tested with 1,421 players, should work |
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
