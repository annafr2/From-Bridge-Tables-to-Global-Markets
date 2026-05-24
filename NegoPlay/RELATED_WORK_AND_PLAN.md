# NegoPlay — Related Work Review & Work Plan

> Mapping 8 public GitHub repositories against NegoPlay's scope
> (4-stage clustering → LLM agents pipeline; Gemini default, Claude/OpenAI
> available) and Anna's PhD architecture (ENN + PNN + PPO).
>
> **Last updated:** 2026-05-19
> **Author:** Anna Ben-Shushan (compiled with Claude)

---

## 1. TL;DR — what to use, what to skip

| # | Repo | Relevance to **NegoPlay MVP** | Relevance to **PhD (Year 1–2)** | Verdict |
|---|------|-------------------------------|--------------------------------|---------|
| 1 | `lorserker/ben` | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | **USE** — opponent/baseline in Stage 4 |
| 2 | `harukaki/brl` | ⭐ | ⭐⭐⭐⭐⭐ | **STUDY** — direct blueprint for PNN/PPO |
| 3 | `yichijin/bridgescape` | ⭐⭐ | ⭐⭐⭐⭐ | **USE LATER** — BBO `.lin` parser (PhD Year 2) |
| 4 | `eduucaldas/AlphaBridge` | ⭐⭐ | ⭐⭐⭐ | **READ** — opening-lead CNN/RL methodology |
| 5 | `chihkuanyeh/Automatic-Bridge-Bidding-by-DRL` | ⭐ (cite only) | ⭐⭐⭐ (cite only) | **CITE** — MATLAB, ECAI 2016, classic reference |
| 6 | `BSalita/ML-Contract-Bridge` | ⭐⭐ | ⭐⭐ | **READ** — feature-engineering inspiration |
| 7 | `google-deepmind/open_spiel` PR #766 | ⭐⭐ | ⭐⭐⭐⭐ | **USE LATER** — OpenSpiel bridge env + WBridge5 |
| 8 | `progclub-iitm/Bridge_Bot` | — | — | **SKIP** — empty/abandoned (3 commits) |

---

## 2. Per-repo breakdown

### ⭐ #1 — `lorserker/ben` — Bridge Engine "BEN"
- **What it is:** Full open-source bridge engine (Python). NN-based bidding + opening lead + card play, augmented with a double-dummy solver. Web UI + scriptable. Supports Blue Chip Bridge protocol.
- **License:** GPL-3.0 → safe to import as an external tool, but cannot vendor its code into NegoPlay without GPL-tainting.
- **How it serves NegoPlay (Stage 4):**
  - Use BEN as a **real opponent / referee** in your bridge simulation. Each Gemini-driven profile agent (Slam Hunter, etc.) plays *against* BEN's NN. This gives you an *objective* "win" signal that is not just LLM-vs-LLM, which strengthens the alignment claim.
  - Use BEN's bidding probabilities as a **baseline distribution** — when a profile agent bids differently from BEN, that's signal of "dialect."
- **How it serves the PhD:** Strong reference for what a publishable bridge NN looks like; you can cite BEN as state-of-the-art open engine in RQ1/RQ4 chapters.
- **Action:** Install in a separate venv. Wrap it behind a thin `src/shared/ben_client.py` adapter so the GPL boundary stays clean.

### ⭐ #2 — `harukaki/brl` — Bridge RL (PPO over pgx)
- **What it is:** Recent (2024) repo. PPO-trained bidding agent that **beats WBridge5**. Uses the `pgx` JAX environment and OpenSpiel datasets for supervised pre-training.
- **Relevance to NegoPlay:** Low — uses JAX/PyTorch and DRL, which is outside the NegoPlay MVP stack (LLM-agents, no DRL).
- **Relevance to the PhD:** **Very high.** This is essentially the architecture Anna's `CLAUDE.md` describes as the planned ENN + PNN + PPO stack. Study before writing the PhD methodology chapter.
- **Action:** Clone and read; do **not** integrate into NegoPlay code. Add to literature.

### #3 — `yichijin/bridgescape`
- **What it is:** Python parser/analyzer for BBO `.lin` files; ships ~650K ACBL Speedball boards.
- **Relevance to NegoPlay:** Not needed — NegoPlay uses the 149K EuroBridge CSV already in the PhD repo.
- **Relevance to the PhD:** **Directly solves the "BBO Vugraph parser" task** flagged in `../CLAUDE.md` ("LIN parsers — to be built"). Saves weeks of work in Year 2.
- **Action:** Fork/bookmark. Schedule for the data-volume expansion task (target 200K–500K rows).

### #4 — `eduucaldas/AlphaBridge`
- **What it is:** Opening-lead prediction. CNN + supervised learning on expert lin data, plus a DRL variant against a Double-Dummy Solver. Includes BBO Vugraph downloader.
- **Relevance to NegoPlay:** Partial — opening lead is *card play*, NegoPlay focuses on *bidding*. But the BBO parsing pipeline is reusable.
- **Relevance to the PhD:** Methodology template for CNN-on-bridge-state. Not central; bidding ≫ leads for negotiation framing.
- **Action:** Read once. Borrow only the data-loader if `bridgescape` doesn't cover ACBL/Vugraph cases.

### #5 — `chihkuanyeh/Automatic-Bridge-Bidding-by-Deep-Reinforcement-Learning`
- **What it is:** Code from ECAI 2016 paper. Deep Q-learning for bidding. Four trained nets for max-bid levels 2–5. **MATLAB.**
- **🔗 Companion paper:** Yeh et al., 2018, *Automatic Bridge Bidding Using Deep Reinforcement Learning* — same authors, extended into 140K deals + penetrative Bellman equation. Filed in [`../PHD_LITERATURE.md`](../PHD_LITERATURE.md) as paper #4.
- **Relevance:** Stack-forbidden (no MATLAB; PhD/NegoPlay are Python). Architecture is also pre-Transformer.
- **Action:** **Cite in literature review.** Do not import.

### #6 — `BSalita/ML-Contract-Bridge`
- **What it is:** sklearn / fastai pipeline that predicts contract result, trick count, par score from hands *without* the auction. ACBL + Common Game data.
- **Relevance to NegoPlay:** Feature-engineering inspiration only. Not predictive of "style."
- **Relevance to the PhD:** Useful for the "double-dummy gap" risk metric — i.e., did the player out-bid or under-bid the DDS-optimal contract? This maps neatly onto `risk_score` in `../CLAUDE.md`.
- **Action:** Skim notebooks; lift feature ideas into `src/stage1_clustering/features.py`.

### #7 — `google-deepmind/open_spiel` PR #766
- **What it is:** Adds support for **WBridge5** (classic rule-based bridge engine that has won the world computer bridge championship) to plug into OpenSpiel's bridge environment.
- **Relevance to NegoPlay:** Not for the MVP — but if you ever want a *non-LLM, non-NN* opponent to benchmark profile agents against, WBridge5 is the canonical choice. OpenSpiel's bridge env is also the same one `harukaki/brl` uses.
- **Relevance to the PhD:** Essential baseline. Every published bridge AI benchmarks against WBridge5.
- **Action:** Defer to Stage 4 v2 or PhD Year 2. Document as a planned baseline.

### #8 — `progclub-iitm/Bridge_Bot`
- **What it is:** Student club placeholder. 3 commits, 1 star. No working code.
- **Action:** Ignore.

---

## 3. Where each repo plugs into NegoPlay's 4-stage pipeline

```
STAGE 1: PROFILE DISCOVERY (clustering on 149K boards)
   ├─ Feature inspiration:        BSalita/ML-Contract-Bridge   (#6)
   └─ Future data expansion:      bridgescape, AlphaBridge      (#3, #4)

STAGE 2: SKILL EXTRACTION (Gemini Flash)
   └─ No external repo needed — pure LLM work.

STAGE 3: AGENT CONSTRUCTION (LLM-prompt agents)
   └─ No external repo needed — Gemini default; Claude/OpenAI for cross-model validation.

STAGE 4: DUAL SIMULATION + ALIGNMENT
   ├─ Bridge opponent / referee:  lorserker/ben                (#1)  ← KEY
   ├─ Optional environment:       OpenSpiel bridge env         (#7)
   └─ Classic benchmark opponent: WBridge5 via PR #766         (#7)

PhD METHODOLOGY (out of NegoPlay scope, but flagged)
   ├─ PNN/PPO blueprint:          harukaki/brl                 (#2)
   ├─ Historical DRL reference:   chihkuanyeh (ECAI 2016)      (#5)
   └─ Engine-level reference:     ben                          (#1)
```

---

## 4. Concrete work plan (next 4 weeks)

NegoPlay is a 5-week course final project running in parallel with the PhD's data-collection month. The plan keeps NegoPlay self-contained (Gemini-only) while letting Year-1 PhD work absorb the heavier repos.

### Week 1 — Foundations & related-work freeze
- [ ] Read `lorserker/ben` README + 1 tutorial notebook end-to-end. (1.5h)
- [ ] Read `harukaki/brl` README + identify the supervised pre-training script. (1h)
- [ ] Skim `BSalita/ML-Contract-Bridge` notebooks; capture 3–5 feature ideas into `src/stage1_clustering/features.py` as TODO docstrings. (2h)
- [ ] Add this file + the chosen repos to `../RESEARCH_INSIGHTS.md` literature section.
- [ ] **Decision gate:** confirm Stage 4 will use BEN as the bridge opponent (GPL boundary OK because BEN runs as a separate process behind `ben_client.py`). If Anna prefers a fully self-contained simulation, fall back to LLM-vs-LLM with a DDS sanity check.

### Week 2 — Stage 1 clustering
- [ ] Implement `features.py`: HCP, suit-length stats, bidding-aggressiveness ratios, slam-attempt rate, double rate, preempt rate. (Reuse risk metrics from `../CLAUDE.md` table.)
- [ ] Fit K-Means (k ∈ {3,4,5}) + HDBSCAN; pick best by silhouette ≥ 0.4.
- [ ] Output: `results/profiles.json` with 3–5 named profile clusters and member player IDs.
- [ ] Tests: 80%+ coverage on `features.py` and `clustering.py`.

### Week 3 — Stage 2 + Stage 3 (Gemini)
- [ ] Build `chunker.py` to slice each profile into 20–30-game windows.
- [ ] Build `extractor.py` calling Gemini Flash 2.0 with structured JSON output for 5–7 skills per profile.
- [ ] Build `base_agent.py` and one concrete `bridge_agent.py` per profile, with system prompts injected from Stage 2 output.
- [ ] Cost guardrail: stop if `results/llm_logs/` shows cumulative spend > $20.

### Week 4 — Stage 4 dual simulation + report
- [ ] **Bridge sim:** 50–100 auctions, each profile pair vs. BEN at the other table. Score = IMPs vs. BEN.
- [ ] **Negotiation sim:** 50–100 paired M&A / supplier negotiations with the same 4 profile agents. Score = utility achieved.
- [ ] **Alignment:** Spearman correlation between `bridge_winner_rate` and `nego_winner_rate` per profile. Hypothesis: ρ ≥ 0.7.
- [ ] Final write-up: `docs/results_analysis.md` + 8-slide deck.

### Deferred (NegoPlay v2 or PhD Year 1 Q3)
- [ ] Plug in `bridgescape` to expand dataset from 149K → 500K+ rows.
- [ ] Replace BEN-only opponent with WBridge5 via OpenSpiel PR #766 for an *independent* benchmark.
- [ ] Read `harukaki/brl` in depth and draft the PhD's PNN/PPO chapter.

---

## 5. Risks & open questions

1. **GPL contagion (BEN, #1):** Keep BEN behind a network/process boundary. Do not paste BEN source into NegoPlay. Confirm with Dr. Segal before submission.
2. **LLM agents vs. real bridge play:** Any LLM agent (Gemini, Claude, or OpenAI) will not play bridge at a strong level. NegoPlay's claim is about **behavioral alignment**, not bridge strength — frame the report carefully. The provider mix is also a *feature*: running the same profile on Gemini + Claude + OpenAI and showing the alignment finding holds across providers makes the result much harder to dismiss as "an artifact of one model."
3. **149K → enough?** PhD-side `CLAUDE.md` flags 200K–500K as "solid." For NegoPlay's clustering (per-player aggregates over ≥20 boards each), 149K is enough; do not block on data expansion.
4. **WBridge5 binary availability:** WBridge5 is Windows-only freeware. The OpenSpiel PR wires it up under Linux via Wine. Cost-of-effort = 1 day; defer.

---

## 6. One-line summary for supervisors

> NegoPlay will run end-to-end on clustering + LLM agents (Gemini by default, Claude/OpenAI for cross-model validation). `lorserker/ben` is the one external dependency worth integrating *now* (Stage 4 opponent); `harukaki/brl` and `bridgescape` are reserved for PhD Year-1 Q3 once the course project is delivered.
