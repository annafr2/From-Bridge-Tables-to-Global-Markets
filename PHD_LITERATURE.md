# PhD Literature — Wider Foundation

> Papers that ground the **PhD** (RQ1-RQ6) but are not directly used by NegoPlay's MVP code.
> NegoPlay-focused literature lives in [`NegoPlay/docs/literature.md`](NegoPlay/docs/literature.md).
>
> **Last updated:** 2026-05-19

These are the "background depth" papers — they prove the PhD is grounded in modern AI research (AlphaZero, MuZero, Pluribus, CFR family), but they're too heavy/general to implement in the 5-week NegoPlay course project. Reserve for the PhD methodology and theoretical-framing chapters.

---

## Quick index

| # | Title (short) | Authors, Year | Main method | Why it grounds the PhD |
|---|---------------|---------------|-------------|------------------------|
| 1 | **AlphaZero** — chess, shogi, Go from scratch | Silver, 2017 | Deep RL + MCTS, self-play | Foundational proof that self-play masters complex strategy without human heuristics |
| 2 | **MuZero** — planning with a learned model | Schrittwieser, 2019 | Model-based RL + MCTS over learned latent | Closer to business negotiation: no perfect simulator, no explicit rules |
| 3 | **Pluribus** — superhuman multiplayer poker | Brown & Sandholm, 2019 | MCCFR + depth-limited subgame solving | Blueprint for scaling beyond 2-player zero-sum into multi-agent imperfect info |
| 4 | **Automatic Bridge Bidding via DRL** | Yeh, 2018 | Deep Q + UCB1 + penetrative Bellman | 🔗 Same authors / approach as the `chihkuanyeh` GitHub repo (ECAI 2016 lineage) |
| 5 | **Regret Minimization in Incomplete-Info Games** | Zinkevich, 2007 | CFR + regret matching (Blackwell) | The mathematical bedrock for handling imperfect information |
| 6 | **CFR for Multiplayer Poker** | Abou Risk & Szafron, 2010 | CFR + imperfect-recall abstraction + strategy grafting | CFR works empirically beyond 2-player zero-sum (no convergence guarantees) |
| 7 | **Opponent Modeling in RTS Games** | Schadd, 2007 | Hierarchical fuzzy macro + discounted reward micro | Macro-style → micro-tactics hierarchy useful for dynamic coopetition framing |
| 8 | **Potential-aware Abstraction (GS3 poker)** | Gilpin, 2007 | Non-myopic K-means on histograms + excessive gap | State abstraction via future potential rather than current value |
| 9 | **Imperfect Information / PBE** | Dougherty, n.d. | Bayesian updating, sequential equilibrium | Game-theoretic vocabulary chapter for the PhD theory section |

---

## How these map to the PhD Research Questions

- **RQ1 (decision-making styles under imperfect info):** Silver 2017, Schrittwieser 2019, Yeh 2018, Schadd 2007
- **RQ2 (partnership):** Schadd 2007 (macro-coordination)
- **RQ3 (bridge as negotiation protocol):** Zinkevich 2007, Dougherty PBE, Schrittwieser 2019 (no-rule planning)
- **RQ4 (coopetition):** Brown 2019, Abou Risk 2010
- **RQ5 (XAI):** — covered by NegoPlay-side papers (Zhang 2022, Liu 2025)
- **RQ6 (business transfer):** Silver 2017 (general game-playing claim parallels), Schrittwieser 2019 (learned models in unknown environments)

---

## Paper-by-paper notes

### Paper 1 — AlphaZero (Silver, 2017)

- **Method:** Deep RL + general-purpose MCTS, fully tabula rasa.
- **Dataset:** 44M (chess) / 24M (shogi) / 21M (Go) self-play games.
- **Headline:** A single general-purpose RL algorithm achieves superhuman play across 3 games with zero domain knowledge.
- **PhD relevance:** Proof-of-concept that self-play masters strategy without human heuristics — justifies training bridge models without rigid pre-existing bidding systems.
- **Limitation:** Perfect information / full observability — not directly applicable to bridge or negotiation.

### Paper 2 — MuZero (Schrittwieser, 2019)

- **Method:** Model-based RL — learns environment dynamics, plans over a learned latent state.
- **Dataset:** 57 Atari games (200M frames each) + board game self-play.
- **Headline:** Planning + superhuman performance without being given the rules or true state.
- **PhD relevance:** Business negotiation lacks a perfect simulator or explicit mathematical rules — MuZero's approach (plan via learned internal dynamics) is the closest analogue.
- **Limitation:** Deterministic dynamics function; stochastic extension left for future work.

### Paper 3 — Pluribus (Brown & Sandholm, 2019)

- **Method:** MCCFR for offline blueprint + real-time depth-limited subgame solving.
- **Dataset:** Offline self-play + 10K live hands vs. 13 elite human poker pros.
- **Headline:** Beats elite humans at 6-player no-limit hold'em **without** exploiting opponents.
- **PhD relevance:** Blueprint for scaling 2-player methods to multi-agent imperfect-information environments — exactly the leap RQ4 (coopetition) requires.
- **Limitation:** Fixed strategy; doesn't adapt to observed opponent weaknesses.

### Paper 4 — Automatic Bridge Bidding via DRL (Yeh, 2018)

- **Method:** Deep Q-learning + UCB1 exploration + a novel "penetrative Bellman equation" handling state separation between partners.
- **Dataset:** 140K random deals evaluated via Double Dummy Analysis.
- **Headline:** Constructs bidding logic from raw data, outperforming human-rule-based champion systems.
- **PhD relevance:** 🔗 **Same authors and approach as the `chihkuanyeh/Automatic-Bridge-Bidding-by-DRL` GitHub repo** (see `NegoPlay/RELATED_WORK_AND_PLAN.md`). Validates deep learning on raw bridge state to discover implicit communication protocols.
- **Limitation:** Only handles uncontested auctions (assumes opponents always pass).

### Paper 5 — CFR (Zinkevich, 2007)

- **Method:** Counterfactual regret minimization via regret matching (Blackwell's approachability theorem).
- **Dataset:** Abstractions of 2-player limit hold'em (up to 10¹² states).
- **Headline:** Local regret minimization at information sets provably converges to Nash equilibrium via self-play.
- **PhD relevance:** Mathematical foundation for imperfect-info handling — needed for the formal/theoretical chapter on bridge bidding as a negotiation protocol (RQ3).
- **Limitation:** Requires explicit card abstraction to be tractable.

### Paper 6 — CFR for Multiplayer Poker (Abou Risk & Szafron, 2010)

- **Method:** CFR + imperfect-recall card abstraction + strategy grafting using Heads-Up Experts.
- **Dataset:** Millions of cross-play hands vs. benchmark bots.
- **Headline:** CFR produces tournament-winning multiplayer strategies despite no convergence guarantees in that setting.
- **PhD relevance:** Empirical license to model multi-agent business ecosystems without strict equilibrium guarantees.
- **Limitation:** Evaluation requires millions of cross-play hands; no exact best-response.

### Paper 7 — Opponent Modeling in RTS Games (Schadd, 2007)

- **Method:** Hierarchical opponent classification — top-level fuzzy macro-style, bottom-level discounted-reward tactics.
- **Dataset:** Simulated matches in the Spring RTS engine.
- **Headline:** Hierarchical classification accurately identifies macro-style early + predicts tactical sub-strategies later.
- **PhD relevance:** Macro→micro hierarchy is a clean framework for bidding-style clustering — bidding "dialect" at the macro level, specific bid choice at the micro level. Useful for RQ1 + RQ4 (shifting alliances).
- **Limitation:** Fog-of-war means early micro-classification is weak.

### Paper 8 — Potential-Aware Abstraction GS3 (Gilpin, 2007)

- **Method:** Non-myopic K-means on histograms of future state distributions + integer programming + excessive gap equilibrium approximation.
- **Dataset:** Texas Hold'em with custom colexicographical indexing for compression.
- **Headline:** Abstracting on **future information potential** (not current strength) produced GS3, beating all prior top poker bots.
- **PhD relevance:** Idea of clustering by *what a state could become* rather than *what it is now* — relevant for long-horizon bidding-style clustering and for evaluating negotiation states by their option value.
- **Limitation:** Multi-dimensional histograms restricted to the immediately following level.

### Paper 9 — PBE / Imperfect Information (Dougherty, n.d.)

- **Method:** Game-theoretic mathematical analysis — Bayesian updating, sequential / Perfect Bayesian Equilibrium, separating vs. pooling equilibria.
- **Dataset:** Pedagogical examples (Battle of the Sexes, Gift Game, Cuban Missile Crisis).
- **Headline:** Define optimal play in imperfect-info games via beliefs derived from strategy + Bayes' rule.
- **PhD relevance:** Vocabulary — when the thesis needs to formally describe a bridge auction as a Bayesian signaling game, this is the lecture deck to cite.
- **Limitation:** Pedagogical only, no scalable algorithms.

---

## How to use this file

- **For the PhD literature-review chapter:** these 9 papers + the 17 in `NegoPlay/docs/literature.md` = 26 sources, well within Dr. Segal's "15–20 relevant sources" funnel (and overshooting toward the deeper "5-8 core" set).
- **For NegoPlay's preliminary report:** mention these as "broader theoretical context" but don't anchor on them — the NegoPlay anchors are Talwadker 2022 + Rong 2019.
- **For the defense:** be ready to say one sentence per paper if asked "why didn't you use CFR / AlphaZero / etc.?" — answer: "we're 5 weeks in; these are the year-2-3 methods. NegoPlay validates the upstream profiling step first."
