# Literature — NegoPlay

> 17 papers grounding NegoPlay (Stage 1 clustering → LLM agents → negotiation simulation).
> Wider PhD-only literature lives in [`../../PHD_LITERATURE.md`](../../PHD_LITERATURE.md).
>
> **Last updated:** 2026-05-19

---

## 🎯 Anchor papers (the two NegoPlay is benchmarked against)

> Per Dr. Segal's framework, every project needs **2 anchor papers** — the works your project is built on top of. If a reader understands these two, they understand 80% of NegoPlay's logic.

### ⚓ Anchor 1 — **Talwadker et al., 2022 — CognitionNet**
*"A Collaborative Neural Network for Play Style Discovery in Online Skill Gaming Platform"*

**Why it's the anchor for Stage 1 (clustering):** This is the closest paper to NegoPlay's Stage 1. It mines sequential micro-patterns from gameplay (Rummy telemetry) and clusters players into behavioral profiles using a Seq2Seq LSTM interpreter + CNN classifier with a novel "bridge loss." Substitute "Rummy telemetry" with "bridge bidding sequences" and you have NegoPlay.

**Stage(s) served:** Stage 1 (profile discovery from bidding sequences).
**Key takeaway for NegoPlay:** validates that automated style discovery from raw gameplay sequences is publishable, and gives us a methodological blueprint (sequence interpreter + classifier).

### ⚓ Anchor 2 — **Rong et al., 2019 — Competitive Bridge Bidding with Deep Neural Networks**

**Why it's the anchor for Stage 3 + bridge-side credibility:** Splits bidding into two networks — ENN (estimates partner's hidden cards) + PNN (chooses the bid). Trains supervised on 1M+ Vugraph deals, fine-tunes with REINFORCE, **beats Wbridge5**. This is the architecture Anna's PhD plans to build (and the same logic NegoPlay mimics with LLMs: cluster-based "profile" = inference; agent prompt = policy).

**Stage(s) served:** Stage 3 (agent decision logic), Stage 4 (benchmark target).
**Key takeaway for NegoPlay:** when reviewers ask "why split inference from decision?" — point here.

### 💡 Theoretical foundation — **Lockett et al., 2007 — Evolving Explicit Opponent Models**

Not a methodological anchor (NEAT is dated), but the **theoretical justification** for NegoPlay's whole premise: model an unknown opponent as a **mixture of cardinal styles**. This is literally what NegoPlay does — every negotiation counterpart is treated as a weighted mixture of the discovered profiles (Slam Hunter / Insurance Player / etc.).

---

## Quick index — sorted by relevance to NegoPlay

⭐⭐⭐⭐⭐ = anchor / directly informs NegoPlay code · ⭐⭐⭐ = useful background · ⭐ = peripheral

| # | Short title | Authors, Year | Main method | NegoPlay relevance | PhD RQs |
|---|-------------|---------------|-------------|--------------------|---------|
| 1 ⚓ | **CognitionNet — Play Style Discovery** | Talwadker, 2022 | Seq2Seq LSTM + CNN + "bridge loss" | ⭐⭐⭐⭐⭐ **ANCHOR — Stage 1 blueprint** | RQ1, RQ2 |
| 2 ⚓ | **Competitive Bridge Bidding with DNNs** | Rong, 2019 | ENN + PNN + REINFORCE | ⭐⭐⭐⭐⭐ **ANCHOR — Stage 3 + benchmark** | RQ1, RQ3, RQ4 |
| 3 💡 | **Evolving Explicit Opponent Models** | Lockett, 2007 | NEAT + mixture identification | ⭐⭐⭐⭐⭐ **theoretical foundation** | RQ1, RQ4 |
| 4 | **Attitude Vectors — repeated games + cheap talk** | James, 2021 | K-means on action+message features | ⭐⭐⭐⭐⭐ bridges games → linguistic signaling = bridge → negotiation | RQ3, RQ6 |
| 5 | **Learning to Bid in Bridge (PIDM)** | Amit & Markovitch, 2006 | Monte Carlo + ID3 decision nets + co-training | ⭐⭐⭐⭐ partner modeling under uncertainty (coopetition) | RQ2, RQ4 |
| 6 | **VERA — Personalized AI Coach** | Buckley, 2024 | Hierarchical clustering + Levenshtein | ⭐⭐⭐⭐ blueprint for Strategic AI Coach | RQ5 |
| 7 | **BridgeHand2Vec** | Sztyber-Betley, 2023 | MLP supervised + Cross-Entropy RL | ⭐⭐⭐⭐⭐ embeddings for Stage 1 clustering | RQ1, RQ2 |
| 8 | **AI Bridge Bidding + Interactive Visualization** | Zhang, 2022 | Stacked RNN + MLP | ⭐⭐⭐⭐⭐ XAI heatmaps, RNN on bidding | RQ1, RQ5 |
| 9 | **Inference-Decision RL with GAN for Bidding** | Wang, 2024 | Conditional GAN + Policy Gradient | ⭐⭐⭐⭐ opponent modeling for negotiation | RQ1, RQ4 |
| 10 | **Cognitive Mechanisms in Bridge Experts** | Liu, 2025 | Behavioral experiments + ANOVA | ⭐⭐⭐⭐ chunking → LLM prompting | RQ1, RQ5 |
| 11 | **RL in Contract Bridge (review)** | Jarosz, 2022 | Literature review | ⭐⭐⭐⭐ foundational survey | all |
| 12 | **Simple Reproducible Bridge Baseline** | Kita, 2024 | Supervised + PPO + FSP | ⭐⭐⭐ blueprint (matches `harukaki/brl`) | RQ1, RQ4 |
| 13 | **Joint Policy Search for Multi-agent** | Tian, 2020 | A2C + Joint Policy Search | ⭐⭐⭐ multi-agent coordination | RQ2, RQ4 |
| 14 | **Policy Inference in Trick-taking Card Games** | Skat paper, 2019 | NN + Monte Carlo inference | ⭐⭐⭐ inferring hidden states from history | RQ3, RQ4 |
| 15 | **Effectiveness of HCP in Bridge** | Igra, 2024 | Logistic regression | ⭐⭐ feature validation for Stage 1 | RQ1 |
| 16 | **ERL with Action Sequence Search** | Wu, 2024 | TD3/PPO + Particle Swarm | ⭐⭐ exploration tactics | RQ4 |
| 17 | **Pgx — Hardware-Accelerated Simulators** | Koyamada, 2023 | JAX environments + AlphaZero/PPO | ⭐ infrastructure (PhD Year 2) | — |

---

## Theme map

- **Player-style clustering (Stage 1 — NegoPlay core):** Talwadker 2022 ⚓, Sztyber-Betley 2023, Igra 2024
- **Inference + Decision split (Stage 3):** Rong 2019 ⚓, Wang 2024, Zhang 2022
- **Opponent modeling as mixture of styles (NegoPlay theoretical premise):** Lockett 2007 💡, Amit & Markovitch 2006
- **Bridge → linguistic negotiation transfer:** James 2021, Liu 2025
- **AI coaching from sequences (RQ5):** Buckley 2024, Zhang 2022
- **DRL / self-play for bridge (PhD methodology):** Kita 2024, Wang 2024, Wu 2024, Tian 2020
- **Hidden-state inference (RQ3 negotiation analogue):** Wang 2024, Skat 2019, Zhang 2022
- **Survey / foundational:** Jarosz 2022
- **Human cognition (XAI grounding, RQ5):** Liu 2025
- **Infrastructure / engineering:** Pgx 2023

---

## Baseline definitions

> Per Dr. Segal: every result must be compared against a baseline. NegoPlay declares:

### Bridge-side baseline
- **Random bidder** (sanity floor)
- **Generic LLM agent with no profile prompt** (the main comparison — does profile-conditioning matter?)
- *Optional benchmark for PhD Year 2:* `lorserker/ben` engine or Wbridge5 (objective non-LLM opponent)

### Negotiation-side baseline
- **Generic negotiation agent with no profile prompt** in the same scenario
- *Metric:* utility / deal-closure rate per scenario

### Alignment metric (the headline number)
- **Spearman correlation** between `bridge_winner_rate` and `negotiation_winner_rate` per profile
- **Hypothesis:** ρ ≥ 0.7 across providers (Gemini + Claude + OpenAI cross-validation)

---

## Cross-reference to GitHub repos (see [`../RELATED_WORK_AND_PLAN.md`](../RELATED_WORK_AND_PLAN.md))

- **Yeh 2018 paper** ≡ **`chihkuanyeh/Automatic-Bridge-Bidding-by-DRL`** repo (the MATLAB ECAI 2016 codebase). Same authors, same approach (deep Q + penetrative Bellman). Filed under PhD literature, not NegoPlay.
- **Kita 2024 paper** ≈ **`harukaki/brl`** repo (PyTorch / pgx re-implementation of the same PPO+FSP recipe).
- **Pgx 2023 paper** = the JAX environment `harukaki/brl` runs on.

---

## Paper-by-paper notes

### Paper 1 ⚓ — CognitionNet (Talwadker, 2022) — **ANCHOR**

- **What is it about?** A dual-stage neural network that mines sequential micro-patterns from gameplay to discover behaviors and classify players' overall styles.
- **Main method:** Collaborative deep NN — Seq2Seq LSTM interpreter + CNN classifier, with a novel "bridge loss" formulation.
- **Dataset:** Real-world Rummy telemetry from 2021 with balanced engagement classes.
- **Most important finding:** Automates discovery of player psychology + game tactics directly from raw telemetry, outperforming standard baselines.
- **Why it matters for NegoPlay:** Provides a concrete algorithmic framework for bidding-style clustering from sequence data **without predefined rules** — exactly what Stage 1 needs. Plus the term "bridge loss" makes for excellent paper writing.
- **Limitation:** K-means cluster identities shift during training and only stabilize once the whole network is trained.

---

### Paper 2 ⚓ — Competitive Bridge Bidding with DNNs (Rong, 2019) — **ANCHOR**

- **What is it about?** Two integrated networks: ENN explicitly estimates partner's hidden cards; PNN outputs the optimal bid.
- **Main method:** Supervised pre-training of ENN+PNN, then REINFORCE fine-tuning.
- **Dataset:** 1M+ expert games from Vugraph, filtered into 12M training instances.
- **Most important finding:** Explicit probabilistic estimates of partner's cards (via ENN) significantly improve bidding strength — **beats Wbridge5**.
- **Why it matters for NegoPlay:** The ENN/PNN split is exactly NegoPlay's logic at the LLM level: cluster-profile (= inference of "what kind of agent am I?") + prompt-conditioned action (= policy). "Theory of Mind before action" is the core argument.
- **Limitation:** Approximates the play phase via Double Dummy Analysis (assumes perfect information during play).

---

### Paper 3 💡 — Evolving Explicit Opponent Models (Lockett, 2007) — **theoretical foundation**

- **What is it about?** Model unseen opponents as a **linear mixture** of predefined "cardinal" opponent strategies.
- **Main method:** NEAT (NeuroEvolution of Augmenting Topologies) for Mixture Identification + Decision modules.
- **Dataset:** Synthetic gameplay in the imperfect-information game "Guess It."
- **Most important finding:** Explicit mixture coefficients identifying the opponent significantly outperform baseline networks without identification.
- **Why it matters for NegoPlay:** **This IS NegoPlay's premise** — every counterpart is a weighted mixture of discovered profiles (Slam Hunter / Insurance Player / etc.). When reviewers ask "why model styles as discrete clusters rather than a continuous distribution?", this is the answer.
- **Limitation:** Assumes the cardinal set fully spans the strategic space.

---

### Paper 4 — Attitude Vectors in Repeated Games (James, 2021)

- **What is it about?** Predictive framework that encodes both actions **and** "cheap talk" messages into unified features, then clusters them into strategic attitudes (Greedy / Placate / Cooperative / Absurd).
- **Main method:** Attitude Vector Automata (AVA) — K-means clusters → decision trees.
- **Dataset:** Human gameplay transcripts with cheap talk from Alternator, Chicken, and Prisoner's Dilemma.
- **Most important finding:** Modeling behavior via attitude vectors generalizes well to novel payoff matrices.
- **Why it matters for NegoPlay:** **Directly bridges game matrices and linguistic signaling** — i.e., bridge bidding (structured signals) → business negotiation (cheap talk). This is the conceptual glue between Stage 4-bridge and Stage 4-negotiation. Best paper for the RQ3 / RQ6 framing.
- **Limitation:** Struggles when humans act inconsistently or switch strategies rapidly.

---

### Paper 5 — Learning to Bid in Bridge / PIDM (Amit & Markovitch, 2006)

- **What is it about?** Decision framework for bridge that handles cooperative partners + imperfect info by **explicitly modeling other agents' selection strategies**.
- **Main method:** Partial Information Decision Making (PIDM) — Monte Carlo sampling + ID3 decision nets, co-training partner agents.
- **Dataset:** 2000 random deals + 100 bidding challenge problems.
- **Most important finding:** Co-training two partner agents to refine their decision nets beats state-of-the-art programs.
- **Why it matters for NegoPlay:** Earliest formal treatment of **bridge as a coopetition problem** — must cooperate with partner under uncertainty while competing with opponents. Foundational citation for RQ4.
- **Limitation:** Relies on Double Dummy Analysis or Losing Trick Count evaluation — expensive + optimistic.

---

### Paper 6 — VERA: Personalized AI Coach (Buckley, 2024)

- **What is it about?** AI exploration coach that classifies learners from activity sequences and provides tailored feedback.
- **Main method:** Hierarchical agglomerative clustering of activity sequences with Levenshtein distance + procedural scaffolding.
- **Dataset:** User log activity in the VERA virtual modeling tool.
- **Most important finding:** Identifying deficiencies enables targeted metacognitive feedback through a full inquiry cycle.
- **Why it matters for NegoPlay:** **Direct blueprint for the eventual "Strategic AI Coach"** — sequence clustering → personalized scaffolding. Maps cleanly onto: bridge bidding sequence clustering → personalized negotiation advice.
- **Limitation:** Post-hoc sequence mining only, not real-time strategy adjustment.

---

### Paper 7 — BridgeHand2Vec (Sztyber-Betley, 2023)

- **What is it about?** Maps a 13-card bridge hand into an 8-D continuous vector by training an NN to estimate partnership trick count.
- **Main method:** MLP supervised + Cross-Entropy Method (RL).
- **Dataset:** 400K hands solved via Bridge Calculator (800K examples after flipping).
- **Most important finding:** The continuous vector groups hands by functional strength and accelerates agent learning.
- **Why it matters for NegoPlay:** Stage 1 v2 — replace hand-crafted features (HCP, etc.) with Hand2Vec embeddings as clustering input.
- **Limitation:** Trained on simplified setup (only N-S cards known); cannot fully replace exact solvers.

---

### Paper 8 — AI Bridge Bidding + Interactive Visualization (Zhang, 2022)

- **What is it about?** NN bidding framework that predicts bids **and** visually explains the inferred partner cards.
- **Main method:** Three-layer stacked RNN + MLP (supervised).
- **Dataset:** ~4M historical bidding records from Synrey.
- **Most important finding:** 89% bid prediction accuracy + interpretable heatmaps of inferred distributions.
- **Why it matters for NegoPlay:** Direct template for Stage 4 + RQ5 (XAI) — show LLM-agents' inferred state as a heatmap.
- **Limitation:** Pure imitator; no RL; pass-action bias.

---

### Paper 9 — Inference-Decision RL with GAN for Bridge Bidding (Wang, 2024)

- **What is it about?** Conditional GAN deduces partner's hidden hand attributes + RL network makes the bidding decision. Adds counterfactual difference reward.
- **Main method:** Conditional GAN + Policy Gradient RL + reward shaping.
- **Dataset:** 1.05M expert bidding sequences from Vugraph.
- **Most important finding:** Won by +0.359 IMPs against Wbridge5.
- **Why it matters for NegoPlay:** The two-network split (inference + decision) is the modern realization of Rong 2019 (Anchor 2). Cite as the SOTA bridge-bidding-with-inference architecture.
- **Limitation:** Too-high counterfactual reward coefficient → short-sighted greedy behavior.

---

### Paper 10 — Cognitive Mechanisms in Bridge Experts (Liu, 2025)

- **What is it about?** Psychology study showing bridge experts' memory advantage comes from suit-categorization templates + honor-card chunks.
- **Main method:** Behavioral memory tasks + ANOVA.
- **Dataset:** 67 participants (33 experts, 34 novices).
- **Most important finding:** Template theory (visual familiarity) + chunking theory (rule abstraction) together explain expert superiority.
- **Why it matters for NegoPlay:** Tells us how to design Stage 3 prompts — wrap features as cognitive chunks ("balanced 15-17 NT hand"), not raw cards.
- **Limitation:** Static memory tests don't reflect real-time decision pressure.

---

### Paper 11 — RL in Contract Bridge Review (Jarosz, 2022)

- **What is it about?** Chronological lit review of ML in bridge bidding: contextual bandits → Q-Learning → RNNs → deep RL.
- **Main method:** Comparative literature review.
- **Dataset:** N/A (review).
- **Most important finding:** RNN hidden states for bidding-sequence history are the most promising path.
- **Why it matters for NegoPlay:** Saves weeks of lit-review work. Cite as the entry point.
- **Limitation:** No novel algorithm.

---

### Paper 12 — Simple Reproducible Bridge Baseline (Kita, 2024)

- **What is it about?** SOTA via standard recipe — supervised pretraining + PPO + Fictitious Self-Play.
- **Main method:** Supervised + PPO + FSP.
- **Dataset:** 1M SAYC (OpenSpiel) + 12.5M DDS (Pgx).
- **Most important finding:** Beats Wbridge5 by +1.24 IMPs/board.
- **Why it matters for NegoPlay:** **This IS the `harukaki/brl` GitHub repo.** Architecture for PhD methodology chapter, not NegoPlay MVP.
- **Limitation:** Fails without supervised pretraining.

---

### Paper 13 — Joint Policy Search for Multi-agent Collaboration (Tian, 2020)

- **What is it about?** JPS — coordinate sparse policy updates without re-evaluating the full game tree.
- **Main method:** A2C + depth-first Joint Policy Search.
- **Dataset:** 2.5M random hands + DDS.
- **Most important finding:** Beats Wbridge5 by +0.63 IMPs/board.
- **Why it matters for NegoPlay:** Multi-agent communication-protocol shift blueprint (RQ2, RQ4).
- **Limitation:** Search is expensive; used in only 5% of training games.

---

### Paper 14 — Policy Inference in Trick-taking (Skat, 2019)

- **What is it about?** Inference of hidden cards in Skat using opponent models conditioned on action history.
- **Main method:** Supervised NN + Monte Carlo state sampling.
- **Dataset:** Human Skat game logs.
- **Most important finding:** Aggregated opponent models dramatically improve state inference vs. naive location inference.
- **Why it matters for NegoPlay:** Hidden-state inference = core of negotiation. Direct analogue for predicting opponent BATNAs.
- **Limitation:** Vulnerable to deceptive opponents who deviate from average human play.

---

### Paper 15 — Effectiveness of HCP in Bridge (Igra, 2024)

- **What is it about?** Statistical evaluation of the HCP heuristic.
- **Main method:** Logistic regression vs. ground truth.
- **Dataset:** 253,583 synthetic deals + DDS.
- **Most important finding:** HCP is a strong baseline; extra weight on 10s adds negligible accuracy.
- **Why it matters for NegoPlay:** Validates HCP-based features as reasonable Stage 1 clustering inputs.
- **Limitation:** Ignores suit distribution and partnership fit complexities.

---

### Paper 16 — ERL with Action Sequence Search (Wu, 2024)

- **What is it about?** ERL-A2S uses Particle Swarm Optimization on action sequences to escape local optima.
- **Main method:** TD3/PPO + Particle Swarm.
- **Dataset:** Self-play + DDS on AAMAS 2023 benchmark.
- **Most important finding:** Won first place in AAMAS 2023 Bridge Competition.
- **Why it matters for NegoPlay:** Framing for "how do our profile agents avoid converging to one boring strategy?"
- **Limitation:** No interpretability.

---

### Paper 17 — Pgx (Koyamada, 2023)

- **What is it about?** JAX-based bridge/Go/chess simulators on GPU/TPU.
- **Main method:** Hardware-accelerated environment + AlphaZero/PPO.
- **Dataset:** Pure synthetic self-play.
- **Most important finding:** 10–100× faster than OpenSpiel/PettingZoo.
- **Why it matters for NegoPlay:** Not for the MVP. PhD Year 2 infrastructure.
- **Limitation:** No UI for human-vs-agent play.

---

## Anna's quick takeaways

1. **Anchors (Talwadker + Rong)** are the two papers you must be able to talk about for 5 minutes each in the defense. Read them once carefully, take 1-page notes.
2. **Lockett 2007 is the slide you need for the "why mixture of profiles?" question** — keep handy.
3. **James 2021 is your secret weapon** for the bridge → negotiation transfer chapter (RQ3, RQ6). Cite when reviewers ask "but is this analogy even valid?"
4. **For Stage 1 implementation:** start with hand-crafted features (HCP + risk metrics), then v2 = BridgeHand2Vec embeddings (Sztyber-Betley), then v3 = full CognitionNet-style sequence model (Talwadker).
5. **For Stage 3 prompts:** apply Liu 2025's chunking theory — feed hands as expert chunks, not raw cards.
6. **For Stage 4 visualization (XAI):** copy Zhang 2022's heatmap idea — show what each profile-agent *infers* about its counterpart.

---

## Literature Funnel (Dr. Segal's framework — slide 7)

```
15–20 relevant sources   ← target: meet this with the 17 papers here + 3-5 GitHub repos from RELATED_WORK_AND_PLAN.md
        ↓
5–8 core academic papers ← have 11 NegoPlay-focused papers; pick the 5-8 we cite heavily
        ↓
2 anchor papers           ← ⚓ Talwadker 2022 + Rong 2019
        ↓
Baseline + reproducibility ← bridge: random + generic LLM; negotiation: generic agent
```

Status: **literature foundation is complete.** Next step → דוח מכין (preliminary report).

---

## The one prompt to give NotebookLM (kept for reuse)

```
For each of the sources I uploaded, give me a short structured summary in English.
For every paper, use exactly this format:

### Paper: <short title> (<first author>, <year>)

- **What is it about?** (1-2 sentences, plain language)
- **Main method:** (e.g. PPO, Transformer, K-Means, supervised learning)
- **Dataset:** (source + size, e.g. "1M deals from BBO")
- **Most important finding:** (one sentence)
- **Why it matters for me (Anna, PhD on bridge → business negotiation):**
  (1-2 sentences linking to: bidding style clustering, multi-agent LLM,
   coopetition, or negotiation simulation)
- **Limitation worth noting:** (one sentence)

Go through all sources in turn, no extras.
```
