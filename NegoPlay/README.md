# NegoPlay

> **Strategic Profiles from Bridge to Business**
> An empirical investigation of decision-making styles using LLM agents

[![Status](https://img.shields.io/badge/status-active-success)]()
[![Course](https://img.shields.io/badge/course-AI%20Development%20Expert-blue)]()
[![Research](https://img.shields.io/badge/research-LUT%20PhD-purple)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()
[![Python](https://img.shields.io/badge/python-3.11+-blue)]()
[![IDE](https://img.shields.io/badge/built%20in-Antigravity-orange)]()

---

## 🎯 What is NegoPlay?

NegoPlay investigates whether **decision-making profiles** identified
from 149,208 elite bridge tournament hands can predict behavior in
**business negotiation scenarios**.

The hypothesis: a player who succeeds in bridge through a *Slam Hunter*
profile (aggressive, calculated risk-taker) will also succeed in M&A
negotiations using the same strategic pattern.

**Bridge serves as the laboratory. Business negotiation is the application.**

---

## 🧪 Research Question

> *Can LLM agents built from automatically-discovered bridge decision-making
> profiles exhibit **behavioral consistency** (≥70% alignment) between
> winning in bridge bidding simulations and winning in parallel business
> negotiation simulations?*

This is a **proof-of-concept** in a simulated environment, not validation
against real negotiation data.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  STAGE 1: PROFILE DISCOVERY (Machine Learning)              │
│  • Feature engineering on 149K bridge boards                │
│  • K-Means + HDBSCAN clustering → 3-5 profiles              │
│  • Validation: Silhouette ≥ 0.4, p < 0.05                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  STAGE 2: SKILL EXTRACTION (LLM)                            │
│  • Send chunks of 20-30 games per profile to Gemini Flash   │
│  • LLM identifies 5-7 characteristic skills                 │
│  • Output: skill_profiles.json                              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  STAGE 3: AGENT CONSTRUCTION                                │
│  • 4 LLM-based agents, each with profile-specific prompts   │
│  • Methods: make_bid(), respond_to_offer()                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  STAGE 4: DUAL SIMULATION + ALIGNMENT                       │
│  • 50-100 bridge bidding simulations                        │
│  • 50-100 business negotiation simulations                  │
│  • Correlation analysis: bridge_winner == nego_winner?      │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Dataset

| Metric | Value |
|--------|-------|
| Total rows | 149,208 |
| European Championships | 5 (2016–2025) |
| Unique players | 1,572 |
| Players with ≥20 boards | 1,421 (90%) |
| Rows with full bidding | 46,230 (31%) |
| Total bidding tokens | 630,207 |
| Vocabulary size | 39 unique bids |

**Data source:** European Bridge League official records
**Location:** `../../data/raw/all_matches_full.csv` (shared with PhD repo)

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **WSL2 on Windows** (recommended) — or Linux / macOS
- **Google Antigravity IDE** ([download](https://antigravity.google/)) — agent-first development
- **Google Gemini API key** — default provider for cost reasons
  ([get one here](https://aistudio.google.com/apikey))
- **Optional:** `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` — used when a task
  benefits from a different provider, or for cross-model validation runs.

### Installation

```bash
# Clone the parent PhD research repo
git clone https://github.com/annafr2/bridge-business-research.git
cd bridge-business-research/projects/negoplay

# Create virtual environment (Python 3.11)
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up API key
cp .env.example .env
# Edit .env and add: GOOGLE_API_KEY=your_key_here
```

### Run the pipeline

```bash
# Stage 1: Discover profiles (ML only, no API cost)
python -m src.stage1_clustering

# Stage 2: Extract skills via Gemini Flash 2.0
python -m src.stage2_skills

# Stage 3: Build agents (in-memory construction)
python -m src.stage3_agents --build

# Stage 4: Run dual simulations
python -m src.stage4_simulate --bridge-games 50 --negotiation-games 50

# Generate final report
python -m src.report
```

---

## 📁 Project Structure

```
negoplay/
├── README.md              ← You are here
├── CLAUDE.md              ← Instructions for AI assistants
├── PRD.md                 ← Product Requirements Document
├── TASKS.md               ← Detailed task backlog
├── LICENSE                ← MIT
│
├── src/
│   ├── sdk.py             ← Main SDK entry point
│   ├── stage1_clustering/ ← ML profile discovery
│   ├── stage2_skills/     ← LLM skill extraction
│   ├── stage3_agents/     ← Agent construction
│   ├── stage4_simulate/   ← Dual simulation
│   ├── shared/            ← Common utilities
│   └── report.py          ← Results reporting
│
├── notebooks/             ← Exploratory analysis
├── tests/                 ← pytest tests
├── results/               ← Generated outputs (gitignored)
├── docs/
│   ├── architecture.md
│   ├── prompts.md         ← Agent system prompts
│   └── results_analysis.md
│
├── .env.example
├── requirements.txt
└── pyproject.toml
```

---

## 📚 Literature

NegoPlay is grounded in **2 anchor papers** (per Dr. Segal's framework):

- **⚓ Talwadker et al., 2022 — CognitionNet** — methodological anchor for player-style clustering from gameplay sequences (Stage 1)
- **⚓ Rong et al., 2019 — Competitive Bridge Bidding with DNNs** — methodological anchor for the inference-then-decision agent split (Stage 3)
- **💡 Lockett et al., 2007** — theoretical foundation: model unknown opponents as a mixture of cardinal profiles

Full sources:
- [`docs/literature.md`](docs/literature.md) — 17 NegoPlay-focused papers with anchors marked
- [`../PHD_LITERATURE.md`](../PHD_LITERATURE.md) — 9 broader PhD-foundation papers (AlphaZero, MuZero, CFR family)
- [`RELATED_WORK_AND_PLAN.md`](RELATED_WORK_AND_PLAN.md) — 8 GitHub repos analyzed

---

## 🎓 Connection to PhD Research

NegoPlay is the **first empirical deliverable** of Anna's PhD at LUT
University (2026–2030), addressing:

- **RQ1:** Can AI learn player decision-making styles from bridge data?
- **RQ2:** Do these styles improve strategic matching?
- **RQ3 (partial):** Can bridge patterns serve as negotiation analogues?

See [PhD repo root README](../../README.md) for the full research roadmap.

---

## 📈 Course Connection

NegoPlay is the final project of the **AI Development Expert** course
(Limudey Hutz / OASIS Capital Israel). It demonstrates:

| Course Unit | Applied In |
|-------------|------------|
| Foundation Skills (Pandas, NumPy) | Feature engineering |
| Machine Learning (K-Means, HDBSCAN) | Stage 1: Clustering |
| NLP (Word2Vec, tokenization) | Bidding sequence analysis |
| Generative AI (LLMs, prompts) | Stages 2-4: Agents |
| Multi-Agent Systems | Dual simulation |
| Deep Learning (LSTM) | *Stretch goal* |

---

## 💰 Cost Estimation

NegoPlay is designed around **Gemini Flash 2.0** as the default provider
for cost efficiency. Claude and OpenAI may be used selectively for
cross-model validation or final synthesis, at higher token cost.

| Component | Tokens | Est. Cost |
|-----------|--------|-----------|
| Skill extraction (4 profiles) | ~200K | ~$0.10 |
| 50 bridge simulations | ~650K | ~$0.25 |
| 50 negotiation simulations | ~650K | ~$0.25 |
| Buffer for iterations | — | ~$5 |
| **Total MVP** | **~1.6M** | **~$5–10** |

> 💡 **Bonus:** Google Antigravity IDE is currently in free public preview
> with Gemini 3 Pro included for coding tasks — separate from the
> project's API usage.

---

## 🛠️ Development Setup with Antigravity

NegoPlay is built in **Google Antigravity**, an agent-first IDE.
Recommended workflow:

1. **Manager View** — orchestrate multiple agents in parallel
   - Agent A: feature engineering on bridge data
   - Agent B: prompt testing for NegoPlay agents
   - Agent C: literature review research

2. **Planning Mode** — for complex tasks (clustering, agent design)
3. **Fast Mode** — for quick iterations (prompt tweaking)
4. **Artifacts** — review Implementation Plans and Diffs before approval

See [CLAUDE.md](./CLAUDE.md) for detailed development conventions.

---

## 📚 Citation

If you use this work, please cite:

```bibtex
@misc{negoplay2026,
  author = {Ben-Shushan, Anna},
  title = {NegoPlay: Strategic Profiles from Bridge to Business},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/annafr2/bridge-business-research}},
  note = {Course project (AI Development Expert) and PhD baseline (LUT University)}
}
```

---

## 📝 License

MIT License — see [LICENSE](./LICENSE) for details.

This permissive license allows anyone (including future commercial use)
to build on this research.

---

## 👤 Author

**Anna Ben-Shushan**
PhD Candidate, LUT University (Finland)
Lecturer, Sami Shamoon College of Engineering
PhD Supervisor: Prof. Jari Hämäläinen
Course Supervisor: Dr. Rami

- 🌐 GitHub: [@annafr2](https://github.com/annafr2)

---

## 🙏 Acknowledgments

- **Prof. Jari Hämäläinen** (LUT) — PhD supervision
- **Dr. Rami** — Course supervision and methodology guidance
- **Dr. Yoram Segal** — Course project framework
- **European Bridge League** — Data accessibility
- **Google Antigravity Team** — IDE for agent-first development
