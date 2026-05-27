# NegoPlay — Agent Architecture & Experiment Design

> **Plain-English design document**
> How the 5 LLM agents are built, how they think, and how we test them.
> *Last updated: May 2026*

---

## 1. The Big Picture

NegoPlay builds **5 LLM agents**, each with a different personality.
We then make them play **two completely different games**:

1. **Bridge bidding** — a card game where they decide how high to bid
2. **Business negotiation** — a money game where they decide deals

**The key question:**
> Does an agent that wins in Bridge also win in Negotiation?

If yes → the personality is *real* (it transfers across domains).
If no → the personality is just a costume the LLM wears.

---

## 2. Meet the 5 Agents

Each agent is a **different personality**, discovered from real bridge data
of 807 elite players. They are not invented — they came from statistics.

### 🎯 Agent 1: Slam Hunter

**What they do in Bridge:**
Bid huge contracts (6 or 7 level). They go for the jackpot 11.6% of the time
— almost 3× more than average players.

**Personality in plain English:**
- A **calculated risk-taker**
- Believes in themselves and their partner
- Sees the big picture, ignores small wins
- Dominant, not passive

**How they will behave in Negotiation:**
- Offers big "all-in" deals, not small pieces
- Wants the whole company, not just shares
- Will walk away if the offer feels too small
- High reward, high risk

---

### 🛡️ Agent 2: Insurance Player

**What they do in Bridge:**
Stop at low contracts (partscore, level 1-3) in 69.3% of hands. They take
the small sure win, never the big risky one.

**Personality in plain English:**
- **Loss-averse** (hates losing more than loves winning)
- Conservative — small sure profit beats a big maybe-profit
- Patient, never panics, never chases
- "Bird in the hand" thinker

**How they will behave in Negotiation:**
- Accepts the first reasonable offer
- Closes deals fast, exits quickly
- Prefers cash over stock, guaranteed terms over upside
- Strength: never blows up a deal
- Weakness: leaves money on the table

---

### 💥 Agent 3: Fighter

**What they do in Bridge:**
Penalty-double opponents in 13.4% of hands — 1.6× the average. When they
think the opponent bid too high, they punish them.

**Personality in plain English:**
- **Aggressive toward opponents**
- Watches for weaknesses
- Competitive — not happy just winning, wants to make the other side lose
- Emotional, intense

**How they will behave in Negotiation:**
- Rejects offers with aggressive counter-offers
- Pushes back hard — "this price is not serious"
- Hunts for weakness in the other side's contract
- Threatens to walk away as a tactic
- Strength: never accepts a bad offer
- Weakness: ego can kill the deal

---

### ♠️ Agent 4: NT Specialist

**What they do in Bridge:**
Play 40.8% of their contracts in **No Trump** — 1.5× the average. NT requires
a balanced hand and precise math (counting exactly 9 tricks).

**Personality in plain English:**
- **Analytical**
- Prefers balance over raw power
- Systematic, methodical, math-driven
- Cold-blooded — never carried away by emotion

**How they will behave in Negotiation:**
- Bases everything on data and models (DCF, multiples, comparables)
- Rejects "gut feeling" offers — demands justification
- Builds precise contracts with covenants and earn-outs
- Strength: never makes a math mistake
- Weakness: slow, might miss deal momentum

---

### 👥 Agent 5: Generalist (The Baseline)

**What they do in Bridge:**
**The average player** — 70% of all players in our data. No extreme behavior
in any direction. They are the "normal" baseline.

**Personality in plain English:**
- **Flexible** — adapts to the situation
- Not exceptional at anything, not bad at anything either
- Mirrors the opponent's strategy
- Statistical average

**Why this agent matters — the Control Group:**
Generalist is our **measuring stick**. Every other profile is tested
*against Generalist*. This is how we isolate the effect of personality.

**How they will behave in Negotiation:**
- Default-reasonable: market price, sensible terms
- Closes deals at average rates
- Wins ~50% of the time against another Generalist

---

## 3. How the Agents Are Built (Technical)

Each agent is a **Python class** wrapping a **Gemini Flash 2.0** call.

### Agent Class Structure

```python
class Agent:
    def __init__(self, profile_name: str):
        self.profile = profile_name
        self.system_prompt = load_prompt(profile_name)
        self.llm_client = GeminiClient(temperature=0.3)

    def make_bid(self, hand: list[str], auction: list[str]) -> str:
        """Bridge bidding decision."""
        ...

    def respond_to_offer(self, offer: dict, scenario: dict) -> dict:
        """Negotiation decision."""
        ...
```

### What Goes Into Each Agent (the System Prompt)

Every agent gets a system prompt with 5 parts:

1. **Identity** — "You are a Slam Hunter."
2. **Core skills** — 5-7 traits extracted by Gemini in Stage 2
   (e.g., "calculated risk-taker, optimistic about partner's hand")
3. **Decision rules** — domain-specific (bidding laws, contract logic)
4. **Output format** — strict JSON schema, no free text
5. **Few-shot examples** — 1-2 real examples from the bridge data

**Important:** Skills come from **real bridge hands**, not from us writing
"be aggressive." This avoids the *tautological alignment* trap (see Section 6).

---

## 4. The Two Games

### Game A: Bridge Bidding (4 seats)

Bridge needs 4 players in two partnerships: **N-S** vs **E-W**.

```
        North
         |
West --- + --- East
         |
        South
```

Agents bid in turn. The system records every bid. We measure:
- Final contract reached
- Whether they made it
- IMP score (the official bridge scoring)

### Game B: Business Negotiation (2 sides)

We simulate **4 scenarios** based on real negotiation literature:
1. **M&A acquisition** — buyer vs seller of a company
2. **Salary negotiation** — candidate vs hiring manager
3. **Vendor contract** — client vs supplier
4. **Partnership equity** — co-founder split

Two agents take turns sending offers/counter-offers. The system records:
- Final deal terms
- Walk-away or agreement
- Surplus captured by each side (the "win")

---

## 5. Experiment Design (How We Test the Alignment)

### Phase 1 — Baseline (every profile vs Generalist)

This is the **main experiment**. It answers the core research question.

| Bridge Matchup | Negotiation Matchup | Games |
|---|---|---|
| Slam Hunter (NS) vs Generalist (EW) | Slam Hunter vs Generalist | 50 + 50 |
| Insurance (NS) vs Generalist (EW) | Insurance vs Generalist | 50 + 50 |
| Fighter (NS) vs Generalist (EW) | Fighter vs Generalist | 50 + 50 |
| NT Specialist (NS) vs Generalist (EW) | NT Specialist vs Generalist | 50 + 50 |

**Total: 400 games.**

**Analysis:**
For each of the 4 profiles, compute:
- `win_rate_bridge` = % of bridge games won vs Generalist
- `win_rate_nego` = % of negotiations won vs Generalist

Then compute **Spearman ρ** between the two columns across 4 profiles.

**Success criterion:** ρ ≥ 0.7 (the alignment threshold from the PRD).

---

### Phase 2 — Story Matchups (for the paper)

The 4 Phase-1 results give us the headline number. But papers need
**stories** — interesting matchups that reveal interaction effects.

| Matchup | What it reveals |
|---|---|
| Slam Hunter vs Insurance Player | Aggressive vs Conservative — cleanest contrast |
| Fighter vs NT Specialist | Emotion vs Logic |
| Slam Hunter vs Fighter | Two aggressives — who dominates? |

**50 bridge games + 50 negotiations × 3 matchups = 300 games.**

---

### Phase 3 — Falsifiability Tests (for the PhD paper)

This is what separates a course project from a PhD-quality contribution.

| Test | Purpose |
|---|---|
| **Inverse profile** — agent told "aggressive in bridge, conservative in negotiation" | If alignment still hits 70%, the alignment is fake (just LLM following prompts) |
| **Profile-blind agent** — no identity, only rules | Compare against Generalist — should be similar |

**50 bridge + 50 negotiation × 2 tests = 200 games.**

---

## 6. The Tautological Alignment Trap (Critical)

**The danger:**
If we prompt Gemini *"You are aggressive, you take risks"*, it will behave
aggressively in **both** bridge **and** negotiation — just because we told
it to. The alignment we measure will be **fake**.

**The fix:**
1. In Stage 2, Gemini extracts skills from **real bridge hands**, not from
   our descriptions. We send it: *"Here are 20 hands this player bid.
   What 5-7 skills do you see?"*
2. The skills are **bridge-specific** ("opens light on long suits,
   doubles 1NT on flat hands with 15+ HCP"), not generic personality words.
3. Phase 3 Falsifiability tests confirm the alignment is real, not a prompt
   artifact.

---

## 7. Cost Estimation (Detailed)

### Assumptions

| Parameter | Value | Source |
|---|---|---|
| Gemini Flash 2.0 input | $0.075 per 1M tokens | Google pricing |
| Gemini Flash 2.0 output | $0.30 per 1M tokens | Google pricing |
| Avg tokens per bridge bid | ~500 in, ~50 out | Estimated from prompt size |
| Avg bids per bridge game | ~12 (3 rounds × 4 seats) | Bridge typical |
| Avg tokens per negotiation turn | ~800 in, ~150 out | Estimated |
| Avg turns per negotiation | ~10 | Typical M&A simulation |

### Stage 2: Skill Extraction (one-time)

| Item | Calls | Tokens | Cost |
|---|---|---|---|
| 4 profiles × 5 chunks × ~30 hands per chunk | 20 | 200K in / 20K out | $0.02 |
| Aggregation across chunks | 4 | 20K in / 2K out | <$0.01 |
| **Stage 2 subtotal** | **24** | **~242K** | **~$0.03** |

### Stage 4: Simulations

**Per bridge game:** ~12 bids × (500 in + 50 out) = 6,000 in + 600 out tokens
**Per negotiation:** ~10 turns × (800 in + 150 out) = 8,000 in + 1,500 out tokens

**Per game cost:**
- Bridge: (6,000 × $0.075 + 600 × $0.30) / 1M = **$0.00063**
- Negotiation: (8,000 × $0.075 + 1,500 × $0.30) / 1M = **$0.00105**

| Phase | Bridge games | Nego games | Bridge cost | Nego cost | Total |
|---|---|---|---|---|---|
| **Phase 1** — Baseline | 200 | 200 | $0.13 | $0.21 | **$0.34** |
| **Phase 2** — Stories | 150 | 150 | $0.09 | $0.16 | **$0.25** |
| **Phase 3** — Falsifiability | 100 | 100 | $0.06 | $0.11 | **$0.17** |
| **Subtotal — simulations** | 450 | 450 | $0.28 | $0.48 | **$0.76** |

### Adding a Safety Buffer

The numbers above are clean estimates. Real life:
- Retries on rate limits
- Debugging runs (we'll re-run games during development)
- Cross-provider validation (running 10% of games on Claude + GPT-5 for paper robustness)

| Item | Cost |
|---|---|
| Stage 2 skill extraction | $0.03 |
| Stage 4 simulations (Phases 1-3) | $0.76 |
| **Subtotal** | **$0.79** |
| Development buffer (5×) | $4.00 |
| Cross-model validation (Claude + GPT-5, 10% of games) | $20.00 |
| Cross-model writing / synthesis (Pro tier) | $10.00 |
| Emergency buffer | $15.00 |
| **TOTAL** | **~$50** |

**Bottom line:** Even with all phases, full cross-model validation, and
generous buffers, we stay at the $50 hard cap from the PRD.

---

## 8. Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│  Stage 1 (DONE) — Player Profiles                           │
│  149K bridge hands → 807 players → 5 profiles               │
│  Output: data/processed/player_profiles.csv                 │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 2 — Skill Extraction (Gemini Flash 2.0)              │
│  For each profile:                                          │
│    sample 20-30 hands × 5 chunks                            │
│    → Gemini → 5-7 bridge-specific skills                    │
│  Output: data/processed/skill_profiles.json                 │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 3 — Agent Construction                               │
│  5 agents, each with:                                       │
│    • Profile identity                                       │
│    • 5-7 extracted skills                                   │
│    • Bridge bidding logic                                   │
│    • Negotiation logic                                      │
│  Output: src/stage3_agents/*.py + docs/prompts.md           │
└────────────────────────────┬────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
┌──────────────────────────┐  ┌──────────────────────────┐
│  Stage 4A — Bridge       │  │  Stage 4B — Negotiation  │
│  450 bridge games        │  │  450 negotiation games   │
│  Output: bridge_runs.csv │  │  Output: nego_runs.csv   │
└──────────────┬───────────┘  └──────────────┬───────────┘
               │                             │
               └──────────────┬──────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 5 — Alignment Analysis                               │
│  Spearman ρ(win_rate_bridge, win_rate_nego)                 │
│  Falsifiability tests vs inverse + blind agents             │
│  Output: results/final_report.md + plots                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 9. Open Questions (For Supervisor Discussion)

1. **Negotiation scenarios** — should we use 1 scenario × 50 games, or
   4 scenarios × 12-13 games each? (The latter gives generalization but
   smaller per-scenario power.)
2. **Bridge partnerships** — should both seats of a partnership be the
   same profile, or different? Current plan: both seats same profile
   (cleaner causal interpretation).
3. **Number of bidding rounds** — bridge auctions can be long. Should we
   cap at 12 rounds and call it timeout? Real auctions average 6-10.
4. **Phase 3 inclusion** — required for the PhD paper, optional for
   the course. Include now or after course submission?

---

## 10. Status

- ✅ Stage 1 (profiles) — done, 807 players, 5 profiles
- ⏳ Stage 2 (skills) — next
- ⏳ Stage 3 (agents) — after Stage 2
- ⏳ Stage 4 (simulations) — after Stage 3
- ⏳ Stage 5 (analysis) — after Stage 4

See `TASKS.md` for the detailed task list.
