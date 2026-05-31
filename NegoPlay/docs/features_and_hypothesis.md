# Features & Hypothesis — Plain-English Guide

> A simple map of the NegoPlay data: what every column means, which ones are
> *features* and which are not, and what the research is actually trying to
> predict (`y = f(x)`). Written for a non-specialist reader.

---

## 1. Raw columns vs. features — what's the difference?

A **raw column** is just a piece of recorded data (like "who sat North").
A **feature** is a *number that describes a player's behaviour*, computed from
the raw columns. **Most raw columns are NOT features** — they are IDs, names,
or the raw material we compute features from.

The raw dataset has **48 columns**. They fall into 4 groups:

| Group | Example columns | Is it a feature? | Why |
|-------|-----------------|------------------|-----|
| **IDs / metadata** | `match_id`, `board`, `round`, `year`, `competition`, `category` | ❌ No | Just tells us *which* game and *when* |
| **Player names** | `open_north`, `open_south`, … `closed_west` (8 cols) | ❌ No | Just tells us *who* sat where |
| **Raw game record** | `contract`, `bidding`, `tricks`, `declarer`, `lead`, `ns_score`, `ew_score`, the 16 card columns, `dealer`, `vulnerability` | ⚠️ Not directly | These are the *raw material* — we COMPUTE features from them |
| **Quality flags** | `has_bidding`, `has_cards` | ❌ No | Just says "is this info available or not" |

**Key point for the supervisor:** out of 48 raw columns, the features come
mainly from just **3**: `contract` (what was bid), `bidding` (how it was bid),
and `tricks` (did it work).

---

## 2. The 10 features we computed

Each feature is **one number per player**, describing one behaviour. We built
10, in two families.

### Family A — from the `contract` column ("WHAT the player bid")

| Feature | Plain meaning | Example |
|---------|---------------|---------|
| `slam_rate` | how often they bid a huge contract (level 6–7) | 0.10 = bids slam 10% of the time |
| `nt_rate` | how often they play NoTrump | 0.38 = 38% of contracts are NT |
| `partscore_rate` | how often they stop low/safe (level 1–3) | 0.68 = stops low 68% of the time |
| `game_rate` | how often they bid exactly game level | — |
| `double_rate` | how often their contract got doubled | — |

### Family B — from the `bidding` column ("HOW the player bid")

| Feature | Plain meaning |
|---------|---------------|
| `opening_rate` | how often they open the auction |
| `preempt_rate` | how often they open high (level 2+) — aggressive |
| `intervention_rate` | how often they butt into the opponents' auction |
| `penalty_double_rate` | how often they double opponents to punish them |
| `avg_bids_per_board` | how many calls they make per deal |

### What actually went into K-Means: **8 features**

We dropped **2** features because they were almost the same for everybody
(they don't separate players — "low variance"):

- ❌ `avg_level` (everyone averages ~3.3)
- ❌ `avg_bids_per_board`

So **8 features** entered the clustering.

---

## 3. Independent vs. dependent variables

- **Independent variables (the `x`)** = the **inputs** we measure and feed in.
  Here: the **8 behaviour features** (slam_rate, nt_rate, …). They are
  "independent" because they are given to the model as-is.

- **Dependent variable (the `y`)** = the **output** the model produces, which
  *depends on* the inputs. Here: the **profile label** (Slam Hunter, Insurance
  Player, …).

| | Variable | Role |
|---|----------|------|
| **x** (independent) | the 8 behaviour features | what we put IN |
| **y** (dependent) | the player's profile | what comes OUT |

A simple analogy: `x` = a student's test scores in each subject; `y` = "what
kind of student are they" (science type? arts type?). The type *depends on*
the scores.

---

## 4. The big idea: `y = f(x)` at TWO levels

The whole thesis is two `y = f(x)` steps chained together.

### Level 1 — inside bridge (DONE ✅)

```
x = how a player bids        (the 8 features)
f = the clustering model      (extreme-percentile + significance test)
y = which behaviour type      (Slam Hunter / Insurance / Fighter / NT / Generalist)
```

In words: *"From how you bid, we figure out what kind of decision-maker you are."*

### Level 2 — from bridge to business (the real research question ⏳)

```
x = the bridge behaviour profile   (aggressive? cautious?)
f = the LLM agent                  (acts in character)
y = behaviour in a business negotiation
```

In words: *"The decision style we learned from bridge predicts how the same
'person' behaves in a business negotiation."*

We test Level 2 by checking whether the profile **ranking** is the same in both
domains, using **Spearman's ρ** (a correlation of rankings). Target: **ρ ≥ 0.70**.

> ⚠️ **Honest status:** Level 1 is proven. Level 2 is the experiment running
> now (Stage 4). The Spearman ρ — the heart of the thesis — is not computed yet.

---

## 5. Why this matters for business / negotiation

The problem in business: real negotiation data is **secret and unmeasured** —
you can't train a model on it. Bridge is a **clean laboratory** for the *same*
decisions (when to take a risk, when to stop, when to pressure the other side),
with 149,000 measurable outcomes from expert players.

If the bridge profile predicts negotiation behaviour, the `y` lets us:

| What the `y` enables | Business example |
|----------------------|------------------|
| **Identify the counterpart's style** | "This negotiator behaves like a Slam Hunter → expect bold offers, prepare a defence" |
| **Predict their moves** | "An Insurance Player closes fast → we can push for a better price" |
| **Train managers safely** | "Practise against 5 negotiator types — no real opponent needed" |

**One-sentence takeaway:** *bridge is a measurable training ground for learning
and predicting negotiation styles, without needing private business data.*

---

## 6. Known limitations (be upfront about these)

1. **Level 2 not yet proven** — the bridge→business transfer (Spearman ρ) is
   the current experiment, not a finished result.
2. **Features are mostly outcomes** — most features come from the *final
   contract*, not the full bid-by-bid process. Only the 5 `bidding` features
   capture the process. A reviewer may note this is closer to "result" than
   "decision process".
3. **Small profile groups** — only 5 profiles, so Spearman ρ has low statistical
   power (needs to be near ±1 to be significant).

---

*Generated as a supervisor-facing reference. Source data: 149,208 EuroBridge
rows (2016–2025); 563 qualifying players; 8 clustering features.*
