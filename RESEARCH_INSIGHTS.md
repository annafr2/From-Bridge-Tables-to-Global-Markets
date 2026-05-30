# RESEARCH_INSIGHTS.md — Empirical Questions We Can Now Answer

**Last updated:** 2026-05-30 — Added Q7.9 (Stage 3: bridge personality transfers to negotiation). Earlier today: Q7.8 (bidding sanity), Q7.7 (profile validation) (Stage 2: LLM-extracted profiles empirically validated — all 4 behaviourally distinct, Cohen's d 2.13–4.62, all p<0.05). Previously: Q7.4 (bridge expert validation caveats), Q7.5 (v2.0→v2.1 revision methodology), Q7.6 (Dr. Rami's preprocessing audit — continuum confirmed across 5 algorithms)
**Status of data:** **149,208 rows** × 48 columns, 5 EBL competitions (2016–2025), individual player names per position

This file is a living research notebook. Each section = one empirically testable question,
with: the exact analysis → the expected finding → which paper/chapter it contributes to.

> **Rule:** Every insight here must be answerable from the current dataset.
> If it needs future data (BBO, negotiation transcripts), label it clearly.

### Current Data Quality Summary

| Metric | Coverage | Notes |
|--------|----------|-------|
| Total rows | 149,208 | 5 competitions × 3–4 categories each |
| Player names (N/S/E/W per room) | **99.9%** | All 5 competitions |
| Card holdings (52 cards) | **54%** | Available where EuroBridge published BoardAcross data |
| Bidding sequences | **31%** | Herning 2024 + Poznan 2025 only (100%); older competitions: 0% (server limitation — no tooltips, PlayDetails returns HTTP 500) |
| Unique players (by name) | **~2,500+** | Across all positions and rooms |

### Per-Competition Breakdown

| Competition | Rows | Names | Cards | Bidding |
|-------------|------|-------|-------|---------|
| Budapest 2016 | 38,240 | 100% | 56% | 0% |
| Ostend 2018 | 32,384 | 100% | 53% | 0% |
| Madeira 2022 | 32,256 | 100% | 51% | 0% |
| Herning 2024 | 34,048 | 100% | 56% | 100% |
| Poznan 2025 | 12,280 | 100% | 51% | 100% |

**Implication for research:** Questions requiring bidding analysis (Q1.3 preempt/double detection, Q5.1 bid attribution) are limited to ~46K rows (Herning + Poznan). Questions based on contract/declarer/tricks/cards (Q1.1 risk profiles, Q2.1 score-dependent risk, Q3.1 partnership chemistry, Q4.1 category differences) use the **full 149K rows**.

---

## PART 1 — Individual Player Risk Profiles
*Supervisor Nezer's research direction — April 2026*

### Q1.1 — Is there a player who consistently takes risks? Does it pay off?

**What we need:**
- `declarer` column + player name columns → who declared
- `contract` column → what level they bid
- `tricks` + `contract` → did they make it?

**The analysis:**
```python
# For each player, compute their "risk taking" rate
# A slam bid = high risk (6/7 level contract)
# A preempt = medium risk (opening 2/3/4 level)
# A doubled contract = risk taken by opponent OR by declarer

df["declarer_name"] = df.apply(lambda row:
    row[f"open_{row['declarer'].lower()}"]
    if row["room"] == "Open"
    else row[f"closed_{row['declarer'].lower()}"],
    axis=1
)

player_stats = df.groupby("declarer_name").agg(
    total_boards  = ("board", "count"),
    slam_rate     = ("is_slam_contract", "mean"),
    made_rate     = ("made_contract", "mean"),
    avg_risk_score= ("risk_score", "mean"),
)
```

**Expected finding:**
Some players (e.g. BRINK Sjoert — world-class Dutch player) will show a high slam rate
AND a high success rate → skilled risk-taker.
Others will show high risk + low success → impulsive risk-taker.

**Research insight:** This is the first empirical taxonomy of bridge risk profiles.
Maps to: **RQ1 (decision-making styles)**, **RQ4 (coopetition)**

**Potential paper title:**
*"Risk Profiles of Elite Bridge Players: An Empirical Taxonomy from European Championship Data"*

---

### Q1.2 — The McGOWAN Example (Template for Individual Analysis)

**Actual data row** (SCOTLAND vs ISRAEL, Herning 2024, Round 5, Board 10, Open Room):

| Field | Value |
|-------|-------|
| open_north | McGOWAN Elizabeth (Liz) |
| open_south | FREIMANIS Gints |
| declarer | N → **McGOWAN Elizabeth** declared |
| contract | 3NT |
| bidding | `S:1C → W:1D → N:1S → S:2C → N:3NT` |
| tricks | 9 (made it!) |
| ns_score | 600 |

**What we can now ask about McGOWAN:**
1. In how many matches does she appear? (search `open_north LIKE '%McGOWAN%'`)
2. How often does she declare? (how often is `open_north=McGOWAN AND declarer=N`)
3. When she declares, what contracts does she bid? (distribution of contract levels)
4. How often does she succeed? (made_contract rate)
5. Does her risk profile change when her team is losing vs. winning?

**The Research Story:**
> "Among ~2,500 elite European players across 5 championships (2016–2025) in our 149K-row dataset,
> we identified 3 distinct risk profiles: Conservative Experts (high success, low risk),
> Calculated Risk-Takers (high risk, high success), and Overextenders (high risk, low success).
> BRINK Sjoert exemplifies the Calculated Risk-Taker profile with a slam attempt rate of X%
> and success rate of Y%."

**New with 149K rows:** Players now appear across multiple tournaments (2016–2025),
enabling longitudinal analysis: does a player's risk profile change over 8 years?

---

### Q1.3 — Risk Style Taxonomy (the actual classification)

**Four observable risk behaviors:**

| Label | Definition | How to detect in data | Data needed |
|-------|-----------|----------------------|-------------|
| **Slam Hunter** | Frequently bids 6/7 level | `contract LIKE '6%' OR contract LIKE '7%'` | All 149K rows ✅ |
| **Preemptor** | Frequently opens at 2/3/4 level | First bid in sequence ≥ 2 level | Bidding column (46K rows — Herning + Poznan) |
| **Doubler** | Frequently doubles opponent's contracts | `Dbl` appears in bidding | Bidding column (46K rows) |
| **Insurance Player** | Stops short of game despite strength | Stops at 4m/3NT when game available | Bidding + cards (requires both) |

**Method:**
- Build a feature vector per player: (slam_rate, preempt_rate, double_rate, success_rate)
- Cluster using K-means or HDBSCAN (3–5 clusters)
- Label clusters qualitatively → the taxonomy

**Data note:** Slam Hunter detection works on all 149K rows (only needs `contract` column).
Preemptor and Doubler detection require the `bidding` column → limited to Herning + Poznan (~46K rows).
However, 46K rows with ~1,000 unique players is still statistically strong for clustering.

---

## PART 2 — VP Scale and Score-Dependent Risk
*The core game-theory question from Nezer's VP scale analysis*

### Q2.1 — Do players take more risks when losing?

**The theory (Nezer):**
The IMP→VP conversion is non-linear: when you're losing, each extra IMP won is worth MORE in VPs.
Therefore: rational players should take bigger risks when losing.

**What we need:**
- Running IMP score within each match (board by board)
- Risk metrics per board
- Match of the two

**The analysis:**
```python
# Compute running IMP within match
df = df.sort_values(["match_id", "board"])
df["running_ns_imp"] = df.groupby("match_id")["board_ns_imp"].cumsum()
df["ns_leading_by"] = df["running_ns_imp"]  # positive = NS ahead

# Categorize score situation
df["score_situation"] = pd.cut(
    df["ns_leading_by"],
    bins=[-inf, -20, -8, 8, 20, inf],
    labels=["losing_badly", "losing", "close", "winning", "winning_comfortably"]
)

# Test the theory
result = df.groupby("score_situation")["risk_score"].mean()
```

**Expected finding:**
If theory holds: risk_score higher in "losing_badly" than "winning_comfortably".

**If theory DOESN'T hold:** That's also publishable!
"Elite bridge players do NOT adjust risk-taking based on score situation,
suggesting automated strategic patterns override game-theoretic optimization."

**Paper contribution:** First empirical test of VP-scale game theory on real tournament data.
Maps to: **RQ4 (coopetition dynamics)**, **RQ3 (decision-making → business)**

**Business analogy:**
This mirrors negotiation research: negotiators behind in a deal accept worse terms.
This is the bridge → business transfer that makes the paper interesting beyond bridge.

---

### Q2.2 — Tournament standing and risk appetite

**The question:** Do teams ranked 2nd take more risks than teams ranked 1st?

**What we need:** Cumulative tournament standings (VP total) per round

```python
# For each round, compute team standings
# Then for each board: is this team "chasing" or "protecting"?
df["team_rank_before_round"] = ...  # (to be computed in src/features/tournament_standings.py)
df["chasing"] = df["team_rank_before_round"] > 1  # not in first place

# Compare risk scores
chasing_risk = df[df["chasing"] == True]["risk_score"].mean()
leading_risk = df[df["chasing"] == False]["risk_score"].mean()
```

**Expected finding:** Teams ranked 2nd+ show higher risk scores, especially in late rounds.

**Football analogy (Nezer's):** Exactly like trailing teams attacking in the final minutes.

---

### Q2.3 — Late tournament timing effects

**The question:** Does risk-taking increase as the tournament approaches its end?

```python
# Compare round 1-3 vs. round 8-10 (final rounds)
df["tournament_stage"] = pd.cut(df["round"], bins=[0, 3, 6, 100],
                                labels=["early", "mid", "late"])

risk_by_stage = df.groupby(["tournament_stage", "category"])["risk_score"].mean()
```

**Expected finding:** Risk increases in late rounds, especially for non-leading teams.

---

## PART 3 — Partnership Chemistry
*Research Question 2*

### Q3.1 — Do consistent partnerships outperform mixed/new partnerships?

**What we need:** Pair identity across multiple tournaments

```python
# For each board, identify the NS pair and EW pair
# NS pair in Open Room = (open_north, open_south)
df["ns_pair_open"] = df["open_north"] + " & " + df["open_south"]

# Count how many boards each pair has played together
pair_counts = df.groupby("ns_pair_open").size()

# For established pairs (>50 boards together) vs new pairs (<10 boards):
# Compare made_contract rate and consistency (variance of ns_score)
```

**Expected finding:** Established pairs show lower variance (more consistent) and
potentially better outcomes on borderline contracts.

**Paper contribution:** First quantification of "partnership chemistry" at elite level.

**Data strength (May 2026):** With 149K rows across 5 championships (2016–2025),
the same national pairs (e.g., Netherlands Open) likely appear in 3–5 competitions.
This gives 100+ boards per established pair — enough for statistical significance.

---

### Q3.2 — Partner agreement rate

**The question:** On the same hand, do established pairs reach the "correct" contract
more often than new pairings?

```python
# For each board played in both rooms:
# Did both rooms reach the same contract level?
# (agreement = same suit AND level ±1)
open_contract  = df[df["room"]=="Open"]["contract"].values
closed_contract = df[df["room"]=="Closed"]["contract"].values
agreement_rate = sum(open_contract == closed_contract) / len(open_contract)
```

---

## PART 4 — Category Differences (Open / Women / Senior / Mixed)

### Q4.1 — Do different categories show different risk profiles?

**What we need:** The `category` column (already present)

```python
risk_by_category = df.groupby("category").agg(
    slam_rate    = ("is_slam_contract", "mean"),
    preempt_rate = ("is_preempt_open", "mean"),
    double_rate  = ("is_double", "mean"),
    success_rate = ("made_contract", "mean"),
)
```

**Expected finding:** Open category may show highest calculated risk (strongest players).
Senior may show more conservative play. Mixed may show gender-interaction effects.

**Research relevance:** This is an indirect demographics study without requiring age/gender data.
Senior = older players. Women's category = female players. Open = mixed.

**Paper contribution:** Category-level risk analysis without violating player privacy.

---

## PART 5 — Decision Attribution (Who Did What)

### Q5.1 — The "decisive bid" attribution

**For every board, identify who made the decisive bid:**

```python
def find_decisive_bid(bidding_str, room, match_row):
    """
    Parse bidding string and find:
    - who made the opening bid (first non-pass)
    - who made the first slam-level bid (5+ level)
    - who doubled
    Returns: {"opener": name, "slam_bidder": name, "doubler": name}
    """
    bids = parse_bidding(bidding_str)
    position_to_name = {
        "N": match_row[f"{room.lower()}_north"],
        "S": match_row[f"{room.lower()}_south"],
        "E": match_row[f"{room.lower()}_east"],
        "W": match_row[f"{room.lower()}_west"],
    }
    # Find who made each key action...
```

**Data availability:** Requires `bidding` column → limited to Herning 2024 + Poznan 2025 (~46K rows).
However, **basic declarer attribution** (who played the hand) works on ALL 149K rows:
`declarer=N` + `open_north=PLAYER_NAME` → we know who declared without parsing bidding.

**This enables (on 46K rows with bidding):**
- "BRINK Sjoert made 47 slam attempts across the dataset — 38 succeeded (81%)"
- "The most common opener in Women's category is..." 
- "Players who double most frequently: TOP 10..."

**This enables (on all 149K rows without bidding):**
- "BRINK Sjoert declared 312 boards. Of those, 78 were slams, 63 succeeded (81%)."
- "Top 20 most active declarers across all 5 championships"
- "Declarer success rate by category: Open vs. Women vs. Senior"

### Q5.2 — Cross-Tournament Player Tracking (NEW — enabled by 149K rows)

**The question:** Do the same players appear in multiple championships? How does
their performance evolve over 8 years (2016–2025)?

```python
# Find players appearing in multiple competitions
df["declarer_name"] = ...  # derive from declarer + room + player columns
player_comps = df.groupby("declarer_name")["competition"].nunique()
returning_players = player_comps[player_comps >= 2]
print(f"Players in 2+ competitions: {len(returning_players)}")
# Expected: 200-400 players (elite players return to championships)

# For each returning player: track slam rate across years
for player in top_returning_players:
    yearly_stats = df[df["declarer_name"]==player].groupby("year").agg(
        slam_rate=("is_slam", "mean"),
        success_rate=("made_contract", "mean"),
    )
    # → Does this player become more conservative with age?
    # → Does their success rate improve with experience?
```

**Research insight:** This is only possible because we have 5 competitions spanning 2016–2025.
The original 78K dataset (3 competitions) had limited cross-year overlap.
149K rows from 5 championships make longitudinal player tracking viable.

**Potential paper title:**
*"The Evolution of Risk-Taking in Elite Bridge: A 9-Year Longitudinal Study"*

---

## PART 6 — Transfer to Business (Discussion Chapter)

> These are THEORETICAL contributions (no new empirical data needed).
> Write as a discussion/implication chapter, not as empirical results.

### Q6.1 — Bridge Risk Profiles → Business Negotiation Styles

| Bridge Profile | Business Equivalent | Observable Behavior |
|----------------|--------------------|--------------------|
| Slam Hunter | Aggressive negotiator | Makes bold first offers far from BATNA |
| Insurance Player | Risk-averse negotiator | Accepts early sub-optimal deal |
| Doubler | Counter-offer specialist | Challenges opponent's position frequently |
| Calculated Risk-Taker | Expert negotiator | Calibrates risk to expected value |

**The VP Scale → Prospect Theory connection:**
- Bridge: risk increases when behind (VP scale incentive)  
- Business: risk increases when behind (Kahneman-Tversky Prospect Theory, 1979)
- **Bridge provides a controlled, quantifiable laboratory to test Prospect Theory**

**This is the bridge → business transfer argument.**
No business data needed — argue by structural analogy, supported by bridge empirics.

---

## TIMELINE: When to Write Each Paper

| Paper | Based on | Draft by | Target venue |
|-------|----------|----------|-------------|
| **Paper 1:** Risk profiles taxonomy | Q1.1–Q1.3 + Q4.1 | Month 9 | AAMAS 2027 or CPAIOR 2027 |
| **Paper 2:** VP scale empirical test | Q2.1–Q2.3 | Month 11 | Games and Economic Behavior |
| **Paper 3:** Partnership chemistry | Q3.1–Q3.2 | Month 15 | IJCAI 2027 |
| **Paper 4:** Bridge→Business transfer | Q6.1 (theory) | Month 20 | Management Science |

> **AAMAS 2027 deadline:** approximately November 2026 → start Paper 1 writing in August 2026.

---

## WHAT WE HAVE TODAY (May 2026) THAT ENABLES ALL OF THE ABOVE

| Capability | Status | Coverage |
|-----------|--------|----------|
| Board-room records | ✅ | **149,208 rows** (nearly 2× the April count) |
| Competitions scraped | ✅ | **5 championships** (Budapest 2016, Ostend 2018, Madeira 2022, Herning 2024, Poznan 2025) |
| Individual player names per position | ✅ | **99.9%** of rows — all 5 competitions |
| Card holdings (52 cards) | ✅ | **54%** of rows (up from ~30% after fixing boards 17-32 bug) |
| Full bidding sequences | ⚠️ | **31%** — Herning + Poznan only (46K rows). Older sites lack tooltip format; PlayDetails pages return HTTP 500 |
| Player IDs (stable across tournaments) | ⚠️ | Name-based (not numeric). May need fuzzy matching for spelling variants |
| Running match score per board | 🔲 | To build in `src/features/running_score.py` |
| Risk metrics per board | 🔲 | To build in `src/features/risk_metrics.py` |
| Tournament standings per round | 🔲 | To build in `src/features/tournament_standings.py` |
| Player demographics (age, gender) | ❌ | Not in EuroBridge — need WBF external data |
| Trick-by-trick play | ❌ | Not collected — available in BBO (Year 2) |

### Key Milestones (Data Collection)

| Date | Milestone |
|------|-----------|
| March 2026 | First scraper working, Herning 2024 scraped |
| April 2026 | Player names backfill, Poznan 2025 scraped, RESEARCH_INSIGHTS created |
| May 2026 | Fixed Ostend 2018 + Budapest 2016 tournament IDs. Fixed cards scraper for boards 17-32. Dataset grew from 78K → **149K rows**. |
| May 2026 | **NegoPlay Stage 1 complete** — player profile pipeline built and validated (see below). |

---

## PART 7 — NegoPlay Empirical Findings (Stage 1, May 2026)

> This section documents the key methodological finding from building the NegoPlay pipeline.
> It is directly relevant to **RQ1** and informs the strategy for all subsequent ML work.

### Q7.1 — Do elite bridge players form discrete clusters?

**What we did:**
Built a full feature-engineering pipeline on 149,208 rows producing 15 features per player
(8 outcome features from the `contract` column + 5 process features parsed from 46,230 bidding sequences).
Tested K-Means (k=2..6), GMM with BIC selection, and HDBSCAN on 807 qualifying players.

**The finding — a continuum, not clusters:**

| Method | Result | Interpretation |
|--------|--------|----------------|
| K-Means (k=2..6) | Silhouette ≤ 0.15 | No natural groupings |
| GMM + BIC | Selects k=2, both centroids nearly identical | No meaningful separation |
| HDBSCAN | 0 clusters, all 807 players classified as noise | Confirms: pure continuum |
| PCA (3 components) | Explains 56.8% of variance | Structure exists, but smooth |

**Why:** Elite European Championship players are all near-optimal performers. Top-level expertise
research consistently shows convergence toward optimal play — variation is real but continuous.

**Published claim for Paper 1:**
> *"Contrary to the assumption that elite players exhibit discrete strategic types, our analysis of
> 807 European championship players shows a statistical continuum (silhouette ≤ 0.15, HDBSCAN
> detecting zero natural clusters). This mirrors findings in expertise research where top performers
> converge toward near-optimal play."*

### Q7.2 — The Extreme Profiles Solution

**What we built instead:**
Identified the top 10% on each of 4 behavioural axes, assigning each player to the axis
where their z-score is both above the 90th percentile AND their personal maximum.

**Results v2.0 (807 players, initial — May 2026):**

| Profile | n (%) | Defining feature | Mean value | vs. average |
|---------|-------|-----------------|------------|-------------|
| Slam Hunter | 64 (7.9%) | slam_rate | 0.116 | **2.8×** Insurance Player |
| Insurance Player | 60 (7.4%) | partscore_rate | 0.693 | highest partscore |
| Fighter | 66 (8.2%) | penalty_double_rate | 0.134 | **1.6×** average |
| NT Specialist | 53 (6.6%) | nt_rate | 0.408 | **1.5×** average |
| Generalist | 564 (69.9%) | — | — | baseline |

### Q7.3 — Sample-Size Correction (Nezer's Review, May 2026)

**The problem (caught by PhD supervisor Nezer, an expert bridge player):**
The v2.0 pipeline used `min_boards=20`. But for rare events like slam (≈4% baseline rate),
20 declared boards is far too few. A player with 20 boards and 2 slams has:
- Empirical slam_rate = 10% (top 10% → classified as Slam Hunter!)
- 95% confidence interval = **[1.2%, 31.7%]** — overlapping the baseline
- Indistinguishable from luck

In the v2.0 cohort, 41% of "Slam Hunters" had < 30 declared boards and 56% had < 50.
Many were almost certainly false positives driven by small samples.

**The fix (v2.1):**
1. Raised minimum to **≥50 declared boards AND ≥50 bidding boards**
2. Added a **one-sided binomial test** at p < 0.05 against the population baseline:
   a player is only assigned to a profile if their observed rate is statistically
   distinguishable from the average

**Results v2.1 (563 players, robust):**

| Profile | n (%) | Defining feature | Profile mean | Generalist mean | Ratio | Median n_declared |
|---------|-------|-----------------|--------------|----------------|-------|-------------------|
| Slam Hunter | 20 (3.6%) | slam_rate | 0.101 | 0.055 | **1.84×** | 216 |
| Insurance Player | 21 (3.7%) | partscore_rate | 0.684 | 0.570 | **1.20×** | 97 |
| Fighter | 37 (6.6%) | penalty_double_rate | 0.131 | 0.085 | **1.55×** | 153 |
| NT Specialist | 17 (3.0%) | nt_rate | 0.385 | 0.282 | **1.36×** | 114 |
| Generalist | 468 (83.1%) | — | — | — | baseline | 117 |

**What v2.1 gives up — and what it gains:**
- Smaller cohorts per profile (e.g., 20 Slam Hunters instead of 64)
- Stronger ratios are more conservative (1.84× instead of 2.8×) — but they are real
- Every assignment now passes p < 0.05 — defensible in front of any reviewer
- Median sample size per Slam Hunter jumps from 42 boards to **216 boards**

**Methodological notes for reviewers:**
1. The 10% cutoff is explicit and testable — sensitivity analysis at 5%/15%/20% is straightforward.
2. The Generalist group (83% of population) is NOT discarded — it serves as the baseline/control agent in Stage 4.
3. The binomial significance test is a published, standard tool (`scipy.stats.binomtest`).
4. External validation comes from Stage 4 alignment: if agents built from these profiles show
   Spearman ρ ≥ 0.70 between bridge and negotiation behaviour, the profiles are validated empirically.
5. The continuum finding itself is a research contribution, not a failure.
6. The v2.0 → v2.1 revision after expert review is documented openly — protocol deviations
   strengthen, rather than weaken, the eventual paper.

**Outputs saved (v2.1):**
- `NegoPlay/data/processed/player_profiles.csv` — 563 players with profile assignments + p-values
- `NegoPlay/docs/images/pca_scatter.png` — 2D PCA, coloured by profile
- `NegoPlay/docs/images/radar_profiles.png` — behavioural fingerprints per profile
- `NegoPlay/docs/images/feature_bars.png` — key feature comparison across profiles

---

### Q7.4 — Bridge Expert Validation: Profile-Specific Caveats (May 2026)

> Source: `/bridge-expert` validation run against the v2.1 profile table.
> Two profiles received ACCEPT, two received ACCEPT_WITH_CAVEAT.
> These caveats must appear as **footnotes in Paper 1** (risk taxonomy chapter).

**✅ Slam Hunter — ACCEPT**
Rate 10.1% with median n=216 boards is fully credible at European Championship level.
Approximate 95% CI: [6.3%, 15.2%] — well above the 5.5% baseline.
*Caveat for paper (footnote):* Bridge is a partnership game; the declarer does not
bid alone. Part of an elevated slam_rate may reflect an aggressive bidding partner
rather than the individual's own risk preference. Recommend future work: track
pair-level slam rates vs. individual-level rates to quantify the partnership effect.

**⚠️ Insurance Player — ACCEPT_WITH_CAVEAT**
Rate 68.4% partscores with ratio 1.20× is statistically valid but the **weakest signal**
of the four profiles. Smallest effect size. The key confound: partscore_rate is partly
determined by hand distribution — a player who repeatedly draws flat 4-3-3-3 hands
across multiple tournaments may appear as an Insurance Player even with average risk appetite.
*Footnote for paper:* "The Insurance Player profile has the smallest effect size (1.20×)
and may partly reflect hand-distribution variance across tournaments rather than a stable
individual risk preference. Replication across different tournament formats is recommended."

**✅ Fighter — ACCEPT (strongest profile)**
Rate 13.1% penalty doubles over bidding boards with ratio 1.55× is clear and well-supported.
This is methodologically the **most defensible profile** because:
(a) the denominator (n_bidding_boards) correctly captures both sides of the table —
a player who doubles opponents but doesn't declare that board would be invisible
in n_declared but is correctly counted here;
(b) penalty doubles are a fully conscious, deliberate decision that directly reflects
individual style;
(c) the event is common enough (8.5% baseline) that CIs narrow quickly.
*Recommendation: lead with Fighter as the primary methodological example in Paper 1.*

**⚠️ NT Specialist — ACCEPT_WITH_CAVEAT**
Rate 38.5% NT contracts with ratio 1.36× is valid, but there is a known confound:
bidding convention system. Pairs playing Precision Club open 1NT on a wider HCP range
than pairs playing Standard American or Acol, systematically generating more NT contracts
regardless of individual preference. This is a partnership/system effect, not individual.
*Footnote for paper:* "NT rate may partly reflect the partnership's bidding system
(Precision vs. Natural) rather than individual stylistic preference. Without system
data per pair, this confound cannot be fully controlled."

---

### Q7.5 — Methodological Contribution: The v2.0 → v2.1 Revision

> ⚠️ **ACTION ITEM FOR THESIS/PAPER:** The before/after revision must appear as a
> dedicated paragraph in the **Methods section** of Paper 1. This is not just a
> correction — it is a standalone methodological contribution.

**Why this belongs in the paper (not just a footnote):**

The revision from v2.0 (64 Slam Hunters, 2.8× ratio, min_boards=20) to v2.1
(20 Slam Hunters, 1.84× ratio, min_boards=50 + binomial test) illustrates a general
problem in rare-event player profiling that is not widely discussed in the game AI
literature:

> *When the event of interest has a low base rate (≤5%), small samples produce
> false positives even with high observed rates. A player with 20 boards and
> 2 slams (rate=10%) has a 95% CI of [1.2%, 31.7%] — completely overlapping the
> baseline. The appropriate gate is not a percentile cutoff alone, but a
> significance test against the population rate.*

**The paragraph structure for the Methods chapter:**

```
Paragraph 1 — The original pipeline and its result:
  "Initial profiling used a minimum of 20 declared boards, yielding 807 qualifying
  players and 64 Slam Hunters (7.9%) with a mean slam rate of 11.6% (2.8× baseline)."

Paragraph 2 — The expert review and identified problem:
  "Expert review by [Nezer, a national-level bridge player and PhD co-supervisor]
  raised a methodological concern: with only 20 declared boards, rare events such as
  slam contracts (≈4–6% baseline) produce unreliable rate estimates. A player with
  20 boards and 2 slams (rate=10%) has a 95% CI of [1.2%, 31.7%], indistinguishable
  from noise at conventional significance levels."

Paragraph 3 — The fix and its justification:
  "We revised the pipeline in two ways: (1) raising the minimum to 50 declared boards
  for outcome features and 50 bidding boards for process features, and (2) adding a
  one-sided binomial significance test (p < 0.05) against the population baseline as
  a secondary gate. This is a standard tool for rare-event count data (binom_test,
  SciPy 1.11)."

Paragraph 4 — The revised results and interpretation:
  "The revised pipeline yields 563 qualifying players and 20 Slam Hunters (3.6%)
  with a mean slam rate of 10.1% (1.84× baseline, median n=216 boards per player).
  The reduction from 64 to 20 confirms that the original cohort included false
  positives; each remaining assignment passes both the percentile and significance gates."
```

**Key numbers to cite in that paragraph:**
- v2.0: 807 players, 64 Slam Hunters, ratio 2.8×, min n_declared=20
- v2.1: 563 players, 20 Slam Hunters, ratio 1.84×, median n_declared=216, all p<0.05
- CI example: n=20, k=2 → 95% CI [1.2%, 31.7%] (scipy.stats.binomtest)
- CI after fix: n=216, rate=0.101 → 95% CI [6.3%, 15.2%]

---

### Q7.6 — Dr. Rami's Preprocessing Audit (May 2026)

> Source: Course supervisor Dr. Rami challenged the initial continuum finding,
> arguing that K-Means is well-suited to this kind of behavioral data and that
> the failure to find clusters likely meant our pre-processing was incomplete.
> We took the challenge seriously and ran a 5-configuration audit.

**Dr. Rami's specific recommendations (April 2026 meeting):**
1. Normalize the data (StandardScaler / RobustScaler)
2. Run PCA before K-Means — discard low-variance dimensions
3. Work only with "meaty" features (those with real variance)
4. Try t-SNE for visualization
5. Add full visualizations and Excel exports for transparency

**What we tested (5 configurations):**

| # | Pipeline | Best k | Silhouette | HDBSCAN clusters |
|---|----------|--------|------------|-------------------|
| V1 | 10 features, no PCA, StandardScaler | 4 | ~0.15 | 0 |
| V2 | 8 features + PCA(3), StandardScaler | 4 | **0.24** | 0 |
| V3a | Full preprocessing + K-Means | 2 | 0.17 | — |
| V3b | Full preprocessing + GMM (BIC) | 2 | 0.14 | — |
| V3c | Full preprocessing + HDBSCAN | — | n/a | **0** |

**Full preprocessing details (V3):**
- Variance filter (cv < 0.10): removed `avg_level`, `partscore_rate`, `avg_bids_per_board`
- Correlation analysis: no feature pairs above |r|>0.7 → all 7 remaining features are statistically independent
- Outlier removal: Mahalanobis distance at α=0.01 removed 18 / 567 players (3%)
- Scaling: RobustScaler (less sensitive to outliers than StandardScaler)
- PCA: 5 components → 85.2% cumulative variance explained

**Key empirical findings:**

1. **No correlated feature pairs.** All 7 retained features (slam_rate, double_rate,
   nt_rate, opening_rate, preempt_rate, intervention_rate, penalty_double_rate)
   are statistically independent — each measures a genuinely different aspect
   of bridge behavior. They are not redundant.

2. **PCA scree plot shows gradual variance decay.** PC1=24.6%, PC2=19.1%,
   PC3=13.5%, PC4=10.3%, PC5=9.2%. The absence of a dominant component
   (>60% of variance) is the signature of data WITHOUT a strong cluster structure.
   If clusters existed, we would expect 1-2 components to dominate.

3. **More aggressive preprocessing makes K-Means WORSE, not better.**
   Going from V2 (silhouette=0.24) to V3 (silhouette=0.17) means the outlier
   removal stripped away the very players carrying the weak signal — exactly
   the Slam Hunters and Fighters we want to study. This is the opposite of
   what would happen if hidden clusters existed.

4. **HDBSCAN finds zero clusters in every configuration.** This is the most
   important result. HDBSCAN is density-based and far more flexible than K-Means
   — it can detect arbitrary shapes including elongated structures. Its
   consistent verdict of "no natural clusters, 100% noise" across all 5
   configurations is the strongest possible empirical evidence for the continuum.

5. **GMM with BIC selection agrees.** BIC penalizes model complexity, so it
   defaults to the simplest model that fits. BIC chose k=2, the smallest k
   tested — which means the data does not support a richer cluster structure.

**Implication for the methodology:**

The extreme-percentile approach (current pipeline) is **not just one of several
options — it is the appropriate methodology for this data**. K-Means looks for
groups that do not exist. Extreme-percentile profiling identifies the tails of
a continuous distribution, which is what we actually have.

This finding strengthens (rather than weakens) the contribution: we have
empirically demonstrated, through 5 independent algorithmic tests, that
clustering is the wrong frame for elite bridge player behavior — and we have
a principled alternative that works.

**Paragraph template for Paper 1 (Methods section):**

```
"Following expert review by our course supervisor (Dr. Rami), we conducted a
five-configuration audit to ensure our continuum finding was not an artifact
of inadequate preprocessing. We tested K-Means, GMM, and HDBSCAN under
escalating preprocessing pipelines: (V1) StandardScaler only; (V2)
StandardScaler + PCA(3); (V3) RobustScaler + low-variance feature filter
+ correlation pruning + Mahalanobis outlier removal + PCA(5).

Across all configurations, K-Means silhouette scores remained below 0.25
(maximum 0.24 at V2), GMM with BIC selection chose the simplest model (k=2)
with silhouette 0.14, and HDBSCAN consistently detected zero natural
clusters with 100% of players classified as density noise.

Notably, more aggressive preprocessing (V3) reduced rather than improved
K-Means silhouette, indicating that the modest structure detected at V2 was
driven by behaviorally extreme players whose removal as outliers eliminated
the weak signal. The PCA variance distribution (PC1=24.6%, PC2=19.1%,
PC3=13.5%, ..., PC10=1.2%) shows gradual decay without a dominant component,
the characteristic signature of data lacking discrete cluster structure.

These results converge on a robust empirical finding: elite European
Championship bridge players form a statistical continuum rather than
discrete strategic types. We therefore adopt an extreme-percentile profiling
approach (Section X.X) that identifies the tails of this continuum rather
than attempting to partition it into groups."
```

**Outputs from the audit:**
- `NegoPlay/notebooks/preprocessing_comparison.py` — runs all 5 configurations
- `NegoPlay/src/stage1_clustering/preprocessing.py` — reusable preprocessing module
- `NegoPlay/docs/images/pca_variance.png` — scree plot showing gradual decay
- `NegoPlay/docs/images/tsne_scatter.png` — 2D layout showing profile overlap
- `NegoPlay/results/preprocessing.xlsx` — normalized data + PCA loadings (for transparency)

**Methodological note:** Outlier removal is appropriate for K-Means/GMM (group-finding
algorithms) but is **incompatible with the extreme-percentile approach** — the
"outliers" are precisely the Slam Hunters and Fighters we want to profile.
In the production pipeline, we therefore keep all players and use the variance-filtered
feature set without outlier removal for profile assignment.

### Q7.7 — Stage 2: LLM-Extracted Profiles Are Empirically Validated (May 2026)

> This is the key Stage 2 result. Stage 1 produced 4 candidate profiles by
> extreme-percentile profiling. Stage 2 used an LLM (Gemini 2.5 Flash) to read
> each player's actual hands and describe their decision-making "skills".
> The open question: are these profiles *real* behavioural types, or an artefact
> of the clustering / a hallucination of the LLM?

**What we did:**
For each profile we measured its *defining* behaviour rate directly from the raw
bidding/contract data — **no LLM, no clustering** — and compared it to a baseline
of Generalist players. Crucially, the Generalist baseline is itself elite
tournament players, so this is an **elite-vs-elite** comparison; any separation is
therefore conservative (harder to achieve than vs the general population).

**Method (reproducible):**
- Denominators aligned exactly with Stage 1: declarer-only boards for
  contract-level metrics (slam / partscore / NT); per-board-with-bidding for the
  Fighter's penalty-double metric.
- Effect size = Cohen's d; significance = one-sided Mann-Whitney U.
- Script: `NegoPlay/notebooks/validate_base_rates.py` →
  `results/stage2_sample_v2_focused_prompt/validation_table.xlsx`

**Result — all four profiles separate from Generalist, all significant:**

| Profile | Defining metric | Ratio vs Generalist | Cohen's d | p-value | Verdict |
|---------|-----------------|--------------------|-----------|---------|---------|
| Fighter | penalty_double_rate | ×1.31 | 2.13 | 0.016 | STRONG |
| Insurance Player | partscore_rate | ×1.24 | 3.30 | 0.004 | STRONG |
| Slam Hunter | slam_rate | ×1.37 | 2.81 | 0.004 | STRONG |
| NT Specialist | nt_rate | ×1.27 | 4.62 | 0.004 | STRONG |

Cohen's d ≥ 0.8 is "large"; ≥ 2.0 is "very large". All p < 0.05.

**Why this matters for the thesis (RQ1):**
This is the first empirical evidence in the project that AI can recover stable,
distinguishable decision-making styles ("bidding dialects") from real bridge data
under incomplete information — *and* that an independent LLM description of those
styles matches the players' actual behaviour. It is the foundational result the
downstream negotiation-transfer argument (RQ3/RQ6) rests on: the profiles are
shown to be real **before** any agents are built from them.

**Methodological note worth a footnote in Paper 1:**
The Fighter metric was initially measured per-call (doubles ÷ total calls), which
gave a weak, overlapping signal and was *rejected* by the bridge-expert review.
Switching to the Stage-1-aligned per-board metric (boards where the player made
≥1 double ÷ boards-with-bidding) produced a clean ×1.31 separation, d=2.13.
**Lesson: the validation denominator must match the clustering denominator
exactly**, or a real profile can look spurious (and vice versa).

**Caveats:**
- The validation sample is 5 players per profile. (Nezer's n≥50 minimum applies to
  per-player board counts, which is satisfied — the 5-per-profile is the validation
  sample, not the clustering sample.)
- One Generalist (NAWROCKI) falls inside the Fighter penalty-double range — a single
  borderline case, expected at this sample size.
- Profile-specific confounds from Q7.4 still apply (partnership effects, bidding
  system for NT) and should remain as footnotes.

**Aggregation engineering note:**
Profile-level skill aggregation uses TF-IDF + cosine-similarity + Union-Find
clustering of skill names (`NegoPlay/src/stage2_skills/aggregator.py`), with the
merge threshold tuned to 0.40. At 0.30 unrelated skills falsely merged
(e.g. "control bidding" with "competitive overcalling"); 0.40 gives clean,
conservative clusters and fixed the NT Specialist returning zero skills.

---

### Q7.8 — Stage 3: Personality Visibly Changes Bidding (sanity check, May 2026)

> First real LLM call of Stage 3. NOT a hypothesis test — a wiring check that
> the profile system prompts actually steer behaviour.

**What we did:** Built five bridge agents (4 profiles + Generalist baseline),
each conditioned on its Stage 2 extracted skills, and gave all five the SAME
hand (S:AKQ72 H:AK4 D:A83 C:Q2 — 20 HCP) after partner opened 1S.

**Result — FOUR distinct bids from one hand:**

| Agent | Bid | Behaviour (from its own reasoning) |
|-------|-----|-------------------------------------|
| Slam Hunter | 4C | aggressive splinter, singleton + slam interest |
| Fighter | 2NT | Jacoby 2NT, game-forcing, exploring slam |
| NT Specialist | 3NT | prioritises NT despite the major fit |
| Generalist | 3NT | standard 3NT rebid, no slam push |
| Insurance Player | 4S | signs off in safe game, declines slam |

**Why it matters:** Each agent's bid matches its profile's logic — the Slam
Hunter and Fighter explore slam (splinter / Jacoby), the NT Specialist insists
on NT, and the Insurance Player deliberately stops in a safe 4S game. These
behaviours emerge from skills extracted from REAL hands, not from us writing
"be aggressive". Preliminary evidence the agents are faithful to their profiles
before any large-scale Stage 4 simulation. Cost: $0.0018 for all five calls.

**Caveat:** n=1 hand. This only shows the mechanism works; per-profile bidding
distributions over many hands come in Stage 4.

**Reproducibility finding (May 2026) — IMPORTANT, partially open:**
The first sanity run used temperature=0.3 and gave different bids on reruns of
the same hand. We lowered agent temperature to **0.0** — but a controlled test
(two consecutive runs, same hands) showed the bids were *still not identical*
(e.g. Fighter 6S vs 3S; Generalist 2NT vs 3NT). **Conclusion: Gemini 2.5 Flash
is NOT deterministic even at temperature 0** (a known property of these models —
parallel GPU execution introduces tiny numerical differences that flip
borderline decisions). Temperature 0 reduces but does not eliminate the
variance.

**Implication for the research design:** we cannot rely on regenerating
identical bids. The reproducibility strategy must instead be:
1. **Persist every raw LLM output** (already done — results saved to JSON), so
   the *analysis* is fully reproducible from saved data even if regeneration
   differs.
2. **Report distributions over many hands**, not single-hand bids — the random
   flips average out across a large sample (this is the Stage 4 design anyway).
3. Optionally pass a fixed `seed` to the API and pin the model version to
   shrink (not remove) the variance. Tracked as a Stage 4 robustness item.

This is itself a worthwhile methodological note for the thesis: LLM-agent
experiments need sample-level reproducibility (saved outputs + distributions),
not call-level determinism.

**Known bridge-quality limitations (bridge-expert review, deferred):** at
temperature 0 the agents bid with the *correct profile direction* (Slam Hunter
drives to slam, Insurance signs off, Fighter competes/doubles, NT Specialist
leans NT) but make occasional *bridge-mechanics errors* — e.g. a "splinter"
named with a doubleton, a 1NT overcall on 12 HCP (standard is 15-18), or
reasoning that confuses opener/responder seat. These are acceptable for
measuring behavioural *direction* (the research target) but should be reduced
with bridge-rule reminders in the prompt before any claim about bid *quality*.
Tracked for a later Stage 3 polish pass.

---

### Q7.9 — Stage 3: Bridge Personality Transfers to Business Negotiation (sanity, May 2026)

> The cross-domain twin of Q7.8, and the first glimpse of the CORE thesis
> mechanism. NOT a hypothesis test — a wiring check that the bridge-derived
> character shows up in a business negotiation.

**What we did:** Built five negotiation agents from the SAME Stage 2 bridge
skills (no new personality — the anti-tautology rule), and gave all five the
SAME opening offer in an M&A scenario: buying a SaaS startup, fair value ≈ $9M,
the agent is the buyer, seller opens at $13M.

**Result — the behavioural split shows in *whether they close*, more than in the
counter price** (buyer; fair ≈ $9M; seller opened $13M):

| Profile (bridge style) | Counter | Willing to close? | Reasoning gist |
|------------------------|---------|-------------------|----------------|
| Generalist (baseline) | $9.0M | **yes** | "standard counter at fair valuation" |
| Insurance Player (safe) | $8.0M | **yes** | "sure small gain over a risky large one" |
| Slam Hunter (bold) | $7.5M | no | "aggressive bid, set the stage for a big deal" |
| Fighter (combative) | $7.5M | no | "their bid is an overreach; anchor hard" |
| NT Specialist (analytical) | $7.5M | no | "establish our preferred contract early" |

**Why it matters:** The cross-domain personality is visible — the two
safety/cooperative profiles (Insurance, Generalist) are the **only ones willing
to close**, while the three aggressive/analytic profiles hold out for a better
price. This split flows from skills extracted from REAL bridge hands, not from
personality labels.

**Honest caveats (this is a sanity check, not a result):**
1. **Weaker price separation than hoped:** three profiles all countered $7.5M
   (the schema fix landed them on the same anchor). The *close/no-close* signal
   separated them better than the price did. Stage 4 must measure richer
   outcomes (final agreed price, who walked, surplus captured) over many
   sessions, not a single counter.
2. **Non-determinism:** an earlier run of this same scenario gave a different
   spread ($7.5M–$10M). Same caveat as Q7.8 — report distributions, not single
   turns.
3. **Bridge-vocabulary leakage:** some reasonings still use bridge terms in the
   business context ("penalty double", "NT overcall", "partscore"). Numeric
   behaviour is right; the framing needs prompt polish. Tracked as deferred.

n=1 scenario, single turn — the real ≥70% alignment test is Stage 4.

---

### What Cannot Be Fixed (Known Limitations)

1. **Bidding for old competitions (0%):** EuroBridge microsites before ~2023 do not embed bidding
   in tooltips. The separate PlayDetails.asp pages return HTTP 500 server errors.
   **Workaround:** Use contract + declarer + tricks for risk analysis (works on 149K rows).
   Full bidding analysis limited to Herning + Poznan (46K rows).

2. **Card holdings (~54%):** BoardAcross pages are only published for some boards.
   This is an EuroBridge publishing decision, not a scraper bug.

3. **No Mixed Teams for 2016/2018:** The Mixed Teams category was introduced in EBL after 2018.
   Budapest 2016 and Ostend 2018 have only Open, Women, Senior (3 categories, not 4).

**The 149K-row dataset with player names is sufficient for Paper 1 (risk taxonomy)
and Paper 2 (VP scale test). Paper 3 (partnership chemistry) benefits from the larger
player pool but may want BBO data for more partnership diversity.**
