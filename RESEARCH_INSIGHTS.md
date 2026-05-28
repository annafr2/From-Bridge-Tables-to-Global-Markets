# RESEARCH_INSIGHTS.md — Empirical Questions We Can Now Answer

**Last updated:** May 2026
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
