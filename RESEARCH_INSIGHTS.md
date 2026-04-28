# RESEARCH_INSIGHTS.md — Empirical Questions We Can Now Answer

**Last updated:** April 2026
**Status of data:** ~78K rows, 5 EBL competitions, bidding + cards + **individual player names per position**

This file is a living research notebook. Each section = one empirically testable question,
with: the exact analysis → the expected finding → which paper/chapter it contributes to.

> **Rule:** Every insight here must be answerable from the current dataset.
> If it needs future data (BBO, negotiation transcripts), label it clearly.

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
> "Among 847 elite European players in our dataset, we identified 3 distinct risk profiles:
> Conservative Experts (high success, low risk), Calculated Risk-Takers (high risk, high success),
> and Overextenders (high risk, low success). BRINK Sjoert exemplifies the Calculated Risk-Taker
> profile with a slam attempt rate of X% and success rate of Y%."

---

### Q1.3 — Risk Style Taxonomy (the actual classification)

**Three observable risk behaviors:**

| Label | Definition | How to detect in data |
|-------|-----------|----------------------|
| **Slam Hunter** | Frequently bids 6/7 level | `contract LIKE '6%' OR contract LIKE '7%'` |
| **Preemptor** | Frequently opens at 2/3/4 level | First bid in sequence ≥ 2 level |
| **Doubler** | Frequently doubles opponent's contracts | `Dbl` appears in bidding |
| **Insurance Player** | Stops short of game despite strength | Stops at 4m/3NT when game available |

**Method:**
- Build a feature vector per player: (slam_rate, preempt_rate, double_rate, success_rate)
- Cluster using K-means or HDBSCAN (3–5 clusters)
- Label clusters qualitatively → the taxonomy

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

**This enables:**
- "BRINK Sjoert made 47 slam attempts across the dataset — 38 succeeded (81%)"
- "The most common opener in Women's category is..." 
- "Players who double most frequently: TOP 10..."

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

## WHAT WE HAVE TODAY (April 2026) THAT ENABLES ALL OF THE ABOVE

| Capability | Status |
|-----------|--------|
| 78K board-room records | ✅ |
| Full bidding sequences | ✅ ~60% of rows |
| Card holdings (52 cards) | ✅ ~50% of rows |
| **Individual player names per position** | ✅ **NEW — April 2026** |
| Player IDs (stable across tournaments) | ⚠️ Name-based (not numeric) |
| Running match score per board | 🔲 To build (1 day) |
| Risk metrics per board | 🔲 To build (1 day) |
| Tournament standings per round | 🔲 To build (1 day) |
| Player demographics (age, gender) | ❌ Not in EuroBridge |
| Trick-by-trick play | ❌ Not collected |

**Adding player names was the critical unlock.**
Without names: "SCOTLAND bid 3NT"
With names: "McGOWAN Elizabeth bid 3NT, took the risk, made it."
