# THIS_WEEK.md — Week of April 28, 2026
# Goal: All data Nezer needs for risk-behavior research

## WHAT YOU'RE COLLECTING THIS WEEK AND WHY

Nezer asked: "Who takes risks? When? Why? Does it pay off?"

To answer that you need FOUR things:
1. **Bidding data per board** — what decisions were made (HAVE ✅)
2. **Score situation at each board** — was this team losing/winning when they decided (CAN COMPUTE ✅)
3. **Tournament standing** — were they chasing first place? (CAN COMPUTE ✅)
4. **Player names** — WHO specifically made the risky bid (MISSING ❌ — this week we investigate)

---

## DAY 1 (Sunday) — Let the scrapers run while you do the other tasks

### Morning: Start two scrapers in background (they take 2-3 hours each)

Open TWO terminal windows:

**Terminal 1:**
```
cd "C:\Users\annaf\OneDrive\Desktop\דוקטורט\דוקטורט אנה\דוקטורט - שילוב בינה מלאכותית\collectBridgeData"
python src/downloaders/eurobridge_bulk_scraper.py --competitions EBL_Ostend_2018 --delay 0.8
```

**Terminal 2:**
```
cd "C:\Users\annaf\OneDrive\Desktop\דוקטורט\דוקטורט אנה\דוקטורט - שילוב בינה מלאכותית\collectBridgeData"
python src/downloaders/eurobridge_bulk_scraper.py --competitions EBL_Budapest_2016 --delay 0.8
```

Leave them running. Do the rest of the tasks below in the meantime.

### While scrapers run: Investigate the player name problem

This is the most important detective work of the week.

**Step 1:** Open this URL in your browser:
```
http://db.eurobridge.org/Repository/competitions/24Herning/microsite/Asp/TeamRoster.asp?tournid=2410
```
→ Does it show individual player names? Take a screenshot.

**Step 2:** Try this URL:
```
http://db.eurobridge.org/Repository/competitions/24Herning/microsite/Asp/RoundPairList.asp?tournid=2410&round=1
```
→ Does it show which players played which room in round 1?

**Step 3:** Go to the Herning 2024 website manually:
```
http://db.eurobridge.org/Repository/competitions/24Herning/microsite/
```
Click around and look for any page that shows player names (not just team names).
Look for: "Players", "Roster", "Lineup", "Pairs" in the menu.

**Write your findings in:**
```
data/raw/PLAYER_DATA_INVESTIGATION.md
```

Write: "Found / Did not find player names at URL: [URL]"

---

## DAY 2 (Monday) — Download VP tables + WBF files

### Task A: Download VP scale tables (30 minutes)

Go to: **https://www.ebu.co.uk/regulation-and-conduct/vp-scales**

You'll see a table like what Nezer sent you. Download or manually copy the tables for:
- 8 boards
- 10 boards  
- 12 boards
- 16 boards
- 20 boards

Save each as a CSV file in: `data/raw/vp_tables/`

Format: three columns — `imp_diff, home_vp, visiting_vp`

Example of what vp_scale_12boards.csv should look like:
```
imp_diff,home_vp,visiting_vp
0,10.00,10.00
1,10.36,9.64
2,10.71,9.29
...
25,16.55,3.45
...
```

**Verify:** Check that row for imp_diff=25 shows 16.55 and 3.45 — that's what Nezer used.

### Task B: Download WBF PBN files (1 hour)

Go to: **https://www.worldbridgefed.com/news-media/download-centre/**

Look for "Records of Play" or "Technical Files" section.

Download these files (priority order):
1. Bermuda Bowl 2023
2. Venice Cup 2023
3. World Teams Olympiad 2022

If you can't find them there, try Google:
```
site:worldbridgefed.com bermuda bowl 2023 pbn
```
or:
```
worldbridgefed.com 2023 bermuda bowl records of play filetype:pbn
```

Save to: `data/raw/wbf/`

**After downloading, open one PBN file in Notepad and look for lines like:**
```
[West "John Smith"]
[North "Anna Jones"]
[East "Peter Chen"]
[South "Maria Garcia"]
```
→ If you see player names: ✅ WBF PBN is our source for individual player analysis!
→ Write what you found in `data/raw/PLAYER_DATA_INVESTIGATION.md`

### Task C: Run cards scraper after bulk scrapers finish (evening)

Check if both scrapers from Day 1 finished. Then run:
```
python src/downloaders/eurobridge_cards_scraper.py --competitions EBL_Ostend_2018 EBL_Budapest_2016 --delay 0.8
```

---

## DAY 3 (Tuesday) — Build the Running Score computation

This is the key computation for Nezer's research. No new data needed — it's all derivable from what you have.

**Ask me (Claude) to write this code** — paste this message:
> "Write me src/features/running_score.py that takes the all_matches.parquet file and adds these columns:
> - running_ns_imp: cumulative NS IMP within the match up to this board
> - running_ew_imp: cumulative EW IMP within the match up to this board  
> - ns_leading_by: positive = NS ahead, negative = EW ahead at this point in the match"

Then run it and verify with a real match:
```python
import pandas as pd
df = pd.read_parquet("data/processed/all_matches.parquet")

# Pick one match and check the running scores make sense
match = df[df["match_id"] == df["match_id"].iloc[0]].sort_values("board")
print(match[["board", "ns_score", "ew_score", "running_ns_imp", "ns_leading_by"]])
```

You should see the score accumulating board by board. Check it with the final match IMP totals.

---

## DAY 4 (Wednesday) — Build the Risk Metrics

**Ask me (Claude) to write this code** — paste this message:
> "Write me src/features/risk_metrics.py that takes a row from all_matches.parquet and computes:
> - is_slam_attempt (bool): any bid at 5+ level
> - is_slam_contract (bool): final contract at 6 or 7 level
> - is_preempt_open (bool): opening bid at 2/3/4 level
> - is_double (bool): Dbl in bidding
> - is_sacrifice (bool): heuristic — doubled contract, went down 1-2 tricks
> - risk_score (int 0-10): composite score
> - opening_position (str): N/S/E/W — who opened the bidding"

After it's built, apply it to the full dataset:
```python
df = pd.read_parquet("data/processed/all_matches.parquet")
# apply risk metrics to every row
# save updated file
```

Then do a quick check:
```python
# What % of boards have a slam attempt?
print(df["is_slam_attempt"].mean())  # Should be ~5-8%

# What % have a preempt?
print(df["is_preempt_open"].mean())  # Should be ~15-20%

# Average risk score by competition
print(df.groupby("competition")["risk_score"].mean())
```

---

## DAY 5 (Thursday) — First analysis: Does risk increase when losing?

This answers Nezer's main question directly.

Open a new Jupyter notebook: `notebooks/06_risk_vs_score.ipynb`

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_parquet("data/features/deals_features.parquet")
# (or all_matches.parquet if features not done yet)

# Filter to rows with bidding data
df_bid = df[df["has_bidding"] == True]

# Create "score situation" category
df_bid["score_situation"] = pd.cut(
    df_bid["ns_leading_by"],
    bins=[-float("inf"), -15, -5, 5, 15, float("inf")],
    labels=["losing_badly", "losing", "close", "winning", "winning_badly"]
)

# For NS side: is risk higher when losing?
by_situation = df_bid.groupby("score_situation")["risk_score"].mean()
print(by_situation)

# Plot it
by_situation.plot(kind="bar", title="Average Risk Score by Match Situation")
plt.ylabel("Average Risk Score")
plt.show()
```

**If the bar chart shows higher risk when losing → Nezer's theory is confirmed.**
**If not → that's also interesting! Maybe strong players are rational, weak players aren't.**

Save the chart. Send it to Nezer.

---

## DAY 6 (Friday) — Tournament standing + summary

### Task A: Compute tournament standing per round

```python
# For each competition+category+round, rank teams by VP accumulated so far
# This tells us: is this team in 1st place? Chasing? About to be eliminated?
```

**Ask me (Claude) to write this code.**

### Task B: Write the data coverage report

Run:
```
python explore_data.py
```

Then create `data/DATA_COVERAGE_REPORT.md` with:

```markdown
# Data Coverage Report — [date]

## Total dataset
- Rows: X
- Rows with bidding: X (X%)
- Rows with cards: X (X%)
- Rows with both: X (X%)

## Per competition
| Competition | Rows | Has bidding | Has cards |
| EBL_Herning_2024 | ... | ... | ... |
...

## Player data status
- EuroBridge player names: [FOUND / NOT FOUND]
- WBF PBN player names: [FOUND / NOT FOUND]
- Conclusion: [our plan for player-level analysis]

## Risk metrics
- Boards with slam attempt: X%
- Boards with preempt: X%
- Boards with double: X%
```

### Task C: Email to Nezer

Write a short email (half a page) with:
1. "I computed running match scores from the existing data"
2. "I built risk metrics (slam, preempt, double, sacrifice)"
3. Attach the chart from Thursday: risk score vs match situation
4. "I found / did not find player names in EuroBridge — here's what I found"
5. "For player-level analysis, WBF PBN is our best source — I downloaded X tournaments"

---

## WHAT YOU WILL HAVE AT END OF WEEK

| Data | Status | Used for |
|------|--------|----------|
| EuroBridge 5 competitions (~140-150K rows) | ✅ Done | Everything |
| VP scale tables | ✅ Downloaded | IMP→VP conversion |
| Running IMP per board | ✅ Computed | Nezer: risk vs position |
| Tournament standings per round | ✅ Computed | Nezer: leaders vs chasers |
| Risk metrics per board | ✅ Computed | Nezer: risk profiling |
| Who bid what (position N/S/E/W) | ✅ Parsed from existing data | Nezer: who made risky bid |
| WBF PBN files | ✅ Downloaded | Player names (check!) |
| Player name mapping (EBL) | ❓ Investigate | Nezer: individual players |
| First chart: risk vs score situation | ✅ Built | Send to Nezer! |

---

## THE ONE QUESTION THAT WILL DECIDE YEAR 1

**Do EuroBridge or WBF PBN files give us player names?**

- If YES → we can do individual player analysis with existing data → Nezer is happy
- If NO → we need BBO for player names → that's Year 2 → tell Nezer honestly

**Investigate this on Day 1. Report to Nezer by Friday.**
