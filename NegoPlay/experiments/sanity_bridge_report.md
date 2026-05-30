# Stage 3 — Bridge Agent Sanity Report

> For bridge-expert review. Each agent is an LLM conditioned on a player profile whose skills were extracted from **real** tournament hands (Stage 2). The question for the reviewer: **are these calls legal and sensible for the stated profile, with no hallucinated bids?**

- **Date:** 2026-05-30
- **Model:** gemini-2.5-flash
- **Total LLM cost:** $0.265867

## Hand 1: Strong balanced-ish, partner opened 1S (slam-interest hand)

- **Hand:** `S:AKQ72  H:AK4  D:A83  C:Q2`  (20 HCP)
- **Auction so far:** `1S Pass`
- **Reviewer note:** 20 HCP with a spade fit — slam exploration is reasonable; a cautious player signs off in 4S.
- **Distinct bids across agents:** 2NT, 3NT

| Profile | Bid | Legal? | Agent's reasoning |
|---------|-----|--------|-------------------|
| Slam Hunter | **2NT** | ✅ | Jacoby 2NT shows a game-forcing hand with 4+ card support for partner's spades, initiating slam investigation. |
| Insurance Player | **2NT** | ✅ | Showing game-forcing strength and 4+ card support for partner's spades, initiating a slam investigation. |
| Fighter | **2NT** | ✅ | Jacoby 2NT shows 13+ HCP and 4+ card support for partner's spades, forcing to game and initiating slam exploration with this very strong hand. |
| NT Specialist | **2NT** | ✅ | Strong, game-forcing hand with 5-card support for partner's spades, using Jacoby 2NT to explore slam while keeping NT options open for later. |
| Generalist | **3NT** | ✅ | Strong, balanced hand with a 5-card major, showing game values (20-22 HCP). |

## Hand 2: Flat 12-count, opponents opened 1H (defensive decision)

- **Hand:** `S:KJ95  H:Q72  D:KQ4  C:J83`  (12 HCP)
- **Auction so far:** `1H`
- **Reviewer note:** Balanced minimum over 1H — pass / double / 1S overcall are all defensible; a Fighter is more likely to act.
- **Distinct bids across agents:** 1S, Pass

| Profile | Bid | Legal? | Agent's reasoning |
|---------|-----|--------|-------------------|
| Slam Hunter | **1S** | ✅ | With 12 HCP and a 4-card spade suit, a 1S response shows my strength and keeps the auction alive for slam exploration. |
| Insurance Player | **Pass** | ✅ | My hand is balanced with only 12 HCP and no clear overcall, so passing is the safest option to avoid unnecessary competitive bidding. |
| Fighter | **1S** | ✅ | Making a proactive competitive overcall with my longest suit, prioritizing competitive presence over strict suit length requirements. |
| NT Specialist | **1S** | ✅ | Overcall 1S to show a competitive hand with a 4-card spade suit, allowing for future NT exploration. |
| Generalist | **Pass** | ✅ | My hand is balanced with 12 HCP, which is not enough for a 1NT overcall, nor does it have a 5-card suit for a 1S overcall. A takeout double is not appropriate with 3-card support for their suit and a balanced hand. |

---

### How to read the *Legal?* column
A ✅ means the bid passed our local Laws-of-Bridge legality check (correct rank over the previous contract, valid double/redouble context). A ❌ means the LLM proposed an illegal call and the system automatically substituted **Pass** — this is the guardrail against hallucinated bids. Zero ❌ rows means every agent produced a legal call.
