# Stage 3 — Bridge Agent Sanity Report

> For bridge-expert review. Each agent is an LLM conditioned on a player profile whose skills were extracted from **real** tournament hands (Stage 2). The question for the reviewer: **are these calls legal and sensible for the stated profile, with no hallucinated bids?**

- **Date:** 2026-05-30
- **Model:** gemini-2.5-flash
- **Total LLM cost:** $0.246711

## Hand 1: Strong balanced-ish, partner opened 1S (slam-interest hand)

- **Hand:** `S:AKQ72  H:AK4  D:A83  C:Q2`  (20 HCP)
- **Auction so far:** `1S Pass`
- **Reviewer note:** 20 HCP with a spade fit — slam exploration is reasonable; a cautious player signs off in 4S.
- **Distinct bids across agents:** 2NT, 3D, 4C, 4NT, 4S

| Profile | Bid | Legal? | Agent's reasoning |
|---------|-----|--------|-------------------|
| Slam Hunter | **4C** | ✅ | This is an aggressive splinter bid, showing a singleton in clubs, strong spade support, and immediate slam interest. |
| Insurance Player | **4S** | ✅ | A strong fit and sufficient points for a safe game, avoiding speculative slam bidding. |
| Fighter | **3D** | ✅ | Strong cue-bid showing first-round diamond control and slam interest with excellent spade support, aggressively pursuing the best contract. |
| NT Specialist | **4NT** | ✅ | With 19 HCP and a balanced hand, a 4NT quantitative bid accurately describes my strength and preference for No Trump, inviting slam. |
| Generalist | **2NT** | ✅ | Jacoby 2NT shows a game-forcing hand with 4+ card support for partner's major, initiating slam exploration. |

## Hand 2: Flat 12-count, opponents opened 1H (defensive decision)

- **Hand:** `S:KJ95  H:Q72  D:KQ4  C:J83`  (12 HCP)
- **Auction so far:** `1H`
- **Reviewer note:** Balanced minimum over 1H — pass / double / 1S overcall are all defensible; a Fighter is more likely to act.
- **Distinct bids across agents:** 1NT, 1S, X

| Profile | Bid | Legal? | Agent's reasoning |
|---------|-----|--------|-------------------|
| Slam Hunter | **1S** | ✅ | Showing a 4-card major and forcing partner to describe their hand further, keeping slam possibilities open. |
| Insurance Player | **1NT** | ✅ | My 12 HCP is borderline for game, and I prefer to make a conservative 1NT response to prioritize partscore safety rather than inviting game. |
| Fighter | **X** | ✅ | Proactively using a takeout double to enter the auction and compete, showing interest in the unbid suits. |
| NT Specialist | **1NT** | ✅ | Aggressively overcalling 1NT to show a balanced hand with stoppers in all suits, consistent with my preference for NT and willingness to stretch on points. |
| Generalist | **1S** | ✅ | With 13 HCP and a 4-card spade suit, bidding 1S is the standard response to partner's 1H opening. |

---

### How to read the *Legal?* column
A ✅ means the bid passed our local Laws-of-Bridge legality check (correct rank over the previous contract, valid double/redouble context). A ❌ means the LLM proposed an illegal call and the system automatically substituted **Pass** — this is the guardrail against hallucinated bids. Zero ❌ rows means every agent produced a legal call.
