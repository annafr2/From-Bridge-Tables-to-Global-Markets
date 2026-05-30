# Stage 3 — Bridge Agent Sanity Report

> For bridge-expert review. Each agent is an LLM conditioned on a player profile whose skills were extracted from **real** tournament hands (Stage 2). The question for the reviewer: **are these calls legal and sensible for the stated profile, with no hallucinated bids?**

- **Date:** 2026-05-30
- **Model:** gemini-2.5-flash
- **Total LLM cost:** $0.252370

## Hand 1: Strong balanced-ish, partner opened 1S (slam-interest hand)

- **Hand:** `S:AKQ72  H:AK4  D:A83  C:Q2`  (20 HCP)
- **Auction so far:** `1S Pass`
- **Reviewer note:** 20 HCP with a spade fit — slam exploration is reasonable; a cautious player signs off in 4S.
- **Distinct bids across agents:** 2H, 3NT, 3S

| Profile | Bid | Legal? | Agent's reasoning |
|---------|-----|--------|-------------------|
| Slam Hunter | **2H** | ✅ | Game-forcing bid showing a strong hand with spade support and initiating slam exploration. |
| Insurance Player | **3S** | ✅ | A strong game-forcing raise to ensure a safe game contract in spades, allowing partner to place the final contract. |
| Fighter | **3S** | ✅ | Strong raise with slam interest, allowing partner to describe their hand further for slam exploration and maximizing our score. |
| NT Specialist | **3NT** | ✅ | With a balanced 19 HCP hand and stoppers in all suits, I prioritize a direct 3NT game over exploring a major suit fit. |
| Generalist | **3NT** | ✅ | A 3NT rebid after opening 1S shows a balanced hand with 20-22 HCP. |

## Hand 2: Flat 12-count, opponents opened 1H (defensive decision)

- **Hand:** `S:KJ95  H:Q72  D:KQ4  C:J83`  (12 HCP)
- **Auction so far:** `1H`
- **Reviewer note:** Balanced minimum over 1H — pass / double / 1S overcall are all defensible; a Fighter is more likely to act.
- **Distinct bids across agents:** 1NT, 2H, 3NT, X

| Profile | Bid | Legal? | Agent's reasoning |
|---------|-----|--------|-------------------|
| Slam Hunter | **2H** | ✅ | Showing 10-12 HCP and 3-card support for partner's hearts, inviting game and keeping slam possibilities open. |
| Insurance Player | **2H** | ✅ | Opting for a safe partscore in partner's suit, signaling minimum support to avoid pushing for game. |
| Fighter | **X** | ✅ | A proactive takeout double with good shape and borderline opening points to compete aggressively. |
| NT Specialist | **1NT** | ✅ | Aggressively overcalling 1NT with a balanced 14-count and stoppers in all suits, prioritizing No Trump. |
| Generalist | **3NT** | ✅ | With 13 HCP and a balanced hand, 3NT is the standard response to show 13-15 HCP and deny 4-card support for partner's major. |

---

### How to read the *Legal?* column
A ✅ means the bid passed our local Laws-of-Bridge legality check (correct rank over the previous contract, valid double/redouble context). A ❌ means the LLM proposed an illegal call and the system automatically substituted **Pass** — this is the guardrail against hallucinated bids. Zero ❌ rows means every agent produced a legal call.
