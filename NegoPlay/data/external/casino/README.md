# CaSiNo — Campsite Negotiation Dialogues

**Downloaded:** 2026-07-02 from the authors' official repository
(`github.com/kushalchawla/CaSiNo`, file `data/casino.json`, 4.3 MB).
Also mirrored on Hugging Face as `kchawla123/casino`.

**License:** CC-BY-4.0 (per the Hugging Face dataset card).

**Citation:**
> Chawla, K., Ramirez, J., Clever, R., Lucas, G., May, J., Gratch, J.:
> CaSiNo: A Corpus of Campsite Negotiation Dialogues for Automatic
> Negotiation Systems. In: Proc. NAACL 2021.

## What it contains

- **1,030 human-human negotiation dialogues** (English, crowdsourced,
  846 US participants). Two campsite neighbors negotiate a package of
  **three issues** — food, water, firewood — with private, asymmetric
  priorities (`value2issue`) and stated reasons (`value2reason`).
- **Per-participant trait annotations** (`participant_info`):
  - `personality.big-five` — extraversion, agreeableness,
    conscientiousness, emotional stability, openness (1–7 scale)
  - `personality.svo` — Social Value Orientation (prosocial / proself)
  - `demographics`
  - `outcomes` — points scored, satisfaction, opponent likeness
- **Strategy annotations** on a 396-dialogue subset (4,615 utterances,
  three expert annotators): e.g. self-need, other-need, small-talk,
  no-need, elicit-pref.

## Why NegoPlay needs it (Year-2 plan)

This is the missing **trait→behavior** linkage dataset: it ties the
SAME person's measured decision-making traits (Big-Five, SVO) to their
observed multi-issue negotiation behavior. Three planned uses:

1. **Human-side validation** of the style-transfer hypothesis: does
   trait-level aggression/prosociality predict demand depth in humans
   the way bridge-derived profiles predict it in agents?
2. **Multi-issue scenarios** (integrative, not just price) to replace
   the single-issue MVP scenarios — answering the "multi-criteria
   negotiation" extension suggested by Prof. Hämäläinen.
3. **Counterpart calibration** richer than CraigslistBargain
   (which is distributive/price-only).

## Provenance note

`casino.json` is stored here verbatim (read-only, like all
`data/external/` inputs). Any processing outputs belong in
`data/processed/` or `results/`.
