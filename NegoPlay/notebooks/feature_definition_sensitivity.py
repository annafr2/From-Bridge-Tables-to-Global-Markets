"""
notebooks/feature_definition_sensitivity.py
===========================================
Sensitivity of the Stage-1 profiles and of the style-alignment result to the
two level-based proxy features (reviewer point 3):

  * partscore_rate counts every contract at level 1-3, so 3NT (a game) is
    counted as a low stop and 4C/4D (true partscores) are not.
  * preempt_rate counts every opening at level 2+, so the strong 2C and 2NT
    openings are counted as preempts.

Here both are recomputed with their strict bridge meaning:
  * partscore_strict: level <= 2, or level 3 in a suit (not 3NT), or 4C/4D.
  * preempt_strict:   opening at level >= 2 excluding 2C and 2NT.

The historical columns stay in the software; this script reports what changes
if the strict definitions are used instead: profile membership (Insurance
Player is defined on partscore_rate) and the bridge-aggression composite of
the style analysis (which uses preempt_rate). Negotiation-side values are
unchanged, so the Spearman correlation is recomputed on the bridge side only.

Deterministic, no LLM, no cost. Output: console + results/stage1_definition_sensitivity.csv
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

from notebooks.style_alignment import AGGR_FEATURES, negotiation_aggression
from src.stage1_clustering.bidding_parser import player_bidding_features
from src.stage1_clustering.extreme_profiles import assign_extreme_profiles
from src.stage1_clustering.features import (
    POSITION_MAP,
    _get_declarer_name,
    _is_nt,
    _parse_contract_level,
    compute_player_features,
)

logging.basicConfig(level=logging.WARNING)

ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT.parent / "data" / "processed" / "all_matches_full.csv"
DATA_PARQUET = ROOT.parent / "data" / "processed" / "all_matches.parquet"
STAGE2_JSON = ROOT / "results" / "stage2_sample_v2_focused_prompt" / "skill_profiles_reagg_t040.json"
OUT = ROOT / "results" / "stage1_definition_sensitivity.csv"

STRONG_OPENINGS = {"2C", "2NT"}


def load() -> pd.DataFrame:
    try:
        from src.shared.data_loader import load_matches
        return load_matches(str(DATA_CSV))
    except Exception:  # noqa: BLE001
        return pd.read_parquet(DATA_PARQUET)


def _strain(contract: str) -> str | None:
    """Suit letter of a contract string such as '4S', '3NT', '2Hx' -> 'S','NT','H'."""
    if not isinstance(contract, str) or len(contract) < 2:
        return None
    body = contract.strip().upper()[1:]
    return "NT" if body.startswith("NT") else body[:1]


def is_partscore_strict(contract: str) -> bool:
    level = _parse_contract_level(contract)
    if level is None:
        return False
    if level <= 2:
        return True
    if level == 3:
        return not _is_nt(contract)
    if level == 4:
        return _strain(contract) in ("C", "D")
    return False


def strict_partscore_rates(df: pd.DataFrame) -> pd.Series:
    work = df.copy()
    work["_level"] = work["contract"].apply(_parse_contract_level)
    work["_ps_strict"] = work["contract"].apply(is_partscore_strict)
    work["_declarer_name"] = work.apply(_get_declarer_name, axis=1)
    declared = work.dropna(subset=["_declarer_name", "_level"])
    return declared.groupby("_declarer_name")["_ps_strict"].mean().rename("partscore_strict_rate")


def strict_preempt_rates(df: pd.DataFrame) -> pd.Series:
    bid_df = df[df["has_bidding"] == True]  # noqa: E712
    rows = []
    for tup in bid_df.itertuples(index=False):
        bidding, room = getattr(tup, "bidding", None), getattr(tup, "room", None)
        if not isinstance(bidding, str) or not isinstance(room, str):
            continue
        prefix = "open" if room.strip().lower() == "open" else "closed"
        for pos in ("N", "S", "E", "W"):
            name = getattr(tup, f"{prefix}_{POSITION_MAP[pos]}", None)
            if not isinstance(name, str) or not name.strip():
                continue
            f = player_bidding_features(bidding, pos)
            strict = bool(f["is_preempt"]) and str(f.get("opening_bid")).upper() not in STRONG_OPENINGS
            rows.append((name.strip(), f["is_preempt"], strict))
    long_df = pd.DataFrame(rows, columns=["player_name", "is_preempt", "is_preempt_strict"])
    g = long_df.groupby("player_name")
    return pd.DataFrame({"preempt_rate_check": g["is_preempt"].mean(),
                         "preempt_strict_rate": g["is_preempt_strict"].mean()})


def composite(pf: pd.DataFrame, features: list[str]) -> pd.Series:
    """Bridge aggression per profile: mean z (across profiles) of the given features."""
    means = pf.groupby("profile")[features].mean()
    z = (means - means.mean()) / means.std(ddof=0)
    return z.mean(axis=1)


def main() -> None:
    df = load()
    feats = compute_player_features(df, 50, 50)
    base = assign_extreme_profiles(feats)
    print("baseline profiles:", base["profile"].value_counts().to_dict())

    ps_strict = strict_partscore_rates(df)
    pe_strict = strict_preempt_rates(df)
    feats2 = feats.merge(ps_strict, left_on="player_name", right_index=True, how="left")
    feats2 = feats2.merge(pe_strict, left_on="player_name", right_index=True, how="left")
    chk = (feats2["preempt_rate"] - feats2["preempt_rate_check"]).abs().max()
    print(f"re-implementation check: max |preempt_rate - recomputed| = {chk:.4f}")
    print(f"share of 3NT inside the historical partscore proxy, over players: "
          f"{(feats2['partscore_rate'] - feats2['partscore_strict_rate']).mean():.3f} (mean difference in rate)")

    # (i) profiles under the strict partscore definition
    strict_feats = feats2.copy()
    strict_feats["partscore_rate"] = strict_feats["partscore_strict_rate"]
    strict = assign_extreme_profiles(strict_feats)
    print("strict-partscore profiles:", strict["profile"].value_counts().to_dict())
    m = base[["player_name", "profile"]].merge(strict[["player_name", "profile"]],
                                               on="player_name", suffixes=("_base", "_strict"))
    rows = []
    for prof in ["Slam Hunter", "Insurance Player", "Fighter", "NT Specialist", "Generalist"]:
        a = set(m.loc[m["profile_base"] == prof, "player_name"])
        b = set(m.loc[m["profile_strict"] == prof, "player_name"])
        jacc = len(a & b) / len(a | b) if (a | b) else float("nan")
        rows.append(dict(profile=prof, n_base=len(a), n_strict=len(b), n_common=len(a & b), jaccard=jacc))
        print(f"  {prof:18s} base={len(a):3d} strict={len(b):3d} common={len(a & b):3d} jaccard={jacc:.2f}")
    changed = (m["profile_base"] != m["profile_strict"]).sum()
    print(f"  players changing profile: {changed} / {len(m)} ({changed / len(m):.1%})")

    # Stage-2 source players (whose hands feed the agents): do they keep their profile?
    src = json.load(STAGE2_JSON.open(encoding="utf-8"))["player_profiles"]
    src_df = pd.DataFrame([(e["player_name"], e["profile"]) for e in src], columns=["player_name", "profile_src"])
    s = src_df.merge(strict[["player_name", "profile"]], on="player_name", how="left")
    keep = (s["profile_src"] == s["profile"]).sum()
    print(f"  Stage-2 source players keeping their profile under strict partscore: {keep} / {len(s)}")
    print("   ", s[s["profile_src"] != s["profile"]][["player_name", "profile_src", "profile"]].to_string(index=False)
          if keep < len(s) else "    (all kept)")

    # (ii) style alignment with the strict preempt definition on the bridge side
    nego = negotiation_aggression()
    variants = {
        "historical (paper)": (base, AGGR_FEATURES),
        "strict preempt": (feats2.merge(base[["player_name", "profile"]], on="player_name"),
                           ["slam_rate", "preempt_strict_rate", "penalty_double_rate"]),
        # `strict` already carries preempt_strict_rate (it was built from feats2)
        "strict preempt + strict-partscore profiles": (
            strict, ["slam_rate", "preempt_strict_rate", "penalty_double_rate"]),
    }
    print("\nStyle alignment (bridge side recomputed; negotiation side unchanged):")
    for name, (pf, cols) in variants.items():
        agg = composite(pf, cols)
        profs = [p for p in agg.index if p in nego.index]
        rho = spearmanr([agg[p] for p in profs], [nego[p] for p in profs]).statistic
        order = " > ".join(agg.loc[profs].sort_values(ascending=False).index)
        print(f"  {name:44s} rho = {rho:+.2f}   bridge order: {order}")
        rows.append(dict(profile=f"[alignment] {name}", rho=rho))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
