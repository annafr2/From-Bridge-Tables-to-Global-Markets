"""
notebooks/split_half_validation.py
==================================
Out-of-sample validation of the Stage-1 profiles (reviewer point: profiles are
selected on a measure and then validated on the same measure, so large effects
are partly a selection artefact).

Design: split every player's boards by board-number parity. Profiles are
assigned on the ODD boards only (half A); the defining rates are then measured
on the EVEN boards only (half B), which played no part in the assignment. If
the profiles are stable behaviour, profile members must still exceed the
Generalists on half B. Qualification thresholds are halved (25 declared and
25 bidding boards per half) because each half holds about half of a player's
boards.

Also reported: agreement between the half-A and half-B assignments, and
recovery of the full-data profile members (data/processed/player_profiles.csv).

Deterministic, no LLM, no cost. Output: console + results/stage1_split_half.csv
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, pearsonr, spearmanr

from src.stage1_clustering.extreme_profiles import PROFILE_AXES, assign_extreme_profiles
from src.stage1_clustering.features import compute_player_features

logging.basicConfig(level=logging.WARNING)

ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT.parent / "data" / "processed" / "all_matches_full.csv"
DATA_PARQUET = ROOT.parent / "data" / "processed" / "all_matches.parquet"
PROFILES_CSV = ROOT / "data" / "processed" / "player_profiles.csv"
STAGE2_JSON = ROOT / "results" / "stage2_sample_v2_focused_prompt" / "skill_profiles_reagg_t040.json"
OUT = ROOT / "results" / "stage1_split_half.csv"

# Qualification per half. 25 = half of the paper's n >= 50; pass 50 on the
# command line to require the paper's full qualification inside EACH half.
HALF_MIN_BOARDS = int(sys.argv[1]) if len(sys.argv) > 1 else 25
RELIABILITY_FEATURES = ["slam_rate", "partscore_rate", "nt_rate", "penalty_double_rate",
                        "preempt_rate", "opening_rate", "intervention_rate", "double_rate"]


def load() -> pd.DataFrame:
    """Load the corpus the same way the Stage-1 pipeline does."""
    try:
        from src.shared.data_loader import load_matches
        return load_matches(str(DATA_CSV))
    except Exception:  # noqa: BLE001 - fall back to the parquet master file
        return pd.read_parquet(DATA_PARQUET)


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                     / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else float("nan")


def split(df: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the corpus into two halves that share no board.

    parity: odd vs even board numbers. NOT recommended for seat-dependent
            features: in duplicate bridge the dealer rotates N, E, S, W with
            the board number, so odd boards are always dealt by N or S and
            even boards by E or W, which biases opening and intervention
            rates between the halves.
    block4: boards 1-4, 9-12, ... vs 5-8, 13-16, ...; every half contains all
            four dealers equally.
    match:  a seeded random half of the matches vs the other half; balances
            dealer, vulnerability and opponents, and keeps a whole match on
            one side (the default).
    """
    b = df["_board"].astype(int)
    if mode == "parity":
        mask = b % 2 == 1
    elif mode == "block4":
        mask = ((b - 1) // 4) % 2 == 0
    elif mode == "match":
        matches = pd.Series(sorted(df["match_id"].astype(str).unique()))
        rng = np.random.default_rng(42)
        chosen = set(matches.sample(frac=0.5, random_state=int(rng.integers(1 << 31))))
        mask = df["match_id"].astype(str).isin(chosen)
    else:
        raise ValueError(mode)
    return df[mask], df[~mask]


def main() -> None:
    mode = sys.argv[2] if len(sys.argv) > 2 else "match"
    df = load()
    df["_board"] = pd.to_numeric(df["board"], errors="coerce")
    df = df.dropna(subset=["_board"])
    half_a, half_b = split(df, mode)
    print(f"split mode = {mode} | rows: all={len(df)}  half A={len(half_a)}  half B={len(half_b)}")

    # Full-data reference (should reproduce the paper: 20/21/37/17/468)
    full = assign_extreme_profiles(compute_player_features(df, 50, 50))
    print("full-data profiles:", full["profile"].value_counts().to_dict())

    feats_a = compute_player_features(half_a, HALF_MIN_BOARDS, HALF_MIN_BOARDS)
    feats_b = compute_player_features(half_b, HALF_MIN_BOARDS, HALF_MIN_BOARDS)
    prof_a = assign_extreme_profiles(feats_a)
    prof_b = assign_extreme_profiles(feats_b)
    print("half-A profiles:", prof_a["profile"].value_counts().to_dict())
    print("half-B profiles:", prof_b["profile"].value_counts().to_dict())

    rates_b = feats_b.set_index("player_name")
    gens_a = prof_a.loc[prof_a["profile"] == "Generalist", "player_name"]
    gens_in_b = [p for p in gens_a if p in rates_b.index]

    rows = []
    print("\nOut-of-sample test: profiles assigned on ODD boards, rates measured on EVEN boards")
    print(f"  {'profile':18s} {'axis':22s} {'n_prof':>6s} {'n_gen':>6s} {'prof_mean':>9s} {'gen_mean':>8s} "
          f"{'ratio':>6s} {'d':>6s} {'p(MWU)':>8s}")
    for profile, axis in PROFILE_AXES.items():
        members = prof_a.loc[prof_a["profile"] == profile, "player_name"]
        members_in_b = [p for p in members if p in rates_b.index]
        x = rates_b.loc[members_in_b, axis].to_numpy(dtype=float)
        y = rates_b.loc[gens_in_b, axis].to_numpy(dtype=float)
        if len(x) < 2:
            print(f"  {profile:18s} too few members present in half B ({len(x)})")
            continue
        p = mannwhitneyu(x, y, alternative="greater").pvalue
        d = cohen_d(x, y)
        ratio = x.mean() / y.mean() if y.mean() > 0 else float("nan")
        rows.append(dict(profile=profile, axis=axis, n_profile=len(x), n_generalist=len(y),
                         profile_mean_B=x.mean(), generalist_mean_B=y.mean(), ratio=ratio,
                         cohen_d=d, p_mwu_one_sided=p))
        print(f"  {profile:18s} {axis:22s} {len(x):6d} {len(y):6d} {x.mean():9.3f} {y.mean():8.3f} "
              f"{ratio:6.2f} {d:6.2f} {p:8.4f}")

    # Agreement between the two independent halves
    both = prof_a.merge(prof_b, on="player_name", suffixes=("_a", "_b"))
    extreme_a = both[both["profile_a"] != "Generalist"]
    same = (extreme_a["profile_a"] == extreme_a["profile_b"]).mean() if len(extreme_a) else float("nan")
    still_extreme = (extreme_a["profile_b"] != "Generalist").mean() if len(extreme_a) else float("nan")
    print(f"\nPlayers qualified in both halves: {len(both)}; extreme in A: {len(extreme_a)}")
    print(f"  of A's extreme players, same profile in B: {same:.1%}; any extreme profile in B: {still_extreme:.1%}")

    # Recovery of the published full-data profile members
    pub = pd.read_csv(PROFILES_CSV)
    pub_ext = pub[pub["profile"] != "Generalist"][["player_name", "profile"]]
    rec = pub_ext.merge(prof_a[["player_name", "profile"]], on="player_name", how="left",
                        suffixes=("_full", "_a"))
    present = rec["profile_a"].notna()
    recovered = (rec.loc[present, "profile_full"] == rec.loc[present, "profile_a"]).mean()
    print(f"  published extreme players present in half A: {present.sum()}/{len(rec)}; "
          f"same profile from odd boards alone: {recovered:.1%}")

    # Split-half reliability of each feature: does a player's rate on odd boards
    # predict the same player's rate on even boards? (Spearman-Brown corrects to
    # the full-length record.)
    fa = feats_a.set_index("player_name")
    fb = feats_b.set_index("player_name")
    common = fa.index.intersection(fb.index)
    print(f"\nSplit-half reliability over {len(common)} players qualified in both halves "
          f"(>= {HALF_MIN_BOARDS} declared and bidding boards per half)")
    print(f"  {'feature':22s} {'pearson_r':>9s} {'spearman':>9s} {'Spearman-Brown':>14s}")
    for col in RELIABILITY_FEATURES:
        if col not in fa.columns or col not in fb.columns:
            continue
        x, y = fa.loc[common, col].to_numpy(dtype=float), fb.loc[common, col].to_numpy(dtype=float)
        r = pearsonr(x, y).statistic
        rs = spearmanr(x, y).statistic
        sb = 2 * r / (1 + r) if r > -1 else float("nan")
        rows.append(dict(profile=f"[reliability] {col}", pearson_r=r, spearman_r=rs, spearman_brown=sb,
                         n_players=len(common)))
        print(f"  {col:22s} {r:9.2f} {rs:9.2f} {sb:14.2f}")

    # The 20 Stage-2 source players (whose hands feed the agents): rate on each
    # half of their own boards versus the Generalist mean on the same half.
    src = json.load(STAGE2_JSON.open(encoding="utf-8"))["player_profiles"]
    print("\nStage-2 source players: defining rate on odd boards / even boards (Generalist mean in brackets)")
    gen_full = full.loc[full["profile"] == "Generalist", "player_name"]
    for profile, axis in PROFILE_AXES.items():
        ga = fa.loc[fa.index.intersection(gen_full), axis].mean()
        gb = fb.loc[fb.index.intersection(gen_full), axis].mean()
        names = [e["player_name"] for e in src if e["profile"] == profile]
        parts = []
        above_both = 0
        for n in names:
            va = fa[axis].get(n, np.nan)
            vb = fb[axis].get(n, np.nan)
            above_both += int(va > ga and vb > gb)
            parts.append(f"{va:.3f}/{vb:.3f}")
        print(f"  {profile:18s} [{ga:.3f}/{gb:.3f}]  " + "  ".join(parts)
              + f"   -> above Generalist mean on BOTH halves: {above_both}/{len(names)}")
        rows.append(dict(profile=f"[source players] {profile}", n_profile=len(names),
                         above_generalist_both_halves=above_both))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT.with_name(f"stage1_split_half_{mode}_min{HALF_MIN_BOARDS}.csv"), index=False)
    print(f"\nsaved {OUT.with_name(f'stage1_split_half_min{HALF_MIN_BOARDS}.csv')}")


if __name__ == "__main__":
    main()
