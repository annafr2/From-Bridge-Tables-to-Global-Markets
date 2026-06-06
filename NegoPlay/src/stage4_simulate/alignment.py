"""
src/stage4_simulate/alignment.py
================================
Stage 4c: the heart of the thesis.

Takes the per-profile win rates from BOTH domains:
  - bridge       (results/stage4/bridge_winrates.csv)
  - negotiation  (results/stage4/negotiation_winrates.csv)

and asks the research question:

    Does a profile that wins in BRIDGE also win in NEGOTIATION?

We answer it by ranking the profiles in each domain and computing **Spearman's
rho** between the two rankings. ρ is a rank correlation in [-1, +1]:
    ρ = +1  perfect agreement (same order in both domains)
    ρ =  0  no relationship
    ρ = -1  perfect disagreement (reversed order)

The pre-registered target (PRD H3) is ρ ≥ 0.70.

Honest caveat baked into the output: with only 5 profiles, Spearman has very low
statistical power — ρ must be near ±1 to reach p < 0.05. We therefore report the
ρ value AND its p-value AND state this limitation explicitly.

Outputs:
  results/stage4/alignment_report.md   (human-readable conclusion)
  results/stage4/alignment.png         (scatter: bridge vs negotiation win rate)
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results/stage4")
BRIDGE_CSV = RESULTS_DIR / "bridge_winrates.csv"
NEGO_CSV = RESULTS_DIR / "negotiation_winrates.csv"
REPORT_PATH = RESULTS_DIR / "alignment_report.md"
PLOT_PATH = RESULTS_DIR / "alignment.png"

TARGET_RHO = 0.70

PROFILE_COLORS = {
    "Generalist": "#9aa0a6", "Slam Hunter": "#d62728",
    "Insurance Player": "#1f77b4", "Fighter": "#ff7f0e", "NT Specialist": "#2ca02c",
}


def load_winrates() -> pd.DataFrame:
    """Merge bridge + negotiation win rates into one table keyed by profile."""
    b = pd.read_csv(BRIDGE_CSV).rename(columns={"bridge_winrate": "bridge"})
    n = pd.read_csv(NEGO_CSV).rename(columns={"negotiation_winrate": "negotiation"})
    df = b[["profile", "bridge"]].merge(n[["profile", "negotiation"]], on="profile")
    df = df.sort_values("bridge", ascending=False).reset_index(drop=True)
    return df


def compute_alignment(df: pd.DataFrame) -> dict:
    """Spearman ρ between the bridge and negotiation win-rate rankings."""
    rho, p = spearmanr(df["bridge"], df["negotiation"])
    df = df.copy()
    df["bridge_rank"] = df["bridge"].rank(ascending=False).astype(int)
    df["nego_rank"] = df["negotiation"].rank(ascending=False).astype(int)
    return {
        "rho": round(float(rho), 4),
        "p_value": round(float(p), 4),
        "n_profiles": len(df),
        "target": TARGET_RHO,
        "meets_target": bool(rho >= TARGET_RHO),
        "table": df,
    }


def make_plot(df: pd.DataFrame, rho: float) -> Path:
    fig, ax = plt.subplots(figsize=(9, 7))
    for _, r in df.iterrows():
        c = PROFILE_COLORS.get(r["profile"], "#333333")
        ax.scatter(r["bridge"], r["negotiation"], s=180, color=c,
                   edgecolors="black", zorder=3)
        ax.annotate(r["profile"], (r["bridge"], r["negotiation"]),
                    xytext=(8, 4), textcoords="offset points", fontsize=10)
    # diagonal reference (perfect agreement)
    lo = min(df["bridge"].min(), df["negotiation"].min()) - 0.05
    hi = max(df["bridge"].max(), df["negotiation"].max()) + 0.05
    ax.plot([lo, hi], [lo, hi], ls="--", color="gray", alpha=0.6,
            label="perfect agreement")
    ax.set_xlabel("Bridge win rate  (Stage 4a)", fontsize=12)
    ax.set_ylabel("Negotiation win rate  (Stage 4b)", fontsize=12)
    ax.set_title(f"Cross-domain alignment\nSpearman ρ = {rho:+.2f}  "
                 f"(target ≥ {TARGET_RHO})", fontweight="bold", fontsize=13)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return PLOT_PATH


def write_report(result: dict) -> Path:
    df = result["table"]
    rho, p, n = result["rho"], result["p_value"], result["n_profiles"]
    verdict = (
        "SUPPORTED" if rho >= TARGET_RHO
        else "PARTIALLY SUPPORTED" if rho >= 0.5
        else "NOT SUPPORTED (null result)"
    )

    lines = [
        "# Stage 4c — Cross-Domain Alignment Report",
        "",
        "**Research question:** does a profile that wins in bridge also win in "
        "business negotiation?",
        "",
        f"**Spearman ρ = {rho:+.3f}**  (p = {p:.3f}, n = {n} profiles)",
        f"**Target (PRD H3): ρ ≥ {TARGET_RHO}**  →  **{verdict}**",
        "",
        "## Win rates by profile",
        "",
        "| Profile | Bridge | Bridge rank | Negotiation | Nego rank |",
        "|---------|--------|-------------|-------------|-----------|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['profile']} | {r['bridge']:.3f} | {r['bridge_rank']} | "
            f"{r['negotiation']:.3f} | {r['nego_rank']} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        f"- ρ = +1 means the two domains rank the profiles identically; "
        f"ρ = 0 means no relationship; ρ = -1 means reversed.",
        f"- We observed ρ = {rho:+.3f}.",
        "",
        "## Honest limitations",
        "",
        "1. **Low statistical power:** with only 5 profiles, Spearman needs ρ very "
        "close to ±1 to reach p < 0.05. The ρ *value* is the signal; the p-value "
        "is expected to be weak at this sample size. This is a known constraint, "
        "stated up front.",
        "2. **Bridge win rate is an HCP-based proxy** (par level), not real "
        "double-dummy results.",
        "3. **The Fighter's defining skill (penalty doubles) is not rewarded by "
        "the bridge proxy**, so its bridge rank may understate its real style.",
        "",
        "*Generated by `src/stage4_simulate/alignment.py`.*",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return REPORT_PATH


def run_alignment() -> dict:
    df = load_winrates()
    result = compute_alignment(df)
    make_plot(result["table"], result["rho"])
    write_report(result)
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = run_alignment()
    print("\n=== CROSS-DOMAIN ALIGNMENT ===")
    print(result["table"][["profile", "bridge", "negotiation"]].to_string(index=False))
    print(f"\nSpearman rho = {result['rho']:+.3f}  (p = {result['p_value']:.3f})")
    print(f"Target >= {TARGET_RHO}  ->  meets target: {result['meets_target']}")
    print(f"\nSaved: {REPORT_PATH}")
    print(f"Saved: {PLOT_PATH}")


if __name__ == "__main__":
    main()


__all__ = ["run_alignment", "compute_alignment", "load_winrates"]
