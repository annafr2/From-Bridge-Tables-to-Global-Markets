"""
notebooks/replicate_with_model.py
=================================
Cross-version replication of the two headline negotiation results with a
DIFFERENT language model, using the archived Stage-2 skills and the same
rule-based seller:

  * matched condition: each profile's agent carries its own skills
      -> demand depth theta per profile, Spearman rho against bridge aggression
  * inverse-prompt control: skills swapped to the aggression-opposite profile
      -> theta per profile, rho (should flip sign if behaviour follows skills)

Both conditions run with ONE model, so the comparison is free of the
model mismatch of the original runs (matched: gemini-2.5-flash, control:
gemini-2.5-flash-lite). The archived original logs are never touched; this
script writes to results/stage4/replication_<model>/.

Usage (from the NegoPlay root, GOOGLE_API_KEY in .env; costs real money):
    python -m notebooks.replicate_with_model gemini-3.7-flash
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
from scipy.stats import spearmanr

from notebooks.style_alignment import bridge_aggression
from src.shared.llm_client import LLMClient
from src.shared.prompts import load_profile_signatures
from src.stage3_agents.nego_agent import NegotiationAgent
from src.stage4_simulate.negotiation import SCENARIOS, _run_one_negotiation

load_dotenv()
logging.disable(logging.WARNING)

ORDER = ["Fighter", "Slam Hunter", "NT Specialist", "Generalist", "Insurance Player"]
INVERSE = {p: ORDER[len(ORDER) - 1 - i] for i, p in enumerate(ORDER)}
RUNS = 3


def exact_two_sided_p(x: list[float], y: list[float]) -> float:
    """Exact permutation p-value for Spearman's rho with n <= 7 (ties: average ranks)."""
    rho_obs = abs(spearmanr(x, y).statistic)
    hits = sum(1 for perm in itertools.permutations(range(len(y)))
               if abs(spearmanr(x, [y[i] for i in perm]).statistic) >= rho_obs - 1e-12)
    return hits / len(list(itertools.permutations(range(len(y)))))


def run_condition(agents: dict, out_path: Path) -> tuple[dict[str, float], dict[str, float]]:
    """Run all scenarios x runs for the given agents; return (theta, surplus) per profile."""
    spans = {s["title"]: (s["seller_open"], s["seller_floor"]) for s in SCENARIOS}
    depth: dict[str, list[float]] = {p: [] for p in agents}
    surplus: dict[str, list[float]] = {p: [] for p in agents}
    rows_all: list[dict] = []
    for profile, agent in agents.items():
        for sc in SCENARIOS:
            for run in range(RUNS):
                rows: list[dict] = []
                s = _run_one_negotiation(agent, sc, rows, profile, run)
                surplus[profile].append(s)
                rows_all.extend(rows)
                open_, floor = spans[sc["title"]]
                for r in rows:
                    if r.get("action") == "counter" and r.get("offer") and r["offer"].get("price_musd") is not None:
                        depth[profile].append(max(0.0, min(1.3, (open_ - r["offer"]["price_musd"]) / (open_ - floor))))
                print(f"    {profile:18s} {sc['title'][:34]:34s} run {run}: surplus {s:.2f}", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows_all:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    theta = {p: (sum(v) / len(v) if v else float("nan")) for p, v in depth.items()}
    surp = {p: (sum(v) / len(v) if v else float("nan")) for p, v in surplus.items()}
    return theta, surp


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "gemini-3.7-flash"
    out_dir = Path("results/stage4") / f"replication_{model}"
    client = LLMClient(model=model)
    sigs = load_profile_signatures()
    b = bridge_aggression()
    profiles = [p for p in ORDER if p in b.index]

    print(f"=== replication with {model}: matched condition")
    matched_agents = {p: NegotiationAgent(sigs[p], client=client) for p in ORDER}
    theta_m, surp_m = run_condition(matched_agents, out_dir / "matched.jsonl")

    print(f"=== replication with {model}: inverse-prompt control (skills swapped)")
    swapped_agents = {p: NegotiationAgent(dataclasses.replace(sigs[p], skills=sigs[INVERSE[p]].skills),
                                          client=client) for p in ORDER}
    theta_i, surp_i = run_condition(swapped_agents, out_dir / "inverse.jsonl")

    x = [b[p] for p in profiles]
    rho_m = spearmanr(x, [theta_m[p] for p in profiles]).statistic
    rho_i = spearmanr(x, [theta_i[p] for p in profiles]).statistic
    p_m = exact_two_sided_p(x, [theta_m[p] for p in profiles])
    p_i = exact_two_sided_p(x, [theta_i[p] for p in profiles])
    summary = {
        "model": model, "runs": RUNS, "scenarios": len(SCENARIOS),
        "bridge_aggression": {p: round(float(b[p]), 3) for p in profiles},
        "theta_matched": {p: round(theta_m[p], 3) for p in profiles},
        "theta_inverse": {p: round(theta_i[p], 3) for p in profiles},
        "surplus_matched": {p: round(surp_m[p], 3) for p in profiles},
        "rho_matched": round(float(rho_m), 3), "p_exact_matched": round(p_m, 3),
        "rho_inverse": round(float(rho_i), 3), "p_exact_inverse": round(p_i, 3),
        "llm_cost_usd_logged": round(client.cumulative_cost(), 4),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== SUMMARY")
    print(f"  {'profile':18s} {'bridge_z':>9s} {'theta_matched':>13s} {'theta_inverse':>13s}")
    for p in profiles:
        print(f"  {p:18s} {b[p]:9.2f} {theta_m[p]:13.3f} {theta_i[p]:13.3f}")
    print(f"  matched: rho = {rho_m:+.2f} (exact p = {p_m:.3f}) | inverse: rho = {rho_i:+.2f} (exact p = {p_i:.3f})")
    print(f"  original paper: matched +0.80 (p 0.133), inverse -0.90 (p 0.083)")
    print(f"  saved {out_dir}")


if __name__ == "__main__":
    main()
