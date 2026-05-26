# NegoPlay — Architecture

## Four-Stage Pipeline

```mermaid
flowchart TD
    A[("📊 Dataset\n149,208 bridge boards\nall_matches_full.csv")]

    A --> B

    subgraph S1["Stage 1 — Profile Discovery (ML)"]
        B["features.py\nslam_rate, success_rate\npreempt_rate, double_rate\navg_risk_score"]
        B --> C["clustering.py\nK-Means k=2..6\n+ HDBSCAN validation"]
        C --> D["validation.py\nSilhouette ≥ 0.4\np < 0.05"]
    end

    D --> E

    subgraph S2["Stage 2 — Skill Extraction (LLM)"]
        E["chunker.py\n20-30 games/chunk"]
        E --> F["extractor.py\nGemini Flash 2.0\nJSON schema"]
        F --> G["aggregator.py\n5-7 skills per profile\nSlam Hunter, Insurance Player..."]
    end

    G --> H

    subgraph S3["Stage 3 — Agent Construction"]
        H["base_agent.py\nBaseAgent ABC"]
        H --> I["bridge_agent.py\nmake_bid()"]
        H --> J["nego_agent.py\nrespond_to_offer()"]
    end

    I & J --> K

    subgraph S4["Stage 4 — Dual Simulation + Alignment"]
        K["bridge_game.py\n50 bridge games"]
        K --> L["negotiation.py\n48 negotiation sessions\nM&A / JV / SaaS / Tender"]
        L --> M["alignment.py\nSpearman ρ ≥ 0.7?\n← THE RESULT"]
    end

    style S1 fill:#D6EAF8,stroke:#2980B9
    style S2 fill:#D5F5E3,stroke:#27AE60
    style S3 fill:#FDEBD0,stroke:#E67E22
    style S4 fill:#F9EBEA,stroke:#E74C3C
```

## Data Flow

```
data/raw/all_matches_full.csv          (read-only)
  ↓ src/stage1_clustering/features.py
data/processed/player_features.csv
  ↓ src/stage1_clustering/clustering.py
data/processed/player_clusters.csv
  ↓ src/stage2_skills/extractor.py     (Gemini Flash 2.0)
data/processed/skill_profiles.json
  ↓ src/stage3_agents/*.py
[agents in memory]
  ↓ src/stage4_simulate/*.py           (Gemini Flash 2.0)
results/bridge_simulations.jsonl
results/negotiation_simulations.jsonl
  ↓ src/stage4_simulate/alignment.py
results/alignment_report.md            ← FINAL OUTPUT
```

## LLM Provider Policy

| Task | Default | Alt |
|------|---------|-----|
| Skill extraction | Gemini Flash 2.0 | Claude / GPT for validation |
| Bridge agents | Gemini Flash 2.0 | Cross-model comparison |
| Negotiation agents | Gemini Flash 2.0 | Cross-model comparison |
| Final synthesis | Gemini 2.5 Pro | Claude Opus |

All calls route through `src/shared/llm_client.py` with `provider=` argument.
