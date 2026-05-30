"""
notebooks/build_dashboard.py
============================
Build a single, self-contained HTML dashboard of the player-profile results.

The output (`docs/dashboard.html`) embeds every figure as a base64 data-URI, so
it is ONE portable file: double-click to open in any browser, e-mail it, or put
it on a USB stick. No server, no Python, no internet needed to view it.

Run:
    python notebooks/build_dashboard.py
"""

from __future__ import annotations

import base64
from datetime import date
from pathlib import Path

import pandas as pd

PROFILES_CSV = Path("data/processed/player_profiles.csv")
IMG_DIR = Path("docs/images")
OUT = Path("docs/dashboard.html")

# Figure file -> (title, plain-language caption)
FIGURES: list[tuple[str, str, str]] = [
    (
        "supervisor_box_defining_metrics.png",
        "1 · Each profile spikes on its own behaviour",
        "Four habits — bidding big (slam), playing safe (partscore), fighting "
        "opponents (penalty double), preferring NT. Each mini-chart picks one "
        "habit and shows how much every group does it. In every chart one group "
        "clearly stands out — and it is named after that habit.",
    ),
    (
        "supervisor_heatmap_fingerprint.png",
        "2 · The fingerprint of each profile",
        "One row per group, one column per behaviour. Red = does it more than "
        "average, blue = less. The strong red square on each row is the group's "
        "own habit. The other coloured squares are side-habits we did NOT select "
        "for — that coherence is the real evidence the groups are genuine.",
    ),
    (
        "supervisor_population_breakdown.png",
        "3 · How the players split up",
        "Left: most players (83%) are 'average'; the special types are small "
        "groups at the edges. Right: each special group vs. the average player, "
        "with a '×' showing how many times more they do their signature habit.",
    ),
    (
        "supervisor_scatter_risk_axis.png",
        "4 · The risk line: bold vs. careful",
        "Every dot is one player. Right = more risk (slams), up = more caution "
        "(stops low). No separate islands — one big cloud with the special types "
        "at the edges. This is the main finding: a smooth continuum, not boxes.",
    ),
    (
        "supervisor_scatter_fighter_nt.png",
        "5 · Fighters vs. NT-lovers",
        "Chart 4 separates two of the four types; this one separates the other "
        "two. Right = fights opponents more (penalty doubles), up = prefers NT "
        "contracts. Fighters drift right, NT Specialists drift up.",
    ),
]

# Profile cards: name, emoji, colour, plain description
PROFILE_CARDS: list[tuple[str, str, str, str]] = [
    ("Slam Hunter", "🎯", "#d62728", "Bids the big, risky contracts"),
    ("Insurance Player", "🛡️", "#1f77b4", "Plays it safe, stops low"),
    ("Fighter", "⚔️", "#ff7f0e", "Doubles & punishes opponents"),
    ("NT Specialist", "📐", "#2ca02c", "Prefers NoTrump contracts"),
    ("Generalist", "👤", "#9aa0a6", "The average elite player (baseline)"),
]

# Validation table rows: profile, metric, ratio, cohen_d, p
VALIDATION = [
    ("Fighter", "penalty-double rate", "×1.31", "2.13", "0.016"),
    ("Insurance Player", "partscore rate", "×1.24", "3.30", "0.004"),
    ("Slam Hunter", "slam rate", "×1.37", "2.81", "0.004"),
    ("NT Specialist", "NT rate", "×1.27", "4.62", "0.004"),
]


def img_data_uri(path: Path) -> str:
    """Return a base64 data-URI for a PNG file."""
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def main() -> None:
    df = pd.read_csv(PROFILES_CSV, encoding="utf-8-sig")
    counts = df["profile"].value_counts().to_dict()
    total = len(df)

    # Profile cards HTML
    cards_html = ""
    for name, emoji, color, desc in PROFILE_CARDS:
        n = counts.get(name, 0)
        pct = 100 * n / total
        cards_html += f"""
      <div class="card" style="border-top:5px solid {color}">
        <div class="emoji">{emoji}</div>
        <div class="cardname">{name}</div>
        <div class="count" style="color:{color}">{n}</div>
        <div class="pct">{pct:.1f}% of players</div>
        <div class="desc">{desc}</div>
      </div>"""

    # Validation table HTML
    rows_html = ""
    for prof, metric, ratio, d, p in VALIDATION:
        rows_html += f"""
        <tr>
          <td class="prof">{prof}</td>
          <td>{metric}</td>
          <td class="num">{ratio}</td>
          <td class="num">{d}</td>
          <td class="num">{p}</td>
          <td class="verdict">✓ strong</td>
        </tr>"""

    # Figures HTML
    figures_html = ""
    for fname, title, caption in FIGURES:
        path = IMG_DIR / fname
        if not path.exists():
            continue
        uri = img_data_uri(path)
        figures_html += f"""
      <section class="fig">
        <h3>{title}</h3>
        <img src="{uri}" alt="{title}"/>
        <p class="caption">{caption}</p>
      </section>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>NegoPlay — Player Profiles Snapshot</title>
<style>
  :root {{ --ink:#1a1a2e; --muted:#5a5a72; --bg:#f4f5fa; --card:#fff; --accent:#3b3b98; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
         color:var(--ink); background:var(--bg); line-height:1.55; }}
  .wrap {{ max-width:1040px; margin:0 auto; padding:32px 22px 80px; }}
  header {{ background:linear-gradient(135deg,#3b3b98,#5b2a86); color:#fff;
            border-radius:18px; padding:34px 30px; box-shadow:0 8px 30px rgba(59,59,152,.25); }}
  header h1 {{ margin:0 0 6px; font-size:30px; letter-spacing:-.5px; }}
  header p {{ margin:0; opacity:.92; font-size:16px; }}
  .chips {{ display:flex; flex-wrap:wrap; gap:12px; margin-top:20px; }}
  .chip {{ background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.25);
           padding:9px 15px; border-radius:30px; font-size:14px; font-weight:600; }}
  h2 {{ font-size:21px; margin:42px 0 16px; color:var(--accent); }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:16px; }}
  .card {{ background:var(--card); border-radius:14px; padding:18px 16px; text-align:center;
           box-shadow:0 2px 10px rgba(0,0,0,.06); }}
  .emoji {{ font-size:30px; }}
  .cardname {{ font-weight:700; margin-top:6px; font-size:15px; }}
  .count {{ font-size:30px; font-weight:800; margin-top:4px; }}
  .pct {{ font-size:12.5px; color:var(--muted); }}
  .desc {{ font-size:13px; color:var(--muted); margin-top:8px; }}
  table {{ width:100%; border-collapse:collapse; background:#fff; border-radius:14px;
           overflow:hidden; box-shadow:0 2px 10px rgba(0,0,0,.06); }}
  th, td {{ padding:13px 15px; text-align:left; border-bottom:1px solid #eef0f6; font-size:14.5px; }}
  th {{ background:#2f2f7a; color:#fff; font-weight:600; }}
  td.num {{ font-variant-numeric:tabular-nums; font-weight:600; }}
  td.prof {{ font-weight:700; }}
  td.verdict {{ color:#1a9850; font-weight:700; }}
  tr:last-child td {{ border-bottom:none; }}
  .note {{ background:#fff7e6; border-left:5px solid #f0a500; border-radius:10px;
           padding:16px 18px; margin-top:18px; font-size:14px; color:#5c4a13; }}
  .fig {{ background:#fff; border-radius:16px; padding:22px; margin-top:22px;
          box-shadow:0 2px 14px rgba(0,0,0,.07); }}
  .fig h3 {{ margin:0 0 14px; font-size:18px; color:var(--ink); }}
  .fig img {{ width:100%; height:auto; border-radius:8px; border:1px solid #eef0f6; }}
  .caption {{ font-size:14px; color:var(--muted); margin:14px 0 0; }}
  footer {{ margin-top:46px; text-align:center; color:var(--muted); font-size:13px; }}
</style>
</head>
<body>
  <div class="wrap">
    <header>
      <h1>NegoPlay — Player Profiles Snapshot</h1>
      <p>Behavioural decision-making styles of elite bridge players, learned from real tournament data.</p>
      <div class="chips">
        <span class="chip">{total} qualifying players</span>
        <span class="chip">5 European Championships (2016–2025)</span>
        <span class="chip">149K boards analysed</span>
        <span class="chip">4 validated profiles + baseline</span>
      </div>
    </header>

    <h2>The five player types</h2>
    <div class="cards">{cards_html}
    </div>

    <h2>Are the profiles real? — Statistical validation</h2>
    <p style="font-size:14.5px;color:var(--muted);margin-top:-6px">
      Each profile's defining behaviour, measured straight from the raw data
      (no AI, no clustering) and compared to the average player. Cohen's
      <em>d</em> ≥ 0.8 is "large", ≥ 2.0 is "very large"; all are significant
      at <em>p</em> &lt; 0.05.
    </p>
    <table>
      <thead><tr><th>Profile</th><th>Defining behaviour</th><th>vs. average</th>
        <th>Cohen's d</th><th>p-value</th><th>Verdict</th></tr></thead>
      <tbody>{rows_html}
      </tbody>
    </table>
    <div class="note">
      <strong>Honest note:</strong> charts 1 &amp; 3 partly re-describe how the
      groups were defined (we picked the most extreme players, so of course they
      look extreme). The genuinely independent evidence is chart 2's side-habits
      plus this validation table. The "Insurance Player" group is the weakest of
      the four (only ×1.24), so we describe it carefully.
    </div>

    <h2>The visualizations</h2>
    {figures_html}

    <footer>
      Generated {date.today().isoformat()} · NegoPlay (PhD baseline, Anna Ben-Shushan, LUT University)<br/>
      Reproduce: <code>python notebooks/build_dashboard.py</code>
    </footer>
  </div>
</body>
</html>"""

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    size_kb = OUT.stat().st_size / 1024
    print(f"Wrote {OUT}  ({size_kb:.0f} KB, self-contained)")
    print(f"Open it: {OUT.resolve()}")


if __name__ == "__main__":
    main()
