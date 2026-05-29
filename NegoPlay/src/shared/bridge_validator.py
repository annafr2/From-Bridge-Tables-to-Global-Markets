"""
src/shared/bridge_validator.py
================================
Bridge Expert Validation Skill — first-pass statistical sanity checker.

Wraps a Gemini (default) / Claude / OpenAI call with a deep bridge
domain-knowledge system prompt so you can quickly validate claims like:

    "Is slam_rate=0.22 plausible for a player with 40 declared boards?"
    "Should penalty_double_rate=0.131 over 89 boards qualify as a Fighter?"

The LLM acts as an expert reviewer trained on:
  - Laws of Duplicate Bridge (auction, scoring, trick counts)
  - NegoPlay population baselines (563 qualifying players, ≥50 boards)
  - Statistical rules (sample size gates, binomial CIs, Nezer's minimum)

Usage (Python):
    from src.shared.bridge_validator import BridgeValidator
    v = BridgeValidator()
    result = v.validate("SMITH John has slam_rate=0.22 over 60 declared boards")
    print(result)            # coloured one-liner + action
    print(result.raw_json)   # full 4-part JSON

Usage (convenience method):
    result = v.validate_profile_assignment(
        player_name="SMITH John",
        profile="Slam Hunter",
        metric_value=0.101,
        n_boards=216,
    )

Usage (CLI):
    python -m src.shared.bridge_validator "slam_rate=0.22 over 60 boards"
    BRIDGE_VALIDATOR_PROVIDER=anthropic python -m src.shared.bridge_validator "..."
"""

from __future__ import annotations

import json
import logging
import os
import sys
import textwrap
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ── NegoPlay population baselines ─────────────────────────────────────────────
# Source: Stage 1 results — 563 qualifying players (≥50 declared + ≥50 bidding)
# European Championships 2016–2025, 149,208 boards

POPULATION_BASELINES: dict[str, float] = {
    "slam_rate": 0.055,             # 5.5% of declared contracts are slams
    "partscore_rate": 0.570,        # 57.0% of declared contracts are partscores
    "nt_rate": 0.282,               # 28.2% of declared contracts are NT
    "penalty_double_rate": 0.085,   # 8.5% of bidding boards have a penalty double
}

# Profile → metric + denominator column + minimum qualifying ratio
PROFILE_THRESHOLDS: dict[str, dict] = {
    "Slam Hunter": {
        "metric": "slam_rate",
        "denominator": "n_declared",
        "ratio": 1.84,
    },
    "Insurance Player": {
        "metric": "partscore_rate",
        "denominator": "n_declared",
        "ratio": 1.20,
    },
    "NT Specialist": {
        "metric": "nt_rate",
        "denominator": "n_declared",
        "ratio": 1.36,
    },
    "Fighter": {
        "metric": "penalty_double_rate",
        "denominator": "n_bidding_boards",
        "ratio": 1.55,
    },
}

# Minimum boards before any rate estimate is reliable (Nezer's rule, May 2026)
MINIMUM_BOARDS: int = 50


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = textwrap.dedent(f"""
You are a **certified Bridge Expert Validator** embedded in the NegoPlay
research pipeline (PhD research, LUT University 2026–2030).

## Your Role
You are the first-pass statistical sanity checker for claims derived from
149,208 elite bridge tournament hands (European Championships 2016–2025,
563 qualifying players).

You do NOT decide whether a claim is "interesting." You decide:
1. Is it bridge-legal? (can this rate physically exist?)
2. Is the sample large enough to trust?
3. Is the rate plausible for elite play, or does it signal noise/error?
4. What is the practical impact on the NegoPlay profile assignment?

---

## Bridge Domain Knowledge

### Auction basics
- 13 tricks per deal; 4 players (N/S/E/W); 2 partnerships (N-S vs E-W)
- Bidding levels 1–7; suits rank: ♣ < ♦ < ♥ < ♠ < NT
- Game thresholds: 3NT, 4♥, 4♠, 5♣, 5♦
- Small slam = level 6 (needs 12 of 13 tricks)
- Grand slam = level 7 (needs all 13 tricks)
- Doubled (X) / Redoubled (XX): opponent challenges your contract
- Passed-out board ("-"): no contract, no tricks, no score
- Each player declares approximately 25–30% of boards they play
  (roughly 1 in 4 deals; they defend the other ~70–75%)

### Frequency priors (elite pairs, EBL / WBF level)
- **Slam rate ~4–6%** of declared contracts
  → >10% over ≥50 boards is highly notable (top ~5% of the field)
  → >20% over ≥50 boards is extraordinary — requires extreme hand selection
  → >30% is physically implausible in balanced team play
- **Partscore rate ~55–65%**; most deals lack the HCP for game
  → >90% would mean the player never bids game even with values — suspicious
- **NT rate ~25–35%**; balanced hands with stoppers
  → >60% is implausible (can't play NT on unbalanced hands without opponent objection)
- **Penalty double rate ~7–13%** of boards with competitive bidding
  → >25% suggests systematic overbidding or misclassification
  → >40% is physically implausible (most opponents do not invite penalties)

### Sample size rules (Nezer's principle, April 2026)
- **Hard minimum:** ≥50 declared boards (outcome features) OR ≥50 bidding boards
  (process features) before ANY rate estimate is considered reliable
- **Rare events like slam (baseline ~5%):** With n=20 boards the 95% CI for
  observed rate 10% is approximately [1%, 32%] — overlapping the baseline.
  Need n≥80 for CIs to narrow below ±10pp for slam.
- **Binomial test gate:** p<0.05 (one-sided) vs population baseline is required
  before labelling a player as an "extreme profile"
- **Nezer's rule:** "20 declared boards is not enough to say anything about
  a player's slam frequency." Any claim based on n<30 for rare events must be
  flagged as SUSPICIOUS or IMPLAUSIBLE.

### NegoPlay population baselines (563 qualifying players, ≥50 boards each)
| Metric               | Pop. mean | Profile label    | Min ratio |
|----------------------|-----------|------------------|-----------|
| slam_rate            | {POPULATION_BASELINES['slam_rate']:.3f}    | Slam Hunter      | 1.84×     |
| partscore_rate       | {POPULATION_BASELINES['partscore_rate']:.3f}    | Insurance Player | 1.20×     |
| nt_rate              | {POPULATION_BASELINES['nt_rate']:.3f}    | NT Specialist    | 1.36×     |
| penalty_double_rate  | {POPULATION_BASELINES['penalty_double_rate']:.3f}    | Fighter          | 1.55×     |

Denominator: outcome features (slam, partscore, NT) use `n_declared`;
process feature (penalty double) uses `n_bidding_boards`.

---

## Output Format

Respond ONLY with the following JSON — no preamble, no markdown fence:

{{
  "legality_check": {{
    "is_legal": true,
    "issue": null,
    "details": "Explanation of whether the rate is within the physically
                possible range for bridge."
  }},
  "probability_assessment": {{
    "verdict": "PLAUSIBLE",
    "percentile_estimate": "e.g., top 5% of all qualifying players",
    "sample_size_adequate": true,
    "sample_size_comment": "e.g., n=216 is well above the 50-board minimum",
    "ci_note": "e.g., approximate 95% CI [7%, 14%] for rate=0.101 with n=216"
  }},
  "expert_analysis": {{
    "bridge_plausibility": "Is this rate consistent with elite play?",
    "statistical_concern": "Any red flags (noise, selection bias, data error)?",
    "alternative_explanation": "Could something other than true skill explain this?",
    "negoplay_impact": "Does this affect NegoPlay profile assignment reliability?"
  }},
  "final_verdict": {{
    "label": "ACCEPT",
    "one_liner": "≤20-word plain-English verdict",
    "recommended_action": "e.g., Proceed as-is / Flag in paper footnote / Raise min_boards"
  }}
}}

Allowed labels:
- "is_legal": true | false
- "verdict": "PLAUSIBLE" | "SUSPICIOUS" | "IMPLAUSIBLE"
- "sample_size_adequate": true | false
- "label": "ACCEPT" | "ACCEPT_WITH_CAVEAT" | "REJECT"

---

## Validation Rules (apply in this order)

1. **Legal gate first.** Does the claimed value violate bridge laws?
   Rate > 1.0 is impossible. slam_rate > 0.50 is physically implausible even in theory.

2. **Sample size gate.** n<50 for outcome features or n<50 for process features
   → always set `sample_size_adequate: false` and include CI estimate in `ci_note`.

3. **Nezer's rare-event rule.** For slam_rate or penalty_double_rate,
   n<30 → SUSPICIOUS regardless of observed rate. n<20 → IMPLAUSIBLE.

4. **Ratio check.** Does the observed rate exceed the profile threshold ratio
   (1.84× for slam, 1.55× for fights, etc.)? If barely above threshold, note it.

5. **Calibrated skepticism thresholds:**
   - slam_rate > 0.20 over ≥50 boards → SUSPICIOUS
   - slam_rate > 0.30 → IMPLAUSIBLE (set is_legal: false)
   - partscore_rate > 0.90 → SUSPICIOUS
   - penalty_double_rate > 0.25 → SUSPICIOUS
   - penalty_double_rate > 0.40 → IMPLAUSIBLE
   - nt_rate > 0.60 → IMPLAUSIBLE

6. **Honesty about uncertainty.** If context is insufficient, say so in
   `alternative_explanation` and use ACCEPT_WITH_CAVEAT, not ACCEPT.
""").strip()


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """Structured result from the Bridge Expert Validator.

    The four sub-sections mirror the LLM's structured JSON output.
    """

    # Final verdict
    verdict: str             # "ACCEPT" | "ACCEPT_WITH_CAVEAT" | "REJECT"
    one_liner: str           # ≤20-word plain summary
    recommended_action: str  # what to do next

    # Legality
    is_legal: bool
    legality_detail: str

    # Probability assessment
    probability_label: str   # "PLAUSIBLE" | "SUSPICIOUS" | "IMPLAUSIBLE"
    percentile_estimate: str
    sample_size_ok: bool
    sample_size_comment: str
    ci_note: str

    # Expert analysis
    bridge_plausibility: str
    statistical_concern: str
    alternative_explanation: str
    negoplay_impact: str

    # Raw LLM JSON (for logging / debugging)
    raw_json: dict

    # ── factories ──────────────────────────────────────────────────────────────

    @classmethod
    def from_raw(cls, data: dict) -> "ValidationResult":
        """Build a ValidationResult from the LLM's parsed JSON dict."""
        fv = data.get("final_verdict", {})
        pa = data.get("probability_assessment", {})
        ea = data.get("expert_analysis", {})
        lc = data.get("legality_check", {})
        return cls(
            verdict=fv.get("label", "UNKNOWN"),
            one_liner=fv.get("one_liner", ""),
            recommended_action=fv.get("recommended_action", ""),
            is_legal=lc.get("is_legal", True),
            legality_detail=lc.get("details", ""),
            probability_label=pa.get("verdict", "UNKNOWN"),
            percentile_estimate=pa.get("percentile_estimate", ""),
            sample_size_ok=pa.get("sample_size_adequate", True),
            sample_size_comment=pa.get("sample_size_comment", ""),
            ci_note=pa.get("ci_note", ""),
            bridge_plausibility=ea.get("bridge_plausibility", ""),
            statistical_concern=ea.get("statistical_concern", ""),
            alternative_explanation=ea.get("alternative_explanation", ""),
            negoplay_impact=ea.get("negoplay_impact", ""),
            raw_json=data,
        )

    # ── display ────────────────────────────────────────────────────────────────

    def __str__(self) -> str:
        icon = {"ACCEPT": "✅", "ACCEPT_WITH_CAVEAT": "⚠️", "REJECT": "❌"}.get(
            self.verdict, "❓"
        )
        prob_icon = {
            "PLAUSIBLE": "🟢", "SUSPICIOUS": "🟡", "IMPLAUSIBLE": "🔴",
        }.get(self.probability_label, "⬜")
        sample_icon = "✓" if self.sample_size_ok else "✗"

        lines = [
            f"{icon}  {self.verdict}: {self.one_liner}",
            f"",
            f"   Probability:   {prob_icon} {self.probability_label}  ({self.percentile_estimate})",
            f"   Sample size:   [{sample_icon}] {self.sample_size_comment}",
        ]
        if self.ci_note:
            lines.append(f"   95% CI:        {self.ci_note}")
        lines.append(f"")
        lines.append(f"   Bridge check:  {self.bridge_plausibility}")
        if self.statistical_concern:
            lines.append(f"   Stat concern:  {self.statistical_concern}")
        if not self.is_legal:
            lines.append(f"   ⛔ ILLEGAL:     {self.legality_detail}")
        lines.append(f"")
        lines.append(f"   → Action:      {self.recommended_action}")
        return "\n".join(lines)


# ── BridgeValidator ────────────────────────────────────────────────────────────

class BridgeValidator:
    """LLM-backed bridge expert that validates statistical claims.

    Routes through Gemini Flash 2.0 by default for cost efficiency.
    Claude (Haiku) or OpenAI (gpt-4o-mini) can be selected via the
    ``provider`` argument.  All three implement the same 4-part JSON
    output schema.

    Once ``src/shared/llm_client.py`` is built, this class will delegate
    to it instead of managing its own API clients.

    Args:
        provider:    "gemini" (default) | "anthropic" | "openai"
        model:       Override the default model for the chosen provider.
        temperature: 0.1 recommended — we want deterministic expert opinions.

    Example::

        from src.shared.bridge_validator import BridgeValidator
        v = BridgeValidator()
        r = v.validate("slam_rate=0.22 over 60 declared boards")
        print(r)
    """

    _DEFAULT_MODELS: dict[str, str] = {
        "gemini":    "gemini-2.0-flash-exp",
        "anthropic": "claude-haiku-4-5",
        "openai":    "gpt-4o-mini",
    }

    def __init__(
        self,
        provider: str = "gemini",
        model: str | None = None,
        temperature: float = 0.1,
    ) -> None:
        if provider not in self._DEFAULT_MODELS:
            raise ValueError(
                f"Unknown provider {provider!r}. "
                f"Choose from: {list(self._DEFAULT_MODELS)}"
            )
        self.provider = provider
        self.model = model or self._DEFAULT_MODELS[provider]
        self.temperature = temperature
        self._client = self._init_client()

    # ── public interface ───────────────────────────────────────────────────────

    def validate(self, claim: str, context: str = "") -> ValidationResult:
        """Validate a free-form statistical bridge claim.

        Args:
            claim:   What you want checked.  Examples:
                     - "SMITH John has slam_rate=0.22 with 60 declared boards"
                     - "penalty_double_rate=0.40 over 15 bidding boards is unusual"
                     - "nt_rate=0.385 with n=217 boards passes NT Specialist threshold"
            context: Optional extra context (tournament, year, player background).

        Returns:
            ValidationResult with four structured sections + raw JSON.
        """
        message = claim if not context else f"Context: {context}\n\nClaim: {claim}"
        raw = self._call(message)
        result = ValidationResult.from_raw(raw)
        logger.info(
            "bridge_validator claim=%r verdict=%s probability=%s sample_ok=%s",
            claim[:80],
            result.verdict,
            result.probability_label,
            result.sample_size_ok,
        )
        return result

    def validate_profile_assignment(
        self,
        player_name: str,
        profile: str,
        metric_value: float,
        n_boards: int,
        denominator_type: str | None = None,
        pvalue: float | None = None,
    ) -> ValidationResult:
        """Validate a specific NegoPlay profile assignment.

        Convenience wrapper that constructs a well-structured claim string
        from the profile metadata, so you can call it directly from
        ``assign_extreme_profiles()`` output.

        Args:
            player_name:     Player identifier (used in the claim text).
            profile:         One of: "Slam Hunter", "Insurance Player",
                             "NT Specialist", "Fighter".
            metric_value:    Observed rate (e.g. 0.101 for slam_rate).
            n_boards:        Number of qualifying boards (denominator).
            denominator_type: "n_declared" or "n_bidding_boards".
                             Inferred from profile if omitted.
            pvalue:          Binomial p-value from ``assign_extreme_profiles()``.
                             Include when available.

        Returns:
            ValidationResult.

        Raises:
            ValueError: If ``profile`` is not a known NegoPlay profile.
        """
        if profile not in PROFILE_THRESHOLDS:
            raise ValueError(
                f"Unknown profile {profile!r}. Known: {list(PROFILE_THRESHOLDS)}"
            )
        info = PROFILE_THRESHOLDS[profile]
        denom = denominator_type or info["denominator"]
        baseline = POPULATION_BASELINES[info["metric"]]
        ratio = metric_value / baseline if baseline > 0 else float("inf")
        threshold_ratio = info["ratio"]

        pval_str = f"Binomial p-value = {pvalue:.4f}." if pvalue is not None else ""

        claim = (
            f"Player '{player_name}' is assigned to profile '{profile}'. "
            f"Observed {info['metric']} = {metric_value:.4f} over {n_boards} {denom}. "
            f"Population baseline = {baseline:.4f}. "
            f"Observed ratio = {ratio:.2f}× (threshold for this profile = {threshold_ratio}×). "
            f"{pval_str} "
            f"Minimum required boards = {MINIMUM_BOARDS}."
        )
        return self.validate(claim)

    def validate_stage1_results(
        self,
        profile_summary: dict[str, dict],
    ) -> dict[str, ValidationResult]:
        """Validate all profiles from Stage 1 output in one call.

        Args:
            profile_summary: Dict keyed by profile name, each value having:
                ``{"n": int, "metric": float, "n_boards_median": int,
                   "pvalue_median": float}``

        Returns:
            Dict keyed by profile name → ValidationResult.
        """
        results: dict[str, ValidationResult] = {}
        for profile_name, stats in profile_summary.items():
            if profile_name == "Generalist":
                continue   # Generalist is the baseline — nothing to validate
            metric_value = stats.get("metric", 0.0)
            n_boards = stats.get("n_boards_median", stats.get("n", 0))
            pvalue = stats.get("pvalue_median")
            results[profile_name] = self.validate_profile_assignment(
                player_name=f"median {profile_name} player",
                profile=profile_name,
                metric_value=metric_value,
                n_boards=n_boards,
                pvalue=pvalue,
            )
        return results

    # ── internal helpers ───────────────────────────────────────────────────────

    def _init_client(self):
        """Initialise the provider-specific API client."""
        if self.provider == "gemini":
            try:
                import google.generativeai as genai
            except ImportError as exc:
                raise ImportError(
                    "google-generativeai is not installed. "
                    "Run: pip install google-generativeai"
                ) from exc
            key = os.getenv("GOOGLE_API_KEY")
            if not key:
                raise EnvironmentError(
                    "GOOGLE_API_KEY not found. Set it in .env or environment."
                )
            genai.configure(api_key=key)
            return genai.GenerativeModel(
                model_name=self.model,
                system_instruction=_SYSTEM_PROMPT,
                generation_config={
                    "response_mime_type": "application/json",
                    "temperature": self.temperature,
                },
            )

        elif self.provider == "anthropic":
            try:
                import anthropic
            except ImportError as exc:
                raise ImportError(
                    "anthropic is not installed. Run: pip install anthropic"
                ) from exc
            key = os.getenv("ANTHROPIC_API_KEY")
            if not key:
                raise EnvironmentError("ANTHROPIC_API_KEY not found.")
            return anthropic.Anthropic(api_key=key)

        elif self.provider == "openai":
            try:
                import openai
            except ImportError as exc:
                raise ImportError(
                    "openai is not installed. Run: pip install openai"
                ) from exc
            key = os.getenv("OPENAI_API_KEY")
            if not key:
                raise EnvironmentError("OPENAI_API_KEY not found.")
            return openai.OpenAI(api_key=key)

    def _call(self, message: str) -> dict:
        """Send message to LLM, return parsed JSON dict."""
        try:
            if self.provider == "gemini":
                response = self._client.generate_content(message)
                raw_text = response.text

            elif self.provider == "anthropic":
                import anthropic
                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    system=_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": message}],
                )
                raw_text = response.content[0].text

            elif self.provider == "openai":
                response = self._client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": message},
                    ],
                )
                raw_text = response.choices[0].message.content

        except Exception as exc:
            logger.error("BridgeValidator API error (%s): %s", self.provider, exc)
            raise

        try:
            return json.loads(raw_text)
        except json.JSONDecodeError as exc:
            logger.error("BridgeValidator JSON parse failed: %s\nraw: %s", exc, raw_text[:500])
            raise RuntimeError(
                f"LLM returned non-JSON response. "
                f"First 200 chars: {raw_text[:200]!r}"
            ) from exc


# ── CLI entry point ────────────────────────────────────────────────────────────

def _cli() -> None:
    """
    CLI: python -m src.shared.bridge_validator "<claim>"

    Env vars:
      BRIDGE_VALIDATOR_PROVIDER  gemini (default) | anthropic | openai
      GOOGLE_API_KEY / ANTHROPIC_API_KEY / OPENAI_API_KEY
    """
    # Try to load .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    if len(sys.argv) < 2:
        print("Bridge Expert Validation Skill — NegoPlay")
        print()
        print("Usage:")
        print("  python -m src.shared.bridge_validator '<claim>'")
        print()
        print("Examples:")
        print("  python -m src.shared.bridge_validator 'slam_rate=0.22 over 60 declared boards'")
        print("  python -m src.shared.bridge_validator 'penalty_double_rate=0.131 over 89 boards — is Fighter profile justified?'")
        print("  python -m src.shared.bridge_validator 'nt_rate=0.385 with n=217 boards'")
        print()
        print("Set BRIDGE_VALIDATOR_PROVIDER=anthropic to use Claude instead of Gemini.")
        sys.exit(0)

    provider = os.getenv("BRIDGE_VALIDATOR_PROVIDER", "gemini")
    claim = " ".join(sys.argv[1:])

    print(f"🔍  Bridge Expert Validator  (provider={provider}, model={BridgeValidator._DEFAULT_MODELS[provider]})")
    print(f"    Claim: {claim!r}")
    print()

    validator = BridgeValidator(provider=provider)
    result = validator.validate(claim)

    print(str(result))
    print()
    print("─" * 60)
    print("Full JSON:")
    print(json.dumps(result.raw_json, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _cli()
