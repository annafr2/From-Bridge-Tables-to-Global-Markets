"""
src/shared/bridge_validator.py
==============================
Bridge Expert Validation Skill — statistical sanity checker for bridge claims.

This module provides a domain-expert "second opinion" on statistical claims
about bridge player behaviour. It wraps an LLM with a carefully designed system
prompt encoding:
  - The Laws of Duplicate Bridge (auction, scoring, contract structure)
  - NegoPlay population baselines (slam ~5.5%, partscore ~57%, etc.)
  - Sample-size rules (Nezer's n>=50 minimum)
  - Binomial confidence intervals for rare events

All LLM calls go through the shared ``LLMClient`` (src/shared/llm_client.py),
which uses the modern ``google-genai`` SDK. (This module previously used the
deprecated ``google-generativeai`` package directly; it was migrated in
May 2026 to the single-entry-point client.)

Usage:
    validator = BridgeValidator()
    result = validator.validate("slam_rate=0.22 over 60 boards")
    print(result.verdict)   # ACCEPT / ACCEPT_WITH_CAVEAT / REJECT

Or from the command line:
    python -m src.shared.bridge_validator "slam_rate=0.22 over 60 boards"
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from src.shared.llm_client import LLMClient

logger = logging.getLogger(__name__)

# ── Baselines from the v2.1 production run (563 players, >=50 boards) ──────────
BASELINES: dict[str, dict[str, float]] = {
    "slam_rate":           {"baseline": 0.055, "profile_threshold": 0.101},
    "partscore_rate":      {"baseline": 0.570, "profile_threshold": 0.684},
    "nt_rate":             {"baseline": 0.282, "profile_threshold": 0.385},
    "penalty_double_rate": {"baseline": 0.085, "profile_threshold": 0.131},
}

# Minimum boards for a trustworthy rate estimate (Nezer's rule)
MIN_BOARDS_FOR_TRUST = 50

# Default model for validation (cheap, fast — quality is not critical here).
DEFAULT_VALIDATOR_MODEL = "gemini-2.0-flash"

SYSTEM_PROMPT = """You are a Bridge Expert Validator, a domain expert in both \
duplicate bridge and statistics. Your job is to assess whether a statistical \
claim about bridge player behaviour is plausible, given known baselines and \
sample-size constraints.

You know these population baselines (from 563 elite players, >=50 boards each):
- Slam rate: 5.5% average, 10.1% for a "Slam Hunter" profile
- Partscore rate: 57.0% average, 68.4% for an "Insurance Player"
- NT rate: 28.2% average, 38.5% for an "NT Specialist"
- Penalty double rate: 8.5% average, 13.1% for a "Fighter"

Sample-size rule: rates from fewer than 50 boards are unreliable for rare \
events (slam, penalty double). Always flag small samples.

Respond ONLY with valid JSON matching this schema:
{
  "legality_check": "<is this rate physically possible? 1-2 sentences>",
  "probability_assessment": "<PLAUSIBLE | SUSPICIOUS | IMPLAUSIBLE>",
  "expert_analysis": "<2-4 sentences of bridge-domain reasoning>",
  "verdict": "<ACCEPT | ACCEPT_WITH_CAVEAT | REJECT>",
  "recommended_action": "<what the researcher should do next>"
}
"""

# Structured-output schema (provider-agnostic; enforced via LLMClient).
VALIDATION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "legality_check": {"type": "string"},
        "probability_assessment": {"type": "string"},
        "expert_analysis": {"type": "string"},
        "verdict": {"type": "string"},
        "recommended_action": {"type": "string"},
    },
    "required": [
        "legality_check",
        "probability_assessment",
        "expert_analysis",
        "verdict",
        "recommended_action",
    ],
}


@dataclass
class ValidationResult:
    """Structured output from a bridge validation check."""

    legality_check: str
    probability_assessment: str
    expert_analysis: str
    verdict: str
    recommended_action: str
    raw_response: str = ""
    error: str | None = None

    @property
    def is_valid(self) -> bool:
        """True if the verdict is ACCEPT or ACCEPT_WITH_CAVEAT."""
        return self.verdict in ("ACCEPT", "ACCEPT_WITH_CAVEAT")


def _extract_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from a text blob (handles ``` fences)."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = lines[1:]  # drop opening fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines)

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in response: {text[:100]}")
    return json.loads(cleaned[start : end + 1])


def _error_result(stage: str, exc: Exception) -> ValidationResult:
    """Build an ERROR ValidationResult with a helpful message."""
    return ValidationResult(
        legality_check="",
        probability_assessment="ERROR",
        expert_analysis=f"{stage}: {exc}",
        verdict="ERROR",
        recommended_action="Retry or check API key/quota.",
        error=str(exc),
    )


class BridgeValidator:
    """LLM-backed bridge claim validator.

    All calls route through the shared LLMClient (google-genai under the hood).
    Inject a pre-built ``client`` to share a budget/cost log, or to supply a
    mock in tests.
    """

    def __init__(
        self,
        provider: str = "gemini",
        model: str | None = None,
        client: LLMClient | None = None,
    ):
        self.provider = provider
        if client is not None:
            self.client = client
            self.model = model or getattr(client, "model", None)
        else:
            # LLMClient resolves the API key from the environment and raises a
            # clear ValueError if it is missing.
            self.model = model or DEFAULT_VALIDATOR_MODEL
            self.client = LLMClient(provider=provider, model=self.model)

    def validate(self, claim: str) -> ValidationResult:
        """Validate a single bridge claim and return a structured result."""
        user_prompt = (
            f"Assess this bridge statistical claim:\n\n{claim}\n\n"
            "Respond with the JSON schema specified."
        )

        try:
            response = self.client.generate(
                system=SYSTEM_PROMPT,
                user=user_prompt,
                purpose="bridge_validation",
                temperature=0.3,
                response_schema=VALIDATION_SCHEMA,
            )
        except Exception as e:  # noqa: BLE001 — provider raises many types
            logger.error("Validation LLM call failed: %s", e)
            return _error_result("LLM call failed", e)

        raw = response.text or ""

        # Prefer adapter-parsed JSON; fall back to manual extraction.
        try:
            data = response.json if isinstance(response.json, dict) else _extract_json(raw)
        except (ValueError, json.JSONDecodeError) as e:
            return _error_result("Could not parse LLM response", e)

        return ValidationResult(
            legality_check=data.get("legality_check", ""),
            probability_assessment=data.get("probability_assessment", ""),
            expert_analysis=data.get("expert_analysis", ""),
            verdict=data.get("verdict", ""),
            recommended_action=data.get("recommended_action", ""),
            raw_response=raw,
        )


def validate_profile_assignment(
    player_name: str,
    profile: str,
    rate: float,
    n_boards: int,
    provider: str = "gemini",
    client: LLMClient | None = None,
) -> ValidationResult:
    """Convenience: validate that a player's rate justifies a profile label."""
    claim = (
        f"Player '{player_name}' assigned profile '{profile}' "
        f"based on rate={rate:.3f} over {n_boards} boards."
    )
    validator = BridgeValidator(provider=provider, client=client)
    return validator.validate(claim)


def validate_stage1_results(
    summary: dict[str, Any],
    provider: str = "gemini",
    client: LLMClient | None = None,
) -> dict[str, ValidationResult]:
    """Validate a batch of Stage 1 profile assignments.

    A single validator (and thus a single LLMClient) is reused across all
    profiles so cost is logged together.
    """
    validator = BridgeValidator(provider=provider, client=client)
    results: dict[str, ValidationResult] = {}
    for profile, info in summary.items():
        claim = f"Profile '{profile}': {info}"
        results[profile] = validator.validate(claim)
    return results


def main() -> None:
    """CLI entry point."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.shared.bridge_validator '<claim>'")
        sys.exit(1)

    claim = " ".join(sys.argv[1:])
    validator = BridgeValidator()
    result = validator.validate(claim)
    print(json.dumps(result.__dict__, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
