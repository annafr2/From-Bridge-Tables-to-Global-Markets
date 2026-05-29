"""
tests/test_bridge_validator.py
Tests for the Bridge Expert Validation Skill.

All LLM calls are mocked — these tests verify the parsing logic,
data constants, and the CLI behaviour WITHOUT making real API calls.

Run:
    pytest tests/test_bridge_validator.py -v --no-cov
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.shared.bridge_validator import (
    MINIMUM_BOARDS,
    POPULATION_BASELINES,
    PROFILE_THRESHOLDS,
    BridgeValidator,
    ValidationResult,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_llm_json(
    verdict: str = "ACCEPT",
    probability: str = "PLAUSIBLE",
    sample_ok: bool = True,
    is_legal: bool = True,
) -> dict:
    """Return a minimal well-formed LLM response dict."""
    return {
        "legality_check": {
            "is_legal": is_legal,
            "issue": None,
            "details": "Rate is within physical range for bridge.",
        },
        "probability_assessment": {
            "verdict": probability,
            "percentile_estimate": "top 5% of all players",
            "sample_size_adequate": sample_ok,
            "sample_size_comment": "n=100 is above the 50-board minimum.",
            "ci_note": "[7%, 14%] approximate 95% CI for rate=0.10 with n=100",
        },
        "expert_analysis": {
            "bridge_plausibility": "Consistent with elite-level play.",
            "statistical_concern": "",
            "alternative_explanation": "No obvious confound.",
            "negoplay_impact": "Profile assignment looks robust.",
        },
        "final_verdict": {
            "label": verdict,
            "one_liner": "Slam rate is elevated but statistically credible.",
            "recommended_action": "Proceed as-is.",
        },
    }


@pytest.fixture
def mock_gemini_response():
    """Patch the Gemini GenerativeModel so no real API call is made."""
    mock_response = MagicMock()
    mock_response.text = json.dumps(_make_llm_json())

    mock_model_instance = MagicMock()
    mock_model_instance.generate_content.return_value = mock_response

    with patch("google.generativeai.GenerativeModel", return_value=mock_model_instance), \
         patch("google.generativeai.configure"):
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key-for-tests"}):
            yield mock_model_instance


# ── Constants sanity checks ────────────────────────────────────────────────────

class TestConstants:
    def test_population_baselines_in_range(self):
        for metric, rate in POPULATION_BASELINES.items():
            assert 0 < rate < 1, f"{metric} baseline {rate} is out of (0, 1)"

    def test_slam_baseline_is_realistic(self):
        # Elite slam rate should be between 3% and 8%
        assert 0.03 <= POPULATION_BASELINES["slam_rate"] <= 0.08

    def test_partscore_baseline_is_majority(self):
        # Most deals end in partscores
        assert POPULATION_BASELINES["partscore_rate"] > 0.50

    def test_all_profiles_have_threshold(self):
        expected_profiles = {"Slam Hunter", "Insurance Player", "NT Specialist", "Fighter"}
        assert set(PROFILE_THRESHOLDS.keys()) == expected_profiles

    def test_each_profile_has_required_keys(self):
        for profile, info in PROFILE_THRESHOLDS.items():
            assert "metric" in info, f"{profile} missing 'metric'"
            assert "denominator" in info, f"{profile} missing 'denominator'"
            assert "ratio" in info, f"{profile} missing 'ratio'"

    def test_profile_metrics_are_in_baselines(self):
        for profile, info in PROFILE_THRESHOLDS.items():
            assert info["metric"] in POPULATION_BASELINES, \
                f"{profile} metric {info['metric']} not in POPULATION_BASELINES"

    def test_minimum_boards_is_reasonable(self):
        assert MINIMUM_BOARDS >= 50


# ── ValidationResult.from_raw ──────────────────────────────────────────────────

class TestValidationResult:
    def test_from_raw_accept(self):
        data = _make_llm_json(verdict="ACCEPT", probability="PLAUSIBLE", sample_ok=True)
        r = ValidationResult.from_raw(data)
        assert r.verdict == "ACCEPT"
        assert r.probability_label == "PLAUSIBLE"
        assert r.sample_size_ok is True
        assert r.is_legal is True

    def test_from_raw_reject(self):
        data = _make_llm_json(verdict="REJECT", probability="IMPLAUSIBLE", is_legal=False)
        r = ValidationResult.from_raw(data)
        assert r.verdict == "REJECT"
        assert r.is_legal is False

    def test_from_raw_caveat(self):
        data = _make_llm_json(verdict="ACCEPT_WITH_CAVEAT", probability="SUSPICIOUS", sample_ok=False)
        r = ValidationResult.from_raw(data)
        assert r.verdict == "ACCEPT_WITH_CAVEAT"
        assert r.sample_size_ok is False

    def test_str_contains_verdict(self):
        r = ValidationResult.from_raw(_make_llm_json())
        text = str(r)
        assert "ACCEPT" in text

    def test_str_contains_action(self):
        r = ValidationResult.from_raw(_make_llm_json())
        text = str(r)
        assert "Action" in text or "Proceed" in text

    def test_raw_json_preserved(self):
        data = _make_llm_json()
        r = ValidationResult.from_raw(data)
        assert r.raw_json == data

    def test_from_raw_handles_missing_keys_gracefully(self):
        """from_raw must not crash on empty / partial JSON."""
        r = ValidationResult.from_raw({})
        assert r.verdict == "UNKNOWN"
        assert r.probability_label == "UNKNOWN"


# ── BridgeValidator construction ───────────────────────────────────────────────

class TestBridgeValidatorConstruction:
    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            BridgeValidator(provider="cohere")

    def test_default_model_assigned(self, mock_gemini_response):
        v = BridgeValidator(provider="gemini")
        assert v.model == "gemini-2.0-flash-exp"

    def test_custom_model_overrides_default(self, mock_gemini_response):
        v = BridgeValidator(provider="gemini", model="gemini-2.0-pro")
        assert v.model == "gemini-2.0-pro"

    def test_missing_api_key_raises(self, tmp_path, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with patch("google.generativeai.configure"), \
             patch("google.generativeai.GenerativeModel"):
            with pytest.raises(EnvironmentError, match="GOOGLE_API_KEY"):
                BridgeValidator(provider="gemini")


# ── BridgeValidator.validate ───────────────────────────────────────────────────

class TestBridgeValidatorValidate:
    def test_validate_returns_validation_result(self, mock_gemini_response):
        v = BridgeValidator(provider="gemini")
        result = v.validate("slam_rate=0.10 over 100 boards")
        assert isinstance(result, ValidationResult)

    def test_validate_plausible_slam_rate(self, mock_gemini_response):
        v = BridgeValidator(provider="gemini")
        result = v.validate("slam_rate=0.101 over 216 boards")
        assert result.verdict == "ACCEPT"
        assert result.probability_label == "PLAUSIBLE"

    def test_validate_passes_context(self, mock_gemini_response):
        """Context string should be prepended to the user message."""
        v = BridgeValidator(provider="gemini")
        v.validate("slam_rate=0.10 over 50 boards", context="ECB 2024, Open category")
        call_args = mock_gemini_response.generate_content.call_args[0][0]
        assert "ECB 2024" in call_args
        assert "slam_rate=0.10" in call_args

    def test_validate_non_json_response_raises(self, mock_gemini_response):
        mock_gemini_response.generate_content.return_value.text = "Not JSON at all"
        v = BridgeValidator(provider="gemini")
        with pytest.raises(RuntimeError, match="non-JSON"):
            v.validate("slam_rate=0.10")


# ── BridgeValidator.validate_profile_assignment ────────────────────────────────

class TestValidateProfileAssignment:
    def test_slam_hunter_valid(self, mock_gemini_response):
        v = BridgeValidator(provider="gemini")
        result = v.validate_profile_assignment(
            player_name="SMITH John",
            profile="Slam Hunter",
            metric_value=0.101,
            n_boards=216,
            pvalue=0.003,
        )
        assert isinstance(result, ValidationResult)
        # The claim sent to the LLM should include all our metadata
        call_args = mock_gemini_response.generate_content.call_args[0][0]
        assert "SMITH John" in call_args
        assert "Slam Hunter" in call_args
        assert "0.1010" in call_args
        assert "216" in call_args
        assert "0.0030" in call_args

    def test_fighter_uses_bidding_boards_denominator(self, mock_gemini_response):
        v = BridgeValidator(provider="gemini")
        v.validate_profile_assignment(
            player_name="JONES Bob",
            profile="Fighter",
            metric_value=0.131,
            n_boards=89,
        )
        call_args = mock_gemini_response.generate_content.call_args[0][0]
        assert "n_bidding_boards" in call_args

    def test_slam_hunter_uses_declared_boards_denominator(self, mock_gemini_response):
        v = BridgeValidator(provider="gemini")
        v.validate_profile_assignment(
            player_name="SMITH Anna",
            profile="Slam Hunter",
            metric_value=0.101,
            n_boards=150,
        )
        call_args = mock_gemini_response.generate_content.call_args[0][0]
        assert "n_declared" in call_args

    def test_unknown_profile_raises(self, mock_gemini_response):
        v = BridgeValidator(provider="gemini")
        with pytest.raises(ValueError, match="Unknown profile"):
            v.validate_profile_assignment(
                player_name="X",
                profile="Grand Master",
                metric_value=0.5,
                n_boards=100,
            )

    def test_ratio_included_in_claim(self, mock_gemini_response):
        """The ratio (observed / baseline) must appear in the claim string."""
        v = BridgeValidator(provider="gemini")
        # slam_rate=0.110, baseline=0.055 → ratio=2.00
        v.validate_profile_assignment(
            player_name="X",
            profile="Slam Hunter",
            metric_value=0.110,
            n_boards=100,
        )
        call_args = mock_gemini_response.generate_content.call_args[0][0]
        assert "2.00" in call_args


# ── BridgeValidator.validate_stage1_results ────────────────────────────────────

class TestValidateStage1Results:
    def test_returns_dict_keyed_by_profile(self, mock_gemini_response):
        v = BridgeValidator(provider="gemini")
        summary = {
            "Slam Hunter":      {"metric": 0.101, "n_boards_median": 216, "pvalue_median": 0.003},
            "Fighter":          {"metric": 0.131, "n_boards_median": 89,  "pvalue_median": 0.012},
            "Generalist":       {},   # should be skipped
        }
        results = v.validate_stage1_results(summary)
        assert "Slam Hunter" in results
        assert "Fighter" in results
        assert "Generalist" not in results   # Generalist is baseline, skip

    def test_each_result_is_validation_result(self, mock_gemini_response):
        v = BridgeValidator(provider="gemini")
        summary = {
            "NT Specialist": {"metric": 0.385, "n_boards_median": 130},
        }
        results = v.validate_stage1_results(summary)
        assert isinstance(results["NT Specialist"], ValidationResult)
