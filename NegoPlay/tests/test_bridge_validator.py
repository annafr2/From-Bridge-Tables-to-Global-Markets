"""
tests/test_bridge_validator.py
==============================
Tests for the Bridge Expert Validation Skill.

All LLM calls are mocked via an injected mock LLMClient — no API key, no
network, no provider SDK needed. (Migrated May 2026 from mocking the deprecated
google.generativeai package to mocking the shared LLMClient.)
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from src.shared.bridge_validator import (
    BridgeValidator,
    ValidationResult,
    _extract_json,
    validate_profile_assignment,
    validate_stage1_results,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mock_client(payload: dict | None = None, text: str | None = None):
    """Build a mock LLMClient whose generate() returns a fixed response.

    Args:
        payload: dict to expose as response.json (structured output path).
        text:    raw text to expose as response.text (fallback path). If
                 payload is None, response.json is set to None so the validator
                 falls back to parsing text.
    """
    client = MagicMock()
    resp = MagicMock()
    resp.json = payload
    resp.text = text if text is not None else (json.dumps(payload) if payload else "")
    client.generate.return_value = resp
    client.model = "gemini-2.0-flash"
    return client


_OK_PAYLOAD = {
    "legality_check": "Possible.",
    "probability_assessment": "PLAUSIBLE",
    "expert_analysis": "Reasonable.",
    "verdict": "ACCEPT",
    "recommended_action": "Proceed.",
}


# ── ValidationResult dataclass ────────────────────────────────────────────────

class TestValidationResult:
    def test_is_valid_accept(self):
        assert ValidationResult("", "", "", "ACCEPT", "").is_valid is True

    def test_is_valid_accept_with_caveat(self):
        assert ValidationResult("", "", "", "ACCEPT_WITH_CAVEAT", "").is_valid is True

    def test_is_valid_reject(self):
        assert ValidationResult("", "", "", "REJECT", "").is_valid is False

    def test_is_valid_error(self):
        assert ValidationResult("", "", "", "ERROR", "").is_valid is False


# ── _extract_json helper ──────────────────────────────────────────────────────

class TestExtractJson:
    def test_plain_json(self):
        assert _extract_json('{"verdict": "ACCEPT"}')["verdict"] == "ACCEPT"

    def test_json_with_fences(self):
        assert _extract_json('```json\n{"verdict": "REJECT"}\n```')["verdict"] == "REJECT"

    def test_json_with_surrounding_text(self):
        text = 'Here is my assessment:\n{"verdict": "ACCEPT"}\nDone.'
        assert _extract_json(text)["verdict"] == "ACCEPT"

    def test_no_json_raises(self):
        with pytest.raises(ValueError):
            _extract_json("no json here")


# ── BridgeValidator construction ──────────────────────────────────────────────

class TestBridgeValidatorConstruction:
    def test_missing_api_key_raises(self, monkeypatch):
        """With no injected client and no key, LLMClient construction raises."""
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with pytest.raises(ValueError):
            BridgeValidator()

    def test_injected_client_is_used(self):
        client = _mock_client(_OK_PAYLOAD)
        v = BridgeValidator(client=client)
        assert v.client is client

    def test_default_model_assigned(self):
        client = _mock_client(_OK_PAYLOAD)
        v = BridgeValidator(client=client)
        assert v.model == "gemini-2.0-flash"

    def test_custom_model_overrides_default(self):
        client = _mock_client(_OK_PAYLOAD)
        v = BridgeValidator(model="gemini-2.5-pro", client=client)
        assert v.model == "gemini-2.5-pro"


# ── BridgeValidator.validate ──────────────────────────────────────────────────

class TestBridgeValidatorValidate:
    def test_validate_returns_validation_result(self):
        v = BridgeValidator(client=_mock_client(_OK_PAYLOAD))
        result = v.validate("slam_rate=0.10 over 200 boards")
        assert isinstance(result, ValidationResult)
        assert result.verdict == "ACCEPT"
        assert result.is_valid is True

    def test_validate_plausible_slam_rate(self):
        v = BridgeValidator(client=_mock_client(_OK_PAYLOAD))
        result = v.validate("slam_rate=0.10")
        assert result.probability_assessment == "PLAUSIBLE"

    def test_validate_passes_system_prompt_and_purpose(self):
        client = _mock_client(_OK_PAYLOAD)
        BridgeValidator(client=client).validate("some claim")
        _, kwargs = client.generate.call_args
        assert "Bridge Expert Validator" in kwargs["system"]
        assert kwargs["purpose"] == "bridge_validation"
        assert "some claim" in kwargs["user"]

    def test_validate_falls_back_to_text_when_no_parsed_json(self):
        # response.json is None → validator must parse response.text
        client = _mock_client(payload=None, text=json.dumps(_OK_PAYLOAD))
        result = BridgeValidator(client=client).validate("claim")
        assert result.verdict == "ACCEPT"

    def test_validate_non_json_response_returns_error(self):
        client = _mock_client(payload=None, text="This is not JSON at all")
        result = BridgeValidator(client=client).validate("some claim")
        assert result.verdict == "ERROR"
        assert result.error is not None

    def test_validate_llm_exception_returns_error(self):
        client = MagicMock()
        client.model = "gemini-2.0-flash"
        client.generate.side_effect = RuntimeError("network down")
        result = BridgeValidator(client=client).validate("claim")
        assert result.verdict == "ERROR"
        assert "network down" in (result.error or "")


# ── module-level convenience functions ────────────────────────────────────────

class TestValidateProfileAssignment:
    def test_slam_hunter_valid(self):
        client = _mock_client(_OK_PAYLOAD)
        result = validate_profile_assignment("BRINK", "Slam Hunter", 0.10, 200, client=client)
        assert result.verdict == "ACCEPT"

    def test_fighter_assignment(self):
        client = _mock_client(_OK_PAYLOAD)
        result = validate_profile_assignment("NAWROCKI", "Fighter", 0.13, 150, client=client)
        assert result.verdict == "ACCEPT"

    def test_claim_includes_rate_and_boards(self):
        client = _mock_client(_OK_PAYLOAD)
        validate_profile_assignment("X", "Slam Hunter", 0.101, 216, client=client)
        _, kwargs = client.generate.call_args
        assert "0.101" in kwargs["user"]
        assert "216" in kwargs["user"]


class TestValidateStage1Results:
    def test_returns_dict_keyed_by_profile(self):
        client = _mock_client(_OK_PAYLOAD)
        results = validate_stage1_results(
            {"Slam Hunter": "rate=0.10, n=200", "Fighter": "rate=0.13, n=150"},
            client=client,
        )
        assert set(results.keys()) == {"Slam Hunter", "Fighter"}

    def test_each_result_is_validation_result(self):
        client = _mock_client(_OK_PAYLOAD)
        results = validate_stage1_results({"Slam Hunter": "rate=0.10"}, client=client)
        assert all(isinstance(r, ValidationResult) for r in results.values())
