"""
tests/test_connection.py
Quick smoke test: can we reach the Gemini API with the key in .env?

Uses the modern google-genai SDK via the shared LLMClient (migrated May 2026
from the deprecated google-generativeai package).

Run:
    pytest tests/test_connection.py -v
"""

import importlib.util
import os

import pytest


def test_gemini_api_key_present():
    """Fail fast if the API key is missing."""
    from dotenv import load_dotenv
    load_dotenv()
    key = os.getenv("GOOGLE_API_KEY")
    assert key, "GOOGLE_API_KEY not found in environment / .env"


def test_gemini_connection():
    """Make a tiny real Gemini call (skipped if SDK or key missing)."""
    if importlib.util.find_spec("google.genai") is None:
        pytest.skip("google-genai not installed — run pip install -r requirements.txt")

    from dotenv import load_dotenv
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        pytest.skip("GOOGLE_API_KEY not set")

    from src.shared.llm_client import LLMClient

    client = LLMClient()
    response = client.generate(
        system="You are a test.",
        user="Say OK",
        purpose="connection_smoke_test",
    )
    assert response.text
