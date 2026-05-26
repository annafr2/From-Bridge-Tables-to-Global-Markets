"""
Smoke test — verify Gemini API key works.
Run: pytest tests/test_connection.py -v
"""

import os
import pytest


def test_google_api_key_exists() -> None:
    """Check that the API key is present in the environment."""
    from dotenv import load_dotenv
    load_dotenv()
    key = os.getenv("GOOGLE_API_KEY")
    assert key is not None, "GOOGLE_API_KEY not found in .env"
    assert len(key) > 10, "GOOGLE_API_KEY looks too short — is it valid?"


def test_gemini_connection() -> None:
    """Make one minimal Gemini call to verify connectivity."""
    from dotenv import load_dotenv
    load_dotenv()

    try:
        import google.generativeai as genai
    except ImportError:
        pytest.skip("google-generativeai not installed — run pip install -r requirements.txt")

    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-2.0-flash-exp")
    response = model.generate_content("Reply with exactly: OK")
    assert response.text is not None
    assert len(response.text) > 0
    print(f"\nGemini response: {response.text.strip()}")
