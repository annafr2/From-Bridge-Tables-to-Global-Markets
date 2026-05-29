"""
src/shared/llm_client.py
=========================
Unified LLM wrapper — Gemini (default), Claude, OpenAI.

All NegoPlay LLM calls go through this client so the call site doesn't care
which provider answered. Logs every call to results/llm_logs/ with token
counts and cost (essential for the $20 budget alert).

Standard call pattern:

    client = LLMClient()  # defaults to Gemini Flash 2.0
    response = client.generate(
        system="You are a Slam Hunter...",
        user="Here are 25 bridge hands...",
        response_schema={...},  # optional JSON schema
        purpose="profile_extraction",  # for logging
    )

    response.text       # str — raw text
    response.json       # dict — parsed JSON (if response_schema given)
    response.usage      # token counts + cost
    response.provider   # "gemini" | "claude" | "openai"

Design choices:
- Provider abstraction via a single `generate()` method
- Retry with exponential backoff (tenacity-style, hand-rolled to avoid dep)
- All logs go to `results/llm_logs/calls.jsonl` (append-only)
- Pricing table is hardcoded — update when providers change rates
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

# ── Pricing (USD per 1M tokens) — May 2026 ────────────────────────────────────

PRICING: dict[str, dict[str, float]] = {
    # Gemini 2.5 Flash — default (Gemini 2.0 Flash deprecated for new users, May 2026)
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "gemini-flash-latest": {"input": 0.30, "output": 2.50},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    # Legacy (kept for backwards compat — no new users):
    "gemini-2.0-flash": {"input": 0.075, "output": 0.30},
    "gemini-2.0-flash-exp": {"input": 0.075, "output": 0.30},

    # Anthropic Claude — quality-critical or cross-model validation
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-opus-4-5": {"input": 15.00, "output": 75.00},

    # OpenAI — cross-model validation
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}

# ── Default models per provider ───────────────────────────────────────────────

DEFAULT_MODELS: dict[str, str] = {
    "gemini": "gemini-2.5-flash",
    "claude": "claude-3-5-haiku-20241022",
    "openai": "gpt-4o-mini",
}

Provider = Literal["gemini", "claude", "openai"]


# ── Response container ────────────────────────────────────────────────────────

@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class LLMResponse:
    """Standardised response from any provider."""

    text: str
    provider: Provider
    model: str
    usage: TokenUsage
    latency_sec: float
    json: dict | list | None = None  # parsed JSON if response_schema was used
    raw: Any = field(default=None, repr=False)  # provider-specific raw object


# ── Cost helper ───────────────────────────────────────────────────────────────

def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Look up pricing and return USD cost. Returns 0.0 if model unknown."""
    if model not in PRICING:
        logger.warning("Unknown model %s — cost set to 0.0", model)
        return 0.0
    p = PRICING[model]
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000


# ── Call logger ───────────────────────────────────────────────────────────────

class CallLogger:
    """Append-only JSONL log of every LLM call. Also enforces budget cap."""

    def __init__(self, log_path: Path, budget_cap_usd: float = 50.0):
        self.log_path = log_path
        self.budget_cap_usd = budget_cap_usd
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._cumulative_cost: float | None = None

    def cumulative_cost(self) -> float:
        """Sum all costs from the log file (re-read on demand)."""
        if not self.log_path.exists():
            return 0.0
        total = 0.0
        with self.log_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    total += entry.get("cost_usd", 0.0)
                except json.JSONDecodeError:
                    continue
        return total

    def log(self, response: LLMResponse, purpose: str) -> None:
        """Append one call record to the log."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "provider": response.provider,
            "model": response.model,
            "purpose": purpose,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cost_usd": response.usage.cost_usd,
            "latency_sec": round(response.latency_sec, 3),
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        # Budget check
        total = self.cumulative_cost()
        if total > self.budget_cap_usd:
            raise BudgetExceededError(
                f"Cumulative LLM cost ${total:.4f} exceeded "
                f"hard cap ${self.budget_cap_usd:.2f}"
            )
        if total > self.budget_cap_usd * 0.5:
            logger.warning(
                "Budget alert: cumulative cost $%.4f is >50%% of cap $%.2f",
                total, self.budget_cap_usd,
            )


class BudgetExceededError(RuntimeError):
    """Raised when cumulative LLM cost passes the hard cap."""


# ── Provider adapters ─────────────────────────────────────────────────────────

class _GeminiAdapter:
    """Wraps google-genai SDK."""

    def __init__(self, api_key: str):
        from google import genai  # imported lazily so other providers don't need it
        self._client = genai.Client(api_key=api_key)
        self._types = __import__("google.genai.types", fromlist=["GenerateContentConfig"])

    def generate(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float,
        response_schema: dict | None,
        max_retries: int,
    ) -> LLMResponse:
        config: dict[str, Any] = {
            "system_instruction": system,
            "temperature": temperature,
        }
        if response_schema is not None:
            config["response_mime_type"] = "application/json"
            config["response_schema"] = response_schema

        last_err: Exception | None = None
        for attempt in range(max_retries):
            try:
                start = time.time()
                resp = self._client.models.generate_content(
                    model=model,
                    contents=user,
                    config=config,
                )
                latency = time.time() - start

                text = resp.text or ""
                usage_meta = getattr(resp, "usage_metadata", None)
                in_tok = getattr(usage_meta, "prompt_token_count", 0) or 0
                out_tok = getattr(usage_meta, "candidates_token_count", 0) or 0

                parsed: dict | list | None = None
                if response_schema is not None and text:
                    try:
                        parsed = json.loads(text)
                    except json.JSONDecodeError as e:
                        logger.warning("Gemini returned invalid JSON: %s", e)

                return LLMResponse(
                    text=text,
                    provider="gemini",
                    model=model,
                    usage=TokenUsage(
                        input_tokens=in_tok,
                        output_tokens=out_tok,
                        cost_usd=compute_cost(model, in_tok, out_tok),
                    ),
                    latency_sec=latency,
                    json=parsed,
                    raw=resp,
                )
            except Exception as e:  # noqa: BLE001 — provider raises many types
                last_err = e
                wait = 2 ** attempt
                logger.warning(
                    "Gemini call failed (attempt %d/%d): %s — retrying in %ds",
                    attempt + 1, max_retries, e, wait,
                )
                time.sleep(wait)

        raise RuntimeError(f"Gemini call failed after {max_retries} retries") from last_err


class _ClaudeAdapter:
    """Wraps anthropic SDK."""

    def __init__(self, api_key: str):
        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key)

    def generate(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float,
        response_schema: dict | None,
        max_retries: int,
    ) -> LLMResponse:
        # Claude doesn't have native response_schema like Gemini.
        # We append a JSON instruction to the system prompt.
        sys_prompt = system
        if response_schema is not None:
            sys_prompt += (
                "\n\nRespond with valid JSON only, matching this schema: "
                + json.dumps(response_schema)
            )

        last_err: Exception | None = None
        for attempt in range(max_retries):
            try:
                start = time.time()
                resp = self._client.messages.create(
                    model=model,
                    max_tokens=4096,
                    system=sys_prompt,
                    messages=[{"role": "user", "content": user}],
                    temperature=temperature,
                )
                latency = time.time() - start

                text = resp.content[0].text if resp.content else ""
                in_tok = resp.usage.input_tokens
                out_tok = resp.usage.output_tokens

                parsed: dict | list | None = None
                if response_schema is not None and text:
                    try:
                        # Claude sometimes wraps JSON in ```json ... ``` blocks
                        clean = text.strip()
                        if clean.startswith("```"):
                            clean = clean.split("```")[1]
                            if clean.startswith("json"):
                                clean = clean[4:]
                            clean = clean.strip()
                        parsed = json.loads(clean)
                    except json.JSONDecodeError as e:
                        logger.warning("Claude returned invalid JSON: %s", e)

                return LLMResponse(
                    text=text,
                    provider="claude",
                    model=model,
                    usage=TokenUsage(
                        input_tokens=in_tok,
                        output_tokens=out_tok,
                        cost_usd=compute_cost(model, in_tok, out_tok),
                    ),
                    latency_sec=latency,
                    json=parsed,
                    raw=resp,
                )
            except Exception as e:  # noqa: BLE001
                last_err = e
                wait = 2 ** attempt
                logger.warning(
                    "Claude call failed (attempt %d/%d): %s — retrying in %ds",
                    attempt + 1, max_retries, e, wait,
                )
                time.sleep(wait)

        raise RuntimeError(f"Claude call failed after {max_retries} retries") from last_err


class _OpenAIAdapter:
    """Wraps openai SDK."""

    def __init__(self, api_key: str):
        import openai
        self._client = openai.OpenAI(api_key=api_key)

    def generate(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float,
        response_schema: dict | None,
        max_retries: int,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        if response_schema is not None:
            kwargs["response_format"] = {"type": "json_object"}

        last_err: Exception | None = None
        for attempt in range(max_retries):
            try:
                start = time.time()
                resp = self._client.chat.completions.create(**kwargs)
                latency = time.time() - start

                text = resp.choices[0].message.content or ""
                in_tok = resp.usage.prompt_tokens
                out_tok = resp.usage.completion_tokens

                parsed: dict | list | None = None
                if response_schema is not None and text:
                    try:
                        parsed = json.loads(text)
                    except json.JSONDecodeError as e:
                        logger.warning("OpenAI returned invalid JSON: %s", e)

                return LLMResponse(
                    text=text,
                    provider="openai",
                    model=model,
                    usage=TokenUsage(
                        input_tokens=in_tok,
                        output_tokens=out_tok,
                        cost_usd=compute_cost(model, in_tok, out_tok),
                    ),
                    latency_sec=latency,
                    json=parsed,
                    raw=resp,
                )
            except Exception as e:  # noqa: BLE001
                last_err = e
                wait = 2 ** attempt
                logger.warning(
                    "OpenAI call failed (attempt %d/%d): %s — retrying in %ds",
                    attempt + 1, max_retries, e, wait,
                )
                time.sleep(wait)

        raise RuntimeError(f"OpenAI call failed after {max_retries} retries") from last_err


# ── Public client ─────────────────────────────────────────────────────────────

class LLMClient:
    """Unified LLM client. Default provider is Gemini Flash 2.0.

    Example:
        client = LLMClient()
        resp = client.generate(
            system="You are a bridge expert.",
            user="What is a slam?",
            purpose="quick_query",
        )
        print(resp.text)
        print(f"Cost: ${resp.usage.cost_usd:.6f}")
    """

    def __init__(
        self,
        provider: Provider = "gemini",
        model: str | None = None,
        budget_cap_usd: float | None = None,
        log_dir: str | Path = "results/llm_logs",
    ):
        self.provider: Provider = provider
        self.model = model or DEFAULT_MODELS[provider]

        # Budget cap from env, then arg, then default
        env_cap = os.getenv("BUDGET_CAP_USD")
        if budget_cap_usd is not None:
            cap = budget_cap_usd
        elif env_cap:
            cap = float(env_cap)
        else:
            cap = 50.0

        self.logger = CallLogger(
            log_path=Path(log_dir) / "calls.jsonl",
            budget_cap_usd=cap,
        )
        self._adapter = self._build_adapter(provider)

    def _build_adapter(self, provider: Provider):
        if provider == "gemini":
            key = os.getenv("GOOGLE_API_KEY")
            if not key:
                raise ValueError("GOOGLE_API_KEY is not set in environment")
            return _GeminiAdapter(api_key=key)

        if provider == "claude":
            key = os.getenv("ANTHROPIC_API_KEY")
            if not key:
                raise ValueError("ANTHROPIC_API_KEY is not set in environment")
            return _ClaudeAdapter(api_key=key)

        if provider == "openai":
            key = os.getenv("OPENAI_API_KEY")
            if not key:
                raise ValueError("OPENAI_API_KEY is not set in environment")
            return _OpenAIAdapter(api_key=key)

        raise ValueError(f"Unknown provider: {provider!r}")

    def generate(
        self,
        system: str,
        user: str,
        *,
        purpose: str = "unspecified",
        temperature: float = 0.3,
        response_schema: dict | None = None,
        model: str | None = None,
        max_retries: int = 3,
    ) -> LLMResponse:
        """Run one LLM call. Logs result automatically.

        Args:
            system:           System prompt (provider-agnostic).
            user:             User message / prompt content.
            purpose:          Tag for logs, e.g. "skill_extraction".
            temperature:      0.3 for bridge/structured tasks, 0.7 for creative.
            response_schema:  JSON schema for structured output. If given,
                              `response.json` will be parsed dict/list.
            model:            Override the client's default model.
            max_retries:      Retry attempts on transient failures (default 3).

        Returns:
            LLMResponse with text, usage, cost, and optional parsed JSON.
        """
        m = model or self.model
        response = self._adapter.generate(
            model=m,
            system=system,
            user=user,
            temperature=temperature,
            response_schema=response_schema,
            max_retries=max_retries,
        )
        self.logger.log(response, purpose=purpose)
        return response

    def cumulative_cost(self) -> float:
        """Total USD spent across all logged calls (any provider)."""
        return self.logger.cumulative_cost()


__all__ = [
    "LLMClient",
    "LLMResponse",
    "TokenUsage",
    "BudgetExceededError",
    "PRICING",
    "DEFAULT_MODELS",
    "compute_cost",
]
