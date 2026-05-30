"""
src/stage3_agents/base_agent.py
===============================
The shared skeleton every NegoPlay agent inherits from.

An "agent" is a profile (e.g. "Slam Hunter") plus an LLM that makes decisions
in character. The base class holds the parts that bridge and negotiation agents
share — the profile signature, the LLM client, the system prompt, and a single
structured-decision helper — so the concrete agents only implement their own
domain method (make_bid / respond_to_offer).

Design notes
------------
- One LLMClient is created per agent but can be SHARED across agents by passing
  an existing client in (keeps a single budget/cost log across a whole game).
- All decisions go through `_decide()`, which calls the LLM with a JSON schema
  and returns the parsed dict. This is the single choke-point for cost logging,
  retries, and JSON parsing — concrete agents never call the LLM directly.
- Temperature default is 0.3 (consistent, low-variance) per CLAUDE.md; the
  negotiation agent overrides to 0.7 for more varied bargaining.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any

from src.shared.llm_client import LLMClient
from src.shared.prompts import ProfileSignature

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base for all profile-conditioned LLM agents.

    Subclasses must:
      - set `self.system_prompt` in __init__ (usually via a prompts.py builder)
      - implement the domain decision method (e.g. make_bid / respond_to_offer)

    Attributes:
        profile:       Profile name, e.g. "Slam Hunter".
        signature:     The ProfileSignature (skills extracted in Stage 2).
        client:        Shared/owned LLMClient used for all decisions.
        temperature:   Sampling temperature for this agent's decisions.
        system_prompt: The character card; set by the subclass.
    """

    def __init__(
        self,
        signature: ProfileSignature,
        client: LLMClient | None = None,
        temperature: float = 0.3,
    ):
        self.signature = signature
        self.profile = signature.profile
        self.client = client or LLMClient()
        self.temperature = temperature
        self.system_prompt: str = ""  # subclass MUST set this

    # ── The single LLM choke-point ────────────────────────────────────────────

    def _decide(
        self,
        user_prompt: str,
        response_schema: dict,
        purpose: str,
    ) -> dict[str, Any]:
        """Make ONE structured decision via the LLM.

        Args:
            user_prompt:     The situation-specific prompt (hand, auction, offer).
            response_schema: JSON schema enforcing the output shape.
            purpose:         Tag for the cost log, e.g. "bridge_bid".

        Returns:
            Parsed JSON dict from the model. If the provider could not return
            parsed JSON, falls back to parsing response.text manually.

        Raises:
            ValueError: if `system_prompt` was never set by the subclass.
        """
        if not self.system_prompt:
            raise ValueError(
                f"{type(self).__name__} did not set system_prompt before _decide()"
            )

        response = self.client.generate(
            system=self.system_prompt,
            user=user_prompt,
            purpose=purpose,
            temperature=self.temperature,
            response_schema=response_schema,
        )

        # Prefer the adapter-parsed JSON; fall back to manual parse of .text.
        if isinstance(response.json, dict):
            return response.json
        try:
            return json.loads(response.text)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning(
                "%s: could not parse JSON from response (%s). Raw: %.120s",
                self.profile, exc, response.text,
            )
            return {}

    # ── Domain method (implemented by subclasses) ─────────────────────────────

    @abstractmethod
    def act(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Make a domain decision. Concrete agents give this a precise signature."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{type(self).__name__} profile={self.profile!r} temp={self.temperature}>"
