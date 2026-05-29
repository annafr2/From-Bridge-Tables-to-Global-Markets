"""
src/stage2_skills/extractor.py
================================
Extract behavioural skills from bridge games using an LLM.

Given a PlayerChunk (a batch of 25 boards from one player), this module:
  1. Builds a structured prompt asking the LLM to characterise the player
  2. Sends the chunk to Gemini Flash 2.5 with a JSON response_schema
  3. Returns a typed ChunkExtraction object

The LLM is prompted to act as a bridge expert reviewing the hands and
identifying 3–5 *observable* behavioural skills — not personality traits.
Each skill includes evidence (which boards demonstrate it).

This file is the only place that constructs Stage 2 prompts. The prompt
template is centralised here and exported for documentation.

Usage:
    from src.shared.llm_client import LLMClient
    from src.stage2_skills.extractor import extract_skills_from_chunk

    client = LLMClient()
    extraction = extract_skills_from_chunk(client, chunk)
    for skill in extraction.skills:
        print(f"- {skill['name']}: {skill['description']}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from src.shared.llm_client import LLMClient, LLMResponse
from src.stage2_skills.chunker import PlayerChunk

logger = logging.getLogger(__name__)


# ── System prompt (the LLM's role) ────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an elite bridge expert reviewing the playing style of a specific player.
Your job is to identify the player's behavioural skills based purely on the
hands they played.

You must:
- Focus on OBSERVABLE bridge behaviour (bidding patterns, risk-taking,
  competitive choices, conservatism, system preferences)
- Use bridge terminology precisely (preempt, sacrifice, balancing, invitational,
  splinter, control bid, etc.)
- NOT invent personality traits — only describe what the bidding shows
- NOT assume information not visible in the hands (no demographic guessing)
- Cite evidence by board number when possible

Each skill must be:
1. Concise: 3-7 words for the skill name
2. Falsifiable: another bridge expert could verify it from the same hands
3. Behavioural: an action the player tends to take, not a label like "smart"

Respond as structured JSON only. No extra commentary.
"""


# ── JSON response schema ──────────────────────────────────────────────────────

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "skills": {
            "type": "array",
            "minItems": 3,
            "maxItems": 5,
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Short skill name (3-7 words)",
                    },
                    "description": {
                        "type": "string",
                        "description": "One-sentence behavioural description",
                    },
                    "evidence_boards": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Board numbers that demonstrate this skill",
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                    },
                },
                "required": ["name", "description", "evidence_boards", "confidence"],
            },
        },
        "summary": {
            "type": "string",
            "description": "2-3 sentence overall characterisation of the player",
        },
    },
    "required": ["skills", "summary"],
}


# ── Output container ──────────────────────────────────────────────────────────

@dataclass
class ChunkExtraction:
    """Result of extracting skills from one chunk."""

    player_name: str
    profile: str | None
    n_boards: int
    skills: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    cost_usd: float = 0.0
    latency_sec: float = 0.0
    model: str = ""
    raw_text: str = ""
    error: str | None = None


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_user_prompt(chunk: PlayerChunk) -> str:
    """Build the user-side prompt for one chunk."""
    body = chunk.to_prompt()
    instructions = (
        "\n\n" + "=" * 60 + "\n\n"
        "Based on the boards above, identify 3-5 distinct behavioural skills "
        "this player demonstrates. For each skill, cite specific board numbers "
        "as evidence and rate your confidence.\n\n"
        "Then write a 2-3 sentence summary of their overall playing style.\n"
    )
    return body + instructions


# ── Main extractor ────────────────────────────────────────────────────────────

def extract_skills_from_chunk(
    client: LLMClient,
    chunk: PlayerChunk,
    temperature: float = 0.3,
) -> ChunkExtraction:
    """Run skill extraction on one chunk.

    Args:
        client:      Configured LLMClient (any provider).
        chunk:       PlayerChunk from build_player_chunks().
        temperature: 0.3 is recommended (we want consistent classification,
                     not creative writing).

    Returns:
        ChunkExtraction with parsed skills + cost + metadata.
        If the LLM call or parsing fails, returns an extraction with
        `error` set and `skills=[]`.
    """
    user_prompt = build_user_prompt(chunk)

    try:
        response: LLMResponse = client.generate(
            system=SYSTEM_PROMPT,
            user=user_prompt,
            purpose="skill_extraction",
            temperature=temperature,
            response_schema=RESPONSE_SCHEMA,
        )
    except Exception as e:  # noqa: BLE001 — log and return error extraction
        logger.error("Skill extraction failed for %s: %s", chunk.player_name, e)
        return ChunkExtraction(
            player_name=chunk.player_name,
            profile=chunk.profile,
            n_boards=len(chunk.boards),
            error=str(e),
        )

    if response.json is None or not isinstance(response.json, dict):
        logger.warning(
            "Extraction for %s returned non-dict JSON; falling back to empty",
            chunk.player_name,
        )
        return ChunkExtraction(
            player_name=chunk.player_name,
            profile=chunk.profile,
            n_boards=len(chunk.boards),
            cost_usd=response.usage.cost_usd,
            latency_sec=response.latency_sec,
            model=response.model,
            raw_text=response.text,
            error="non-dict JSON response",
        )

    skills_raw = response.json.get("skills", [])
    if not isinstance(skills_raw, list):
        skills_raw = []

    summary = response.json.get("summary", "")
    if not isinstance(summary, str):
        summary = ""

    return ChunkExtraction(
        player_name=chunk.player_name,
        profile=chunk.profile,
        n_boards=len(chunk.boards),
        skills=skills_raw,
        summary=summary,
        cost_usd=response.usage.cost_usd,
        latency_sec=response.latency_sec,
        model=response.model,
        raw_text=response.text,
    )


__all__ = [
    "SYSTEM_PROMPT",
    "RESPONSE_SCHEMA",
    "ChunkExtraction",
    "build_user_prompt",
    "extract_skills_from_chunk",
]
