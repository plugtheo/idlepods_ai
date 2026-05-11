"""
Evaluator output schemas
========================
Pydantic schemas for the structured JSON output emitted by evaluator agents
(reviewer, critic) under vLLM guided decoding.

Used in:
  1. orchestration/app/graph/nodes.py — passes the JSON Schema to the inference
     backend via GenerateRequest.response_schema, so vLLM constrains generation.
  2. orchestration/app/utils/scoring.py — validates and extracts the score and
     required fields from the JSON output via the helpers below.
  3. training/bootstrap/validate_adapter.py — post-training reviewer/critic
     output validation via the same helpers (single source of truth).
  4. shared/contracts/agent_prompts.py — the reviewer/critic system prompts
     reference these field names; keep them in sync.
"""
from __future__ import annotations

import json
import re
from typing import Dict, Optional, Type

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class ReviewerOutput(BaseModel):
    # extra="forbid" emits additionalProperties: false in the JSON Schema so
    # guided-decoding backends (outlines/xgrammar) compile a closed grammar
    # that strictly enforces the required key set.
    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0.0, le=1.0)
    strengths: list[str]
    issues: list[str]
    suggestions: list[str]


class CriticOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0.0, le=1.0)
    verdict: str
    blockers: list[str]
    improvement: str


EVALUATOR_SCHEMAS: Dict[str, Type[BaseModel]] = {
    "reviewer": ReviewerOutput,
    "critic":   CriticOutput,
}

# ---------------------------------------------------------------------------
# Output-format helpers — single source of truth for parsing/validating
# evaluator output across scoring, validation, and any future consumer.
# Keep these here (next to the schemas) so all readers stay in sync.
# ---------------------------------------------------------------------------

# Prose-fallback markers used by adapters trained before the JSON migration.
SCORE_RE = re.compile(r"\bSCORE\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)

# Legacy uppercase field markers for evaluator prose outputs. Kept so that
# adapters trained on older prose-format data still validate while we phase
# the format out. JSON path is preferred.
EVALUATOR_REQUIRED_PROSE_FIELDS: Dict[str, list[str]] = {
    "reviewer": ["ISSUES", "SUGGESTIONS"],
    "critic":   ["BLOCKERS", "IMPROVEMENT"],
}


def parse_evaluator_output(text: str, role: str) -> Optional[BaseModel]:
    """Parse *text* as JSON and validate against the role's schema.

    Returns the validated Pydantic model on success, or None when *text* is
    not JSON, not a dict, or fails schema validation. Use this for any check
    that needs "is this a well-formed reviewer/critic response?".
    """
    schema_cls = EVALUATOR_SCHEMAS.get(role)
    if schema_cls is None:
        return None
    try:
        obj = json.loads(text, strict=False)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None
    try:
        return schema_cls.model_validate(obj)
    except ValidationError:
        return None


def extract_score(text: str) -> Optional[float]:
    """Extract a numeric score in [0, 1] from *text*.

    Priority:
      1. JSON top-level "score" key (post-guided-decoding output).
      2. Legacy "SCORE: 0.82" / "SCORE: 8/10" prose marker.

    Returns None if no recognisable score is present.
    """
    try:
        obj = json.loads(text, strict=False)
        if isinstance(obj, dict) and "score" in obj:
            value = float(obj["score"])
            if 0.0 <= value <= 1.0:
                return value
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    match = SCORE_RE.search(text)
    if match:
        value = float(match.group(1))
        if 0.0 <= value <= 1.0:
            return value
        if 0 < value <= 10:
            return value / 10.0  # 0–10 scale → 0–1
    return None


def evaluator_required_fields_present(text: str, role: str) -> bool:
    """Return True when *text* satisfies the evaluator contract for *role*.

    JSON path: a successfully schema-validated payload (all required fields
    present, score in range). Prose fallback: legacy uppercase field markers.
    Non-evaluator roles always return True (no contract to check).
    """
    if role not in EVALUATOR_SCHEMAS:
        return True
    if parse_evaluator_output(text, role) is not None:
        return True
    fields = EVALUATOR_REQUIRED_PROSE_FIELDS.get(role, [])
    return bool(fields) and all(
        re.search(rf"(?mi)^{field}:", text) for field in fields
    )
