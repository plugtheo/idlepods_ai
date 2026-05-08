"""
Evaluator output schemas
========================
Pydantic schemas for the structured JSON output emitted by evaluator agents
(reviewer, critic) under vLLM guided decoding.

Used in three places:
  1. orchestration/app/graph/nodes.py — passes the JSON Schema to the inference
     backend via GenerateRequest.response_schema, so vLLM constrains generation.
  2. orchestration/app/utils/scoring.py — validates and extracts the score and
     required fields from the JSON output.
  3. shared/contracts/agent_prompts.py — the reviewer/critic system prompts
     reference these field names; keep them in sync.
"""

from typing import Dict, Type

from pydantic import BaseModel, ConfigDict, Field


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
