"""
Router output schema
====================
Pydantic schema for the structured JSON emitted by the LLM query router under
vLLM guided decoding.  The intent/complexity vocabularies must stay aligned
with shared/routing/query_router.py:Intent and Complexity.
"""

from typing import Literal

from pydantic import BaseModel, Field


class RouteClassification(BaseModel):
    intent: Literal[
        "coding", "debugging", "research", "analysis",
        "planning", "qa", "general",
    ]
    complexity: Literal["simple", "moderate", "complex"]
    confidence: float = Field(ge=0.0, le=1.0)
