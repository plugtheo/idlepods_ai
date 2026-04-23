"""
Context Service contracts
=========================
Shapes used when Orchestration Service calls the Context Service.

The Context Service performs two jobs per request:
  1. Semantic search over past experiences (RAG/few-shot retrieval).
  2. Repo file scanning for code-related intents.

Both results are bundled into `BuiltContext` and returned to the caller.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class ContextRequest(BaseModel):
    """Request sent from Orchestration → Context Service."""

    prompt: str = Field(description="Raw user prompt to enrich.")
    intent: str = Field(
        description="Classified intent from the QueryRouter, e.g. 'coding'."
    )
    complexity: str = Field(
        description="Classified complexity from the QueryRouter: 'simple', 'moderate', or 'complex'."
    )
    session_id: Optional[str] = Field(
        default=None, description="Opaque string for log correlation."
    )


class FewShotExample(BaseModel):
    """One past experience retrieved by semantic search."""

    problem: str = Field(description="The original user problem.")
    solution: str = Field(description="The high-quality solution that was produced.")
    score: float = Field(description="Convergence quality score (0–1).")
    category: str = Field(description="Capability category, e.g. 'coding'.")


class RepoSnippet(BaseModel):
    """One relevant code snippet found in the local repository."""

    file: str = Field(description="Relative file path of the snippet source.")
    snippet: str = Field(description="Up to 300 chars of the most relevant section.")
    relevance: float = Field(
        description="Relevance score to the prompt (0–1, higher = more relevant)."
    )


class BuiltContext(BaseModel):
    """Enriched context returned from Context Service → Orchestration."""

    few_shots: List[FewShotExample] = Field(
        default_factory=list,
        description="Top-4 semantically similar past solutions.",
    )
    repo_snippets: List[RepoSnippet] = Field(
        default_factory=list,
        description="Relevant code snippets from the local repo (code-related intents only).",
    )
    system_hints: str = Field(
        default="",
        description="Any guidance string to inject into every agent system prompt this request.",
    )
