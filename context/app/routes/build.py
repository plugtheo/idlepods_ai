"""
Context build route — POST /v1/context/build
=============================================
Receives a ContextRequest from the Orchestration Service, runs few-shot
retrieval and (for code-related intents) repo scanning concurrently,
and returns a BuiltContext.

A 200-ms deadline is enforced inside the Orchestration Service via its own
timeout wrapper — this service does not impose an internal deadline.  If
ChromaDB or the repo scan raises an exception the response degrades
gracefully to an empty context rather than propagating an error.

Health check — GET /health
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter

from shared.contracts.context import BuiltContext, ContextRequest
from ..retrieval.few_shot import retrieve_few_shots
from ..retrieval.repo import retrieve_repo_snippets

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/v1/context/build",
    response_model=BuiltContext,
    summary="Build enriched context for one request",
)
async def build_context(request: ContextRequest) -> BuiltContext:
    """
    Run few-shot retrieval and repo scanning concurrently.

    Always returns 200 with a valid BuiltContext — degraded to empty
    lists on any retrieval error so the Orchestration Service can proceed
    without enrichment rather than failing the user request.
    """
    results = await asyncio.gather(
        retrieve_few_shots(request.prompt),
        retrieve_repo_snippets(request.prompt, request.intent),
        return_exceptions=True,
    )
    few_shots = results[0] if not isinstance(results[0], BaseException) else []
    repo_snippets = results[1] if not isinstance(results[1], BaseException) else []
    if isinstance(results[0], BaseException):
        logger.error("Few-shot retrieval raised unexpectedly: %s", results[0])
    if isinstance(results[1], BaseException):
        logger.error("Repo retrieval raised unexpectedly: %s", results[1])

    hints = _build_hints(request.intent, request.complexity)

    return BuiltContext(
        few_shots=few_shots,
        repo_snippets=repo_snippets,
        system_hints=hints,
    )


@router.get("/health", summary="Health check")
async def health() -> dict:
    return {"status": "ok", "service": "context"}


# ── Helpers ──────────────────────────────────────────────────────────────


def _build_hints(intent: str, complexity: str) -> str:
    """Generate brief guidance text for the agents based on intent and complexity."""
    hints: list[str] = []

    if intent == "coding":
        hints.append("Write well-structured, production-quality code with clear naming.")
    elif intent == "debugging":
        hints.append("Identify the root cause before proposing a fix. Show the fix clearly.")
    elif intent == "research":
        hints.append("Cite sources or reasoning. Be accurate and concise.")
    elif intent == "planning":
        hints.append("Break the task into concrete, actionable steps.")
    elif intent == "analysis":
        hints.append("Be systematic. Support conclusions with evidence from the data or code.")

    if complexity == "complex":
        hints.append("Consider edge cases, scalability, and maintainability.")
    elif complexity == "simple":
        hints.append("Keep the response focused and concise.")

    return "  ".join(hints)
