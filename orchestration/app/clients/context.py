"""
Context Service HTTP client
=============================
Used during pipeline initialisation to enrich the user prompt with
few-shot examples and repo snippets before the agent loop starts.

The context call is always wrapped in a timeout (settings.context_timeout).
If the Context Service is slow or unavailable the orchestration pipeline
continues with an empty BuiltContext — it never fails a user request.

A single shared httpx.AsyncClient is created at module import and reused
across requests (connection pooling) — avoids a full TCP handshake on the
critical path of every user request.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

from shared.contracts.context import BuiltContext, ContextRequest
from ..config.settings import settings

logger = logging.getLogger(__name__)

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            base_url=settings.context_url,
            timeout=settings.context_timeout,
        )
    return _client


async def fetch_context(request: ContextRequest) -> BuiltContext:
    """
    Call the Context Service and return a BuiltContext.

    Falls back to an empty BuiltContext on timeout or any HTTP error.
    """
    try:
        client = _get_client()
        resp = await client.post("/v1/context/build", json=request.model_dump(mode="json"))
        resp.raise_for_status()
        return BuiltContext.model_validate(resp.json())

    except asyncio.TimeoutError:
        logger.warning(
            "Context Service timed out after %.1fs — proceeding without enrichment.",
            settings.context_timeout,
        )
    except Exception as exc:
        logger.warning(
            "Context Service unavailable (%s) — proceeding without enrichment.", exc
        )

    return BuiltContext()


async def close() -> None:
    """Close the shared client.  Call from app shutdown hook."""
    global _client
    if _client and not _client.is_closed:
        await _client.aclose()
        _client = None
