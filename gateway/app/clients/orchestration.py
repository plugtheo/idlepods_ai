"""
Orchestration Service client
==============================
Thin async HTTP wrapper used by the Gateway to call `POST /v1/run`
and `POST /v1/run/stream` on the Orchestration Service.

A single shared httpx.AsyncClient is created at module import and
reused across all requests (connection pooling).

Streaming
---------
`stream_pipeline` proxies the orchestration SSE stream line-by-line.
The caller should return a `StreamingResponse` wrapping this generator.
The streaming call uses a separate context-managed client because the
connection must stay open for the duration of the stream.
"""

from __future__ import annotations

import logging
from typing import AsyncIterator

import httpx

from shared.contracts.orchestration import OrchestrationRequest, OrchestrationResponse
from ..config.settings import settings

logger = logging.getLogger(__name__)

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            base_url=settings.orchestration_url,
            timeout=settings.request_timeout,
        )
    return _client


async def run_pipeline(request: OrchestrationRequest) -> OrchestrationResponse:
    """Call the Orchestration Service and return a typed response."""
    client = _get_client()
    try:
        resp = await client.post("/v1/run", json=request.model_dump())
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.error("Orchestration error %d: %s", exc.response.status_code, exc.response.text)
        raise
    except httpx.RequestError as exc:
        logger.error("Orchestration connection error: %s", exc)
        raise
    return OrchestrationResponse.model_validate(resp.json())


async def stream_pipeline(request: OrchestrationRequest) -> AsyncIterator[str]:
    """
    Proxy the orchestration SSE stream to the caller.

    Yields raw SSE lines (``data: {...}\\n\\n``) as they arrive so the
    gateway can forward them directly to the end user with zero buffering.
    Uses a dedicated context-managed client so the connection outlives the
    pooled client lifecycle.
    """
    url = f"{settings.orchestration_url.rstrip('/')}/v1/run/stream"
    async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
        async with client.stream("POST", url, json=request.model_dump()) as response:
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                logger.error("Orchestration stream error %d", exc.response.status_code)
                raise
            async for chunk in response.aiter_bytes():
                yield chunk


async def close() -> None:
    """Close the shared client.  Call from app shutdown hook."""
    global _client
    if _client and not _client.is_closed:
        await _client.aclose()
        _client = None
