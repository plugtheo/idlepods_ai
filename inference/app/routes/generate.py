"""
Generate route — POST /v1/generate  |  POST /v1/generate/stream
=================================================================
Receives a GenerateRequest from the Orchestration Service, delegates to
the active InferenceBackend, and returns a GenerateResponse.

POST /v1/generate
    Blocking — returns the full model response once generation completes.

POST /v1/generate/stream
    Server-Sent Events — streams token fragments as they are produced.
    Each event is ``data: <JSON>`` where the JSON is one of:
        {"token": "<fragment>", "is_final": false}   — one or more per call
        {"token": "",           "is_final": true }   — final sentinel

Health check — GET /health
    Returns per-model-family backend status including loaded adapter list.
    If a backend is unreachable, reports "status": "unavailable" for that
    family without failing the overall health check.
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from shared.contracts.inference import GenerateRequest, GenerateResponse
from ..backends.factory import get_backend
from ..backends.local_vllm import get_fallback_counts
from ..config.settings import settings


class _TokenizeBody(BaseModel):
    model_family: str
    text: str

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/v1/generate",
    response_model=GenerateResponse,
    summary="Generate a model response",
)
async def generate(request: GenerateRequest) -> GenerateResponse:
    """Run one LLM inference call via the backend for request.model_family."""
    try:
        backend = get_backend(request.model_family)
        return await backend.generate(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Inference failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=502, detail="Inference backend error") from exc


@router.post(
    "/v1/generate/stream",
    summary="Stream generated tokens as Server-Sent Events",
)
async def generate_stream(request: GenerateRequest) -> StreamingResponse:
    """
    Stream LLM output token-by-token as SSE.

    Each event carries one ``{"token": "...", "is_final": false}`` fragment.
    A final ``{"token": "", "is_final": true}`` sentinel closes the stream.
    Errors are reported as ``{"error": "<message>"}`` before the connection
    closes.
    """
    backend = get_backend(request.model_family)

    async def _token_stream():
        try:
            async for token in backend.generate_stream(request):
                yield f"data: {json.dumps({'token': token, 'is_final': False})}\n\n"
        except Exception as exc:
            logger.error("Streaming inference failed: %s", exc, exc_info=True)
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
            return
        yield f"data: {json.dumps({'token': '', 'is_final': True})}\n\n"

    return StreamingResponse(
        _token_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/health", summary="Health check")
async def health() -> dict:
    """
    Returns per-family backend status including currently loaded adapter list.

    A single unreachable backend reports "unavailable" for its family but
    does not cause the whole health check to fail.
    """
    backends_status: dict = {}

    for family in ("qwen",):
        try:
            backend = get_backend(family)
            backend_health = await backend.health()
            adapters = await backend.list_adapters()
            backends_status[family] = {
                "type": backend_health.get("backend", "unknown"),
                "url": backend_health.get("url", ""),
                "status": backend_health.get("status", "unknown"),
                "adapters_loaded": adapters,
            }
        except Exception as exc:
            logger.warning("Health check error for %s: %s", family, exc)
            backends_status[family] = {
                "status": "unavailable",
                "error": str(exc),
            }

    overall = (
        "ok"
        if all(b.get("status") == "ok" for b in backends_status.values())
        else "degraded"
    )
    return {"status": overall, "backends": backends_status, "adapter_fallbacks": get_fallback_counts()}


@router.get("/v1/model-info", summary="Return max_model_len for each vLLM backend")
async def model_info() -> dict:
    """Query each vLLM backend for the served model's max_model_len."""
    result: dict = {}
    for family in ("qwen",):
        try:
            backend = get_backend(family)
            result[family] = await backend.max_model_len()
        except Exception as exc:
            logger.error("model_info failed for %s: %s", family, exc)
            raise HTTPException(
                status_code=502,
                detail=f"Failed to get model info for {family}: {exc}",
            ) from exc
    return result


@router.post("/v1/tokenize", summary="Return token count for a text string via a vLLM backend")
async def tokenize_text(body: _TokenizeBody) -> dict:
    """Proxy a tokenization request to the appropriate vLLM backend."""
    if body.model_family not in ("qwen",):
        raise HTTPException(status_code=400, detail="model_family must be 'qwen'")
    try:
        backend = get_backend(body.model_family)
        count = await backend.tokenize(body.text)
        return {"token_count": count}
    except Exception as exc:
        logger.error("tokenize failed for %s: %s", body.model_family, exc)
        raise HTTPException(status_code=502, detail=f"Tokenize failed: {exc}") from exc
