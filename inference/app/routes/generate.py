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
    Returns {"status": "ok", "mode": "<active mode>"} to allow Docker /
    Compose healthchecks and Orchestration's circuit breaker to verify
    the service is reachable.
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from shared.contracts.inference import GenerateRequest, GenerateResponse
from ..backends.factory import get_backend
from ..config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/v1/generate",
    response_model=GenerateResponse,
    summary="Generate a model response",
)
async def generate(request: GenerateRequest) -> GenerateResponse:
    """
    Run one LLM inference call.

    - In **local** mode hits the vLLM server matching `model_family`.
    - Calls the local vLLM servers for all requests.
    """
    try:
        backend = get_backend()
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
    backend = get_backend()

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
    """Returns the service status and active inference mode."""
    return {"status": "ok", "mode": settings.mode}
