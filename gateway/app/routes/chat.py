"""
Chat route — user-facing API
==============================
POST /v1/chat
    Accepts a plain-text prompt, classifies it, forwards it to the
    Orchestration Service, and returns the final response to the caller.

GET /health
    Returns {"status": "ok"} for Docker / load-balancer probes.
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from shared.contracts.orchestration import OrchestrationRequest, OrchestrationResponse
from ..clients.orchestration import run_pipeline, stream_pipeline
from ..routing.query_router import QueryRouter

logger = logging.getLogger(__name__)

router = APIRouter()
_query_router = QueryRouter()


class ChatRequest(BaseModel):
    prompt: str = Field(..., description="User's natural-language request")
    session_id: Optional[str] = Field(None, description="Optional session identifier for continuity")
    task_id: Optional[str] = Field(None, description="Stable identifier scoping multi-turn context state. Defaults to session_id when absent.")
    suppress_few_shots: bool = False
    max_iterations: Optional[int] = Field(None, description="Override default max iterations (1–10)")
    convergence_threshold: Optional[float] = Field(None, description="Override convergence score threshold (0.0–1.0)")


class ChatResponse(BaseModel):
    session_id: str
    output: str
    success: bool
    confidence: float
    iterations: int
    converged: bool


@router.post(
    "/v1/chat",
    response_model=ChatResponse,
    summary="Submit a prompt and receive a multi-agent response",
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    1. Route the prompt to determine intent, complexity, and agent_chain.
    2. Forward to the Orchestration Service as an OrchestrationRequest.
    3. Return the final output to the caller.
    """
    session_id = request.session_id or str(uuid.uuid4())

    route = _query_router.route(request.prompt)

    orch_kwargs: dict = {
        "prompt": request.prompt,
        "agent_chain": route.agent_chain,
        "session_id": session_id,
        "intent": route.intent,
        "complexity": route.complexity,
    }
    if request.task_id is not None:
        orch_kwargs["task_id"] = request.task_id
    if request.suppress_few_shots:
        orch_kwargs["suppress_few_shots"] = True
    if request.max_iterations is not None:
        orch_kwargs["max_iterations"] = request.max_iterations
    if request.convergence_threshold is not None:
        orch_kwargs["convergence_threshold"] = request.convergence_threshold

    orch_request = OrchestrationRequest(**orch_kwargs)

    try:
        result: OrchestrationResponse = await run_pipeline(orch_request)
    except Exception as exc:
        logger.error("[%s] Orchestration call failed: %s", session_id[:8], exc)
        raise HTTPException(status_code=502, detail=f"Orchestration service error: {exc}") from exc

    return ChatResponse(
        session_id=result.session_id,
        output=result.output,
        success=result.success,
        confidence=result.confidence,
        iterations=result.iterations,
        converged=result.converged,
    )


@router.get("/health", summary="Health check")
async def health() -> dict:
    return {"status": "ok", "service": "gateway"}


@router.post(
    "/v1/chat/stream",
    summary="Submit a prompt and receive a streaming Server-Sent Events response",
)
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """
    Identical routing logic to POST /v1/chat but streams SSE events from
    the orchestration pipeline as each agent completes.

    The client should connect with ``Accept: text/event-stream`` and read
    events until it receives ``data: {"type": "done", ...}``.

    Event shapes (see orchestration /v1/run/stream for full schema):
      {"type": "start",          "session_id": "...", "agent_chain": [...]}
      {"type": "agent_start",    "role": "planner",  "message": "Working on a plan..."}
      {"type": "chunk",          "content": "def "}
      {"type": "agent_complete", "role": "planner",  "label": "Here's the plan:", "content": "...", "iteration": 1}
      {"type": "progress",       "message": "Refining the answer (pass 1 complete)...", "score": 0.72, "iteration": 1}
      {"type": "done",           "output": "...", "confidence": 0.87, ...}
      {"type": "error",          "message": "..."}
    """
    session_id = request.session_id or str(uuid.uuid4())
    route = _query_router.route(request.prompt)

    orch_kwargs: dict = {
        "prompt": request.prompt,
        "agent_chain": route.agent_chain,
        "session_id": session_id,
        "intent": route.intent,
        "complexity": route.complexity,
    }
    if request.task_id is not None:
        orch_kwargs["task_id"] = request.task_id
    if request.suppress_few_shots:
        orch_kwargs["suppress_few_shots"] = True
    if request.max_iterations is not None:
        orch_kwargs["max_iterations"] = request.max_iterations
    if request.convergence_threshold is not None:
        orch_kwargs["convergence_threshold"] = request.convergence_threshold

    orch_request = OrchestrationRequest(**orch_kwargs)

    try:
        return StreamingResponse(
            stream_pipeline(orch_request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )
    except Exception as exc:
        logger.error("[%s] Orchestration stream failed: %s", session_id[:8], exc)
        raise HTTPException(status_code=502, detail=f"Orchestration service error: {exc}") from exc
