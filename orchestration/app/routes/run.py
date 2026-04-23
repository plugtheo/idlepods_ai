"""
Orchestration run route
========================
POST /v1/run
    Accepts an OrchestrationRequest, runs the LangGraph pipeline,
    publishes the experience event on convergence, and returns an
    OrchestrationResponse to the caller (the Gateway Service).

POST /v1/run/stream
    Same as /v1/run but streams Server-Sent Events as each agent node
    completes, enabling the Gateway to proxy a live thought-chain feed to
    end users.  Event types: agent_step | iteration_complete | done | error

GET  /health
    Returns {"status": "ok"} for liveness / readiness checks.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from shared.contracts.context import ContextRequest
from shared.contracts.experience import AgentContribution, ExperienceEvent
from shared.contracts.orchestration import (
    AgentStep,
    OrchestrationRequest,
    OrchestrationResponse,
)
from ..clients.context import fetch_context
from ..clients.experience import fire_publish_experience
from ..config.settings import settings
from ..graph import nodes as _nodes_module
from ..graph.state import AgentState
from ..utils.scoring import score_per_entry

logger = logging.getLogger(__name__)

router = APIRouter()

# _PIPELINE is lazy-initialized on first request (or overridden in tests).
# Keeping the build step out of module scope means importing this module does
# not require langgraph to be installed — tests that mock _PIPELINE work even
# in environments where langgraph is absent (e.g. the context/ dev venv).
_PIPELINE = None


def _get_pipeline():
    """Return the compiled LangGraph pipeline, building it on first call."""
    global _PIPELINE
    if _PIPELINE is None:
        from ..graph.pipeline import build_pipeline
        _PIPELINE = build_pipeline()
    return _PIPELINE


# ── Shared pipeline-setup helper ──────────────────────────────────────────


async def _prepare_pipeline_run(request: OrchestrationRequest) -> tuple:
    """
    Shared setup executed by both /v1/run and /v1/run/stream.

    Returns
    -------
    (session_id, context_intent, context_complexity, agent_chain,
     initial_state, recursion_limit)
    """
    session_id = request.session_id or str(uuid.uuid4())

    # Use intent/complexity forwarded from the gateway if available — avoids
    # double routing.  Fall back to a local QueryRouter pass only when the request
    # comes from a caller that doesn't pre-classify (e.g. direct API access or tests).
    if request.intent is not None and request.complexity is not None:
        context_intent = request.intent
        context_complexity = request.complexity
        agent_chain = request.agent_chain
    else:
        from ..routing.query_router import QueryRouter
        _route = QueryRouter().route(request.prompt)
        context_intent = _route.intent
        context_complexity = _route.complexity
        agent_chain = request.agent_chain or _route.agent_chain

    context_req = ContextRequest(
        prompt=request.prompt,
        intent=context_intent,
        complexity=context_complexity,
        session_id=session_id,
    )
    built_context = await fetch_context(context_req)

    max_iterations = request.max_iterations or settings.default_max_iterations
    convergence_threshold = request.convergence_threshold or settings.convergence_threshold

    initial_state: AgentState = {
        "session_id": session_id,
        "user_prompt": request.prompt,
        "agent_chain": agent_chain,
        "agent_chain_index": 0,
        "few_shots": [ex.model_dump() for ex in built_context.few_shots],
        "repo_snippets": [s.model_dump() for s in built_context.repo_snippets],
        "system_hints": built_context.system_hints,
        "current_iteration": 1,
        "max_iterations": max_iterations,
        "convergence_threshold": convergence_threshold,
        "iteration_history": [],
        "last_output": "",
        "iteration_scores": [],
        "best_score": 0.0,
        "best_output": "",
        "converged": False,
        "quality_converged": False,
        "final_output": "",
        "final_score": 0.0,
    }

    from ..graph.pipeline import _recursion_limit
    rec_limit = _recursion_limit(max_iterations, len(agent_chain))

    return session_id, context_intent, context_complexity, agent_chain, initial_state, rec_limit


def _build_contributions(history: list) -> List[AgentContribution]:
    """
    Build a list of AgentContribution objects from the pipeline history.

    Per-agent quality scores are extracted from each history entry using
    score_per_entry() — evaluator roles prefer an explicit SCORE annotation,
    generative roles fall back to a heuristic based on output content.
    """
    return [
        AgentContribution(
            role=h["role"],
            # Use full_output (raw LLM response) when available; fall back
            # to the compact extracted version stored in "output".
            output=str(h.get("full_output") or h.get("output", "")),
            quality_score=score_per_entry(h),
            iteration=h.get("iteration", 1),
            # Full message list paired with output forms a complete SFT training pair.
            messages=h.get("messages"),
        )
        for h in history
    ]


# ── HTTP endpoints ─────────────────────────────────────────────────────────


@router.post(
    "/v1/run",
    response_model=OrchestrationResponse,
    summary="Run the multi-agent pipeline for one user request",
)
async def run_pipeline(request: OrchestrationRequest) -> OrchestrationResponse:
    """
    Entry point for the Orchestration Service.

    1. Enriches the prompt via Context Service (with timeout/circuit-breaker).
    2. Initialises the LangGraph state.
    3. Runs the compiled agent pipeline to completion.
    4. Fires-and-forgets the experience event to the Experience Service.
    5. Returns the final result to the Gateway.
    """
    session_id, context_intent, context_complexity, agent_chain, initial_state, rec_limit = (
        await _prepare_pipeline_run(request)
    )

    try:
        final_state: AgentState = await _get_pipeline().ainvoke(
            initial_state,
            config={"recursion_limit": rec_limit},
        )
    except Exception as exc:
        logger.error("[%s] Pipeline error: %s", session_id[:8], exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}") from exc

    history = final_state.get("iteration_history", [])
    best_score = final_state.get("best_score", 0.0)
    final_output = final_state.get("final_output") or final_state.get("best_output", "")
    converged = final_state.get("quality_converged", False)
    iterations_ran = final_state.get("current_iteration", 1)

    agent_steps = [
        AgentStep(
            role=h["role"],
            iteration=h.get("iteration", 1),
            output_summary=str(h.get("output", ""))[:300],
            score=score_per_entry(h),
        )
        for h in history
    ]

    response = OrchestrationResponse(
        session_id=session_id,
        output=final_output,
        success=bool(final_output),
        confidence=best_score,
        iterations=iterations_ran,
        best_score=best_score,
        agent_steps=agent_steps,
        converged=converged,
        metadata={
            "intent": context_intent,
            "complexity": context_complexity,
            "agent_chain": agent_chain,
        },
    )

    # Fire-and-forget experience event — stored on every run that produced output.
    if final_output:
        event = ExperienceEvent(
            session_id=session_id,
            prompt=request.prompt,
            final_output=final_output,
            agent_chain=agent_chain,
            contributions=_build_contributions(history),
            final_score=best_score,
            iterations=iterations_ran,
            converged=converged,
            iteration_scores=final_state.get("iteration_scores", []),
            intent=context_intent,
            complexity=context_complexity,
            timestamp=datetime.now(tz=timezone.utc),
        )
        fire_publish_experience(event)

    return response


@router.get("/health", summary="Health check")
async def health() -> dict:
    return {"status": "ok", "service": "orchestration"}


# ── Streamable node names (excludes helper/loop nodes) ────────────────────
_STREAMABLE_NODES = {
    "planner", "researcher", "coder", "debugger",
    "reviewer", "critic", "review_critic", "consensus",
}


@router.post(
    "/v1/run/stream",
    summary="Stream the multi-agent pipeline as Server-Sent Events",
)
async def run_pipeline_stream(request: OrchestrationRequest) -> StreamingResponse:
    """
    Same logic as POST /v1/run but yields SSE events as the pipeline runs.
    The gateway proxies these directly to end users.

    Event types emitted on the ``data:`` field (JSON):
      start              — pipeline started; carries session_id and agent_chain
      token              — one token fragment from the active agent (real-time)
      agent_step         — an agent node finished; carries role, iteration, output
      iteration_complete — a full agent-chain iteration finished; carries score
      done               — pipeline finished; carries final output + metadata
      error              — unrecoverable pipeline error

    ``token`` events arrive *before* the corresponding ``agent_step`` event —
    clients can render streamed output character-by-character, then replace it
    with the normalised ``agent_step.output`` when the agent completes.
    """
    session_id, context_intent, context_complexity, agent_chain, initial_state, rec_limit = (
        await _prepare_pipeline_run(request)
    )

    async def _event_stream():
        # Single shared queue for ALL events:
        #   - token events  (put by _run_agent_node as tokens arrive)
        #   - agent_step    (put by _pipeline_runner after node completes)
        #   - iteration_complete (put by _pipeline_runner at loop boundary)
        #   - done / error  (put by _pipeline_runner at pipeline end)
        #   - None sentinel (put by _pipeline_runner to signal generator exit)
        #
        # Ordering guarantee: _run_agent_node puts all token events before
        # returning, then _pipeline_runner's astream() yields the node delta
        # and puts the agent_step event.  So the client always sees:
        #   token … token … agent_step   (never agent_step before final token)
        q: asyncio.Queue = asyncio.Queue()
        _nodes_module.register_token_queue(session_id, q)

        yield f"data: {json.dumps({'type': 'start', 'session_id': session_id, 'agent_chain': agent_chain})}\n\n"

        async def _pipeline_runner() -> None:
            accumulated: dict = {}
            last_iteration_seen = 1
            run_error = None
            try:
                async for chunk in _get_pipeline().astream(
                    initial_state,
                    config={"recursion_limit": rec_limit},
                ):
                    for node_name, delta in chunk.items():
                        # Conditional edge transitions yield None as their delta
                        # (the edge function returns a route string, not a state
                        # update).  Skip these non-state chunks.
                        if delta is None or not isinstance(delta, dict):
                            continue
                        accumulated.update(delta)

                        if node_name in _STREAMABLE_NODES:
                            history_added = delta.get("iteration_history", [])
                            iteration = (
                                history_added[-1].get("iteration", 1)
                                if history_added
                                else accumulated.get("current_iteration", 1)
                            )
                            await q.put({
                                "type": "agent_step",
                                "role": node_name,
                                "iteration": iteration,
                                "output": delta.get("last_output", ""),
                            })

                        elif node_name == "update_loop":
                            iteration = accumulated.get("current_iteration", last_iteration_seen)
                            score = (accumulated.get("iteration_scores") or [0.0])[-1]
                            last_iteration_seen = iteration
                            await q.put({
                                "type": "iteration_complete",
                                "iteration": iteration - 1,
                                "score": round(score, 4),
                            })

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("[%s] Stream pipeline error: %s", session_id[:8], exc, exc_info=True)
                run_error = exc
                await q.put({"type": "error", "message": str(exc)})
            finally:
                if run_error is None:
                    best_score = accumulated.get("best_score", 0.0)
                    final_output = (
                        accumulated.get("final_output") or accumulated.get("best_output", "")
                    )
                    converged = accumulated.get("quality_converged", False)
                    iterations_ran = accumulated.get("current_iteration", 1)
                    await q.put({
                        "type": "done",
                        "session_id": session_id,
                        "output": final_output,
                        "confidence": round(best_score, 4),
                        "iterations": iterations_ran,
                        "converged": converged,
                        "intent": context_intent,
                        "complexity": context_complexity,
                    })

                    # Fire-and-forget experience event
                    history = accumulated.get("iteration_history", [])
                    if final_output and history:
                        event = ExperienceEvent(
                            session_id=session_id,
                            prompt=request.prompt,
                            final_output=final_output,
                            agent_chain=agent_chain,
                            contributions=_build_contributions(history),
                            final_score=best_score,
                            iterations=iterations_ran,
                            converged=converged,
                            iteration_scores=accumulated.get("iteration_scores", []),
                            intent=context_intent,
                            complexity=context_complexity,
                            timestamp=datetime.now(tz=timezone.utc),
                        )
                        fire_publish_experience(event)

                _nodes_module.unregister_token_queue(session_id)
                await q.put(None)  # sentinel — tells the generator to stop

        task = asyncio.create_task(_pipeline_runner())
        try:
            while True:
                event = await q.get()
                if event is None:
                    break
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            # Ensure the pipeline task is cleaned up if the client disconnects
            # or the generator exits for any other reason.
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering for SSE
            "Connection": "keep-alive",
        },
    )
