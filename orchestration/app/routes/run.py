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

from shared.contracts.context import BuiltContext
from shared.contracts.experience import AgentContribution, ExperienceEvent, SCORER_RULE_VERSION
from shared.contracts.orchestration import (
    AgentStep,
    OrchestrationRequest,
    OrchestrationResponse,
)
from ..context import builder
from ..db import redis as session_store
from ..experience import inflight, recorder
from ..config.settings import settings
from ..graph import nodes as _nodes_module
from ..graph.nodes import AGENT_FRIENDLY
from ..graph.state import AgentState
from ..plans.reader import read_plan
from ..plans.writer import write_plan_atomic, validate_transition
from ..plans.schema import Plan
from ..utils.scoring import score_per_entry

logger = logging.getLogger(__name__)

router = APIRouter()

OUTPUT_SUMMARY_MAX_CHARS = 300  # API-visible field; value must not change without a contract update

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
    task_id = getattr(request, "task_id", None) or session_id
    if task_id == session_id:
        logger.debug(
            "[%s] task_id not supplied — conversation history will not persist across separate sessions.",
            session_id[:8],
        )
    allowed_files = getattr(request, "allowed_files", None)

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

    try:
        built_context = await builder.build(
            request.prompt, context_intent, context_complexity,
            task_id=task_id, allowed_files=allowed_files,
        )
    except Exception as exc:
        logger.warning("Context build failed (%s) — proceeding with empty context.", exc)
        built_context = BuiltContext()

    conversation_history = await session_store.get_session(task_id)
    redis_healthy = session_store.is_healthy()
    if not redis_healthy:
        logger.warning("Redis unavailable — session history lost for %s", task_id)

    max_iterations = request.max_iterations or settings.default_max_iterations
    convergence_threshold = request.convergence_threshold or settings.convergence_threshold
    logger.info(
        "[%s] convergence_threshold=%.2f (request_override=%s)",
        session_id[:8], convergence_threshold, request.convergence_threshold,
    )

    # ── Plan loading ─────────────────────────────────────────────────────────
    plan_ephemeral = (task_id == session_id)
    plan_dict: dict | None = None
    plan_path_str = getattr(request, "plan_path", None) or "plans/current-task.md"

    if not plan_ephemeral:
        raw_plan = await session_store.get_plan(task_id)
        if raw_plan:
            plan_dict = raw_plan
        else:
            try:
                plan_obj = read_plan(plan_path_str)
                plan_dict = plan_obj.model_dump(mode="json") if plan_obj else None
            except Exception as exc:
                logger.warning("[%s] plan file unreadable (%s) — proceeding without plan.", session_id[:8], exc)
    else:
        logger.debug("[%s] plan ephemeral — task_id not supplied", session_id[:8])
        try:
            plan_obj = read_plan(plan_path_str)
            plan_dict = plan_obj.model_dump(mode="json") if plan_obj else None
        except Exception:
            plan_dict = None

    initial_state: AgentState = {
        "session_id": session_id,
        "task_id": task_id,
        "user_prompt": request.prompt,
        "agent_chain": agent_chain,
        "agent_chain_index": 0,
        "allowed_files": built_context.allowed_files,
        "file_fingerprints": built_context.file_fingerprints,
        "few_shots": [ex.model_dump() for ex in built_context.few_shots],
        "repo_snippets": [s.model_dump() for s in built_context.repo_snippets],
        "system_hints": built_context.system_hints,
        "current_iteration": 1,
        "max_iterations": max_iterations,
        "convergence_threshold": convergence_threshold,
        "conversation_history": conversation_history,
        "iteration_history": [],
        "last_output": "",
        "iteration_scores": [],
        "best_score": 0.0,
        "best_output": "",
        "converged": False,
        "quality_converged": False,
        "final_output": "",
        "final_score": 0.0,
        "plan": plan_dict,
        "plan_changed": False,
        "current_step_id": None,
    }

    from ..graph.pipeline import _recursion_limit
    rec_limit = _recursion_limit(max_iterations, len(agent_chain))

    return (session_id, context_intent, context_complexity, agent_chain,
            initial_state, rec_limit, redis_healthy,
            plan_ephemeral, plan_path_str)


def _trim_conversation_history(history: list, max_tokens: int) -> list:
    """Return the most recent entries from history that fit within max_tokens (char-based estimate)."""
    chars_budget = max_tokens * settings.chars_per_token
    result: list = []
    used = 0
    for entry in reversed(history):
        chars = len(str(entry.get("output", "")) + str(entry.get("full_output", "")))
        if used + chars > chars_budget and result:
            break
        result.append(entry)
        used += chars
    return list(reversed(result))


def _build_contributions(history: list) -> List[AgentContribution]:
    """
    Build a list of AgentContribution objects from the pipeline history.

    role='tool' entries are execution scaffolding — they are skipped here and
    instead surfaced via tool_turns on the preceding agent contribution.
    """
    contributions = []
    for i, h in enumerate(history):
        role = h.get("role", "")
        if role == "tool":
            continue

        tool_turns = None
        if "tool_calls" in h:
            tool_turns = [{"role": "assistant", "tool_calls": h["tool_calls"], "content": None}]
            j = i + 1
            while j < len(history) and history[j].get("role") == "tool":
                t = history[j]
                tool_turns.append({
                    "role": "tool",
                    "tool_call_id": t.get("tool_call_id", ""),
                    "name": t.get("name", ""),
                    "content": t.get("output", ""),
                })
                j += 1

        contributions.append(
            AgentContribution(
                role=role,
                output=str(h.get("full_output") or h.get("output", "")),
                quality_score=score_per_entry(h),
                iteration=h.get("iteration", 1),
                messages=h.get("messages"),
                tool_turns=tool_turns,
            )
        )
    return contributions


def _maybe_writeback_plan(
    final_state: AgentState,
    initial_state: AgentState,
    plan_ephemeral: bool,
    plan_path_str: str,
    session_id: str,
    converged: bool,
) -> None:
    """On convergence, persist plan to Redis (if non-ephemeral) and write back to markdown."""
    if not final_state.get("plan_changed"):
        return
    plan_dict = final_state.get("plan")
    if not plan_dict:
        return

    task_id = initial_state.get("task_id", session_id)

    if not plan_ephemeral:
        ttl = getattr(settings, "task_state_ttl_s", 7 * 86400)
        asyncio.create_task(session_store.set_plan(task_id, plan_dict, ttl))

    if converged or not plan_ephemeral:
        try:
            from pathlib import Path as _P
            from datetime import datetime as _dt, timezone as _tz

            plan_obj = Plan.model_validate(plan_dict)
            plan_path = _P(plan_path_str)

            # Validate transitions before writing back
            initial_plan_dict = initial_state.get("plan")
            if initial_plan_dict:
                try:
                    old_plan = Plan.model_validate(initial_plan_dict)
                    validate_transition(old_plan, plan_obj)
                except ValueError as exc:
                    logger.warning("[%s] Plan transition validation failed: %s", session_id[:8], exc)
                    return

            write_plan_atomic(plan_path, plan_obj)

            # Archive snapshot
            archive_dir = _P("plans/archive")
            archive_dir.mkdir(parents=True, exist_ok=True)
            stamp = _dt.now(_tz.utc).strftime("%Y%m%d-%H%M%S")
            from ..plans.writer import render_plan
            (archive_dir / f"{task_id}-{stamp}.md").write_text(
                render_plan(plan_obj), encoding="utf-8"
            )
            logger.info("[%s] Plan archived to plans/archive/%s-%s.md", session_id[:8], task_id, stamp)
        except Exception as exc:
            logger.warning("[%s] Plan writeback failed: %s", session_id[:8], exc)


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
    (session_id, context_intent, context_complexity, agent_chain,
     initial_state, rec_limit, redis_healthy,
     plan_ephemeral, plan_path_str) = await _prepare_pipeline_run(request)

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

    prior_history = initial_state.get("conversation_history", [])
    combined = _trim_conversation_history(
        prior_history + history, settings.max_conversation_history_tokens
    )
    asyncio.create_task(session_store.save_session(
        initial_state["task_id"], combined, settings.redis_session_ttl_s
    ))

    # ── Plan writeback on convergence ─────────────────────────────────────
    _maybe_writeback_plan(
        final_state, initial_state, plan_ephemeral, plan_path_str, session_id, converged
    )

    agent_steps = [
        AgentStep(
            role=h["role"],
            iteration=h.get("iteration", 1),
            output_summary=str(h.get("output", ""))[:OUTPUT_SUMMARY_MAX_CHARS],
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
        history_volatile=not redis_healthy,
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
            scorer_rule_version=SCORER_RULE_VERSION,
        )
        inflight.register(asyncio.create_task(recorder.record(event)), event)

    return response


@router.get("/health", summary="Health check")
async def health() -> dict:
    return {
        "status": "ok",
        "service": "orchestration",
        "redis": "ok" if session_store.is_healthy() else "degraded",
    }


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
      start          — pipeline started; carries session_id and agent_chain
      agent_start    — an agent is beginning; carries role and a friendly message
      chunk          — one token fragment from the active agent (real-time text)
      agent_complete — an agent finished; carries role, label, and full content
      progress       — an iteration finished; carries a friendly message and score
      done           — pipeline finished; carries final output + metadata
      error          — unrecoverable pipeline error

    ``chunk`` events arrive between ``agent_start`` and ``agent_complete`` —
    clients render output character-by-character, then have the full content
    available in ``agent_complete`` once the agent finishes.
    """
    (session_id, context_intent, context_complexity, agent_chain,
     initial_state, rec_limit, redis_healthy,
     plan_ephemeral, plan_path_str) = await _prepare_pipeline_run(request)

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
                            _, label = AGENT_FRIENDLY.get(
                                node_name, ("...", node_name.capitalize())
                            )
                            await q.put({
                                "type": "agent_complete",
                                "role": node_name,
                                "label": label,
                                "content": delta.get("last_output", ""),
                                "iteration": iteration,
                            })

                            # Emit task_state when plan changed
                            if delta.get("plan_changed") and delta.get("plan"):
                                steps = delta["plan"].get("steps", [])
                                completed = sum(1 for s in steps if s["status"] == "done")
                                await q.put({
                                    "type": "task_state",
                                    "task_id": initial_state.get("task_id", session_id),
                                    "completed_steps": completed,
                                    "total_steps": len(steps),
                                })

                        elif node_name == "update_loop":
                            iteration = accumulated.get("current_iteration", last_iteration_seen)
                            score = (accumulated.get("iteration_scores") or [0.0])[-1]
                            last_iteration_seen = iteration
                            await q.put({
                                "type": "progress",
                                "message": f"Refining the answer (pass {iteration - 1} complete)...",
                                "score": round(score, 4),
                                "iteration": iteration - 1,
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
                        "history_volatile": not redis_healthy,
                    })

                    # Persist session history to Redis
                    history = accumulated.get("iteration_history", [])
                    prior_history = initial_state.get("conversation_history", [])
                    combined = _trim_conversation_history(
                        prior_history + history, settings.max_conversation_history_tokens
                    )
                    asyncio.create_task(session_store.save_session(
                        initial_state["task_id"], combined, settings.redis_session_ttl_s
                    ))

                    # Fire-and-forget experience event
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
                            scorer_rule_version=SCORER_RULE_VERSION,
                        )
                        inflight.register(asyncio.create_task(recorder.record(event)), event)

                    # Plan writeback on convergence
                    _maybe_writeback_plan(
                        accumulated, initial_state, plan_ephemeral,
                        plan_path_str, session_id, converged,
                    )

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
