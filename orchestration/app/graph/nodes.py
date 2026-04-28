"""
LangGraph pipeline nodes
=========================
Each function here is a LangGraph node — it receives an AgentState,
calls the Inference Service for one generation step, appends the result
to `iteration_history`, and returns the state delta.

No model loading. No framework objects. No AutoGen.
The only external call is `await inference_client.generate(request)` or,
when a token queue is registered for the session, the streaming variant
`async for token in inference_client.generate_stream(request)`.

Token streaming
---------------
The orchestration SSE route (``/v1/run/stream``) registers an
``asyncio.Queue`` keyed by ``session_id`` before running the pipeline.
When a queue is present, ``_run_agent_node`` switches to ``generate_stream``
and puts each token fragment into the queue as it arrives, allowing the
SSE generator to relay tokens to the end user in real-time.

Node naming convention
----------------------
Function name = agent role name (e.g. `coder_node`, `planner_node`).
The pipeline registers them under the role string: "coder", "planner", etc.

Message construction
--------------------
Every generation call receives:
  1. System prompt for the role (from config/settings.py AGENT_PROMPTS)
  2. System hints from context enrichment
  3. Few-shot examples (past similar solutions, injected as user turns)
  4. The original user prompt
  5. Prior agent outputs from the current and previous iterations
     (so each agent sees what predecessors said)
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from shared.contracts.inference import GenerateRequest, Message
from ..clients.inference import get_inference_client
from ..config.settings import AGENT_PROMPTS, settings
from ..utils.inference_optimizer import InferenceOptimizer
from .state import AgentState

# Module-level singleton — shared across all node invocations in this process.
# Flags read once at startup; change requires service restart (or env reload).
_optimizer = InferenceOptimizer(
    role_history_filter=settings.optimize_role_history_filter,
    structured_extraction=settings.optimize_structured_extraction,
)

logger = logging.getLogger(__name__)


# ── Per-session token queues ──────────────────────────────────────────────
#
# The streaming SSE route registers an asyncio.Queue here before invoking
# the pipeline.  _run_agent_node puts token events into the queue while
# generation is in progress; the SSE generator reads from it concurrently.
#
# Lifecycle: register_token_queue → pipeline runs → unregister_token_queue.
# The queue is always cleaned up in the SSE route's finally block.

_token_queues: Dict[str, asyncio.Queue] = {}

# Per-role friendly messages: (thinking_message, output_label)
AGENT_FRIENDLY: Dict[str, tuple] = {
    "planner":      ("Working on a plan...",                "Here's the plan:"),
    "researcher":   ("Researching this topic...",           "Here's what I found:"),
    "coder":        ("Writing the implementation...",       "Here's the code:"),
    "debugger":     ("Debugging the code...",               "Here's what I found and fixed:"),
    "reviewer":     ("Reviewing the solution...",           "My review:"),
    "critic":       ("Critically analyzing the approach...", "My critique:"),
    "review_critic":("Reviewing and critiquing...",         "Review & critique:"),
    "consensus":    ("Synthesizing the final answer...",    "Here's the answer:"),
}


def register_token_queue(session_id: str, q: asyncio.Queue) -> None:
    """Register a queue to receive token events for *session_id*."""
    _token_queues[session_id] = q


def unregister_token_queue(session_id: str) -> None:
    """Deregister the queue for *session_id*.  No-op if not present."""
    _token_queues.pop(session_id, None)


def get_token_queue(session_id: str) -> Optional[asyncio.Queue]:
    """Return the active token queue for *session_id*, or None."""
    return _token_queues.get(session_id)


def _system_message(role: str, hints: str) -> Message:
    base_prompt = AGENT_PROMPTS.get(role, f"You are {role}Agent.")
    if hints:
        content = f"{base_prompt}\n\nAdditional guidance: {hints}"
    else:
        content = base_prompt
    return Message(role="system", content=content)


# Conservative char-to-token ratio (real average: ~3.5 EN, ~2.5 code).
# Deliberately under-estimates to stay safely within the context window.
# Configurable via ORCHESTRATION__CHARS_PER_TOKEN.
# Reserve a small buffer for message-wrapping and role/name overhead.
# Configurable via ORCHESTRATION__CONTEXT_SAFETY_MARGIN.


def _estimate_tokens(text: str) -> int:
    """Fast char-based token estimate — no tokenizer required."""
    return max(1, len(text) // settings.chars_per_token)


def _build_messages(
    role: str,
    state: AgentState,
) -> List[Message]:
    """Assemble the full messages list for one agent generation call.

    Treats the context window as a budget and distributes it between the
    fixed parts (system prompt + user task) and the optional context
    blocks (history, repo snippets, few-shots).  The system prompt and
    user prompt are non-negotiable — they are always included in full.

    Budget allocation (after reserving fixed tokens):
        70 % → iteration history  (recency-biased, newest-first)
        20 % → repo snippets      (code-facing roles only)
        remainder → few-shot examples

    This prevents vLLM from silently truncating from the front of the
    context, which would lose the system prompt and user task.
    """
    hints             = state.get("system_hints", "")
    few_shots         = state.get("few_shots", [])
    repo_snippets     = state.get("repo_snippets", [])
    user_prompt       = state.get("user_prompt", "")
    current_iteration = state.get("current_iteration", 1)

    # Filter history so each agent only sees relevant predecessors.
    raw_history = state.get("iteration_history", [])
    iteration_history = _optimizer.filter_history(role, raw_history)

    model_context_len = settings.model_context_len
    role_max_tokens   = settings.role_max_tokens.get(role, 1024)

    # Build the two non-negotiable messages first so we know their token cost.
    system_msg = _system_message(role, hints)
    user_msg   = Message(role="user", content=user_prompt)

    fixed_tokens = (
        _estimate_tokens(system_msg.content)
        + _estimate_tokens(user_msg.content)
        + 20  # role/name overhead per message
    )

    remaining_budget = (
        model_context_len
        - role_max_tokens
        - settings.context_safety_margin
        - fixed_tokens
    )

    if remaining_budget <= 0:
        logger.warning(
            "System + user prompt alone consume the full context window "
            "(role=%s, fixed_tokens=%d, model_context_len=%d). "
            "Consider reducing system_hints or user_prompt length.",
            role, fixed_tokens, model_context_len,
        )
        return [system_msg, user_msg]

    history_budget = int(remaining_budget * settings.context_budget_history_ratio)
    repo_budget    = int(remaining_budget * settings.context_budget_repo_ratio)
    fewshot_budget = remaining_budget - history_budget - repo_budget

    messages: List[Message] = [system_msg]

    # ── Few-shot block ────────────────────────────────────────────────────
    if few_shots and fewshot_budget > 0:
        lines = ["Here are relevant past solutions for reference:"]
        used  = _estimate_tokens("\n".join(lines))
        for i, ex in enumerate(few_shots[:4], 1):
            problem  = str(ex.get("problem",  ""))[:settings.fewshot_problem_max_chars]
            solution = str(ex.get("solution", ""))[:settings.fewshot_solution_max_chars]
            score    = ex.get("score", 0.0)
            entry    = (
                f"\nExample {i} (quality score {score:.2f}):\n"
                f"Task: {problem}\nSolution: {solution}"
            )
            entry_tokens = _estimate_tokens(entry)
            if used + entry_tokens > fewshot_budget:
                break
            lines.append(entry)
            used += entry_tokens
        if len(lines) > 1:
            messages.append(Message(role="user",      content="\n".join(lines)))
            messages.append(Message(role="assistant", content="Understood. I'll use these examples as reference."))

    # ── Repo snippets block ───────────────────────────────────────────────
    if repo_snippets and repo_budget > 0 and role in ("coder", "debugger", "reviewer"):
        repo_lines = ["Relevant code from the repository:"]
        used       = _estimate_tokens("\n".join(repo_lines))
        for snip in repo_snippets[:5]:
            entry        = f"[{snip.get('file', '?')}]: {snip.get('snippet', '')[:settings.repo_snippet_max_chars]}"
            entry_tokens = _estimate_tokens(entry)
            if used + entry_tokens > repo_budget:
                break
            repo_lines.append(entry)
            used += entry_tokens
        if len(repo_lines) > 1:
            messages.append(Message(role="user",      content="\n".join(repo_lines)))
            messages.append(Message(role="assistant", content="I'll keep this repo context in mind."))

    # ── History block (recency-biased, newest-first) ──────────────────────
    if iteration_history and history_budget > 0:
        # history_lookback_iterations controls how many past iterations are
        # eligible (0 = all iterations; 1 = last iteration only; etc.).
        # Budget trimming below handles overflow for any window size.
        lookback = settings.history_lookback_iterations
        if lookback > 0:
            recent = [
                h for h in iteration_history
                if h.get("iteration", 0) >= current_iteration - lookback
            ]
        else:
            recent = list(iteration_history)
        recent_sorted = sorted(recent, key=lambda h: h.get("timestamp", ""), reverse=True)
        history_lines = ["Prior agent outputs:"]
        used          = _estimate_tokens("\n".join(history_lines))
        for h in recent_sorted:
            h_role    = h.get("role", "agent")
            output    = str(h.get("output", ""))
            iter_n    = h.get("iteration", "?")
            label     = f"[iter {iter_n} \u2014 {h_role}]: "
            available = history_budget - used - _estimate_tokens(label) - 5
            if available <= 0:
                history_lines.append(f"{label}\u2026")
                break
            if _estimate_tokens(output) > available:
                # Trim to fit — keep the most recent characters.
                char_budget = available * settings.chars_per_token
                output      = "\u2026" + output[-char_budget:]
            history_lines.append(label + output)
            used += _estimate_tokens(label + output)
        if len(history_lines) > 1:
            messages.append(Message(role="user",      content="\n".join(history_lines)))
            messages.append(Message(role="assistant", content="I've reviewed the prior outputs."))

    # ── Final user task ───────────────────────────────────────────────────
    messages.append(user_msg)

    return messages


# ── Generic agent node factory ────────────────────────────────────────────


async def _run_agent_node(role: str, state: AgentState) -> dict:
    """
    Core logic shared by every agent node.

    Builds messages → calls Inference Service → appends to history →
    returns state delta.
    """
    session_id = state.get("session_id", "")
    current_iteration = state.get("current_iteration", 1)

    messages = _build_messages(role, state)

    request = GenerateRequest(
        model_family=settings.role_model_family.get(role, "mistral"),
        role=role,
        messages=messages,
        adapter_name=settings.role_adapter.get(role),
        max_tokens=settings.role_max_tokens.get(role, 1024),
        session_id=session_id,
    )

    try:
        client = get_inference_client()
        q = get_token_queue(session_id)

        if q is not None and hasattr(client, "generate_stream"):
            # Streaming path: announce the agent once, then push each token as
            # a bare chunk so the client renders output as it arrives without
            # per-token metadata noise.  Local accumulation is unchanged.
            thinking_msg, _ = AGENT_FRIENDLY.get(role, (f"{role.capitalize()} is working...", role))
            await q.put({"type": "agent_start", "role": role, "message": thinking_msg})
            tokens: List[str] = []
            try:
                async for token in client.generate_stream(request):
                    tokens.append(token)
                    await q.put({"type": "chunk", "content": token})
                output = "".join(tokens)
                logger.info(
                    "[%s] iter=%d  role=%s  tokens=%d (streamed)",
                    session_id[:8], current_iteration, role, len(tokens),
                )
            except Exception as exc:
                logger.error(
                    "[%s] Stream inference failed for role=%s: %s",
                    session_id[:8], role, exc,
                )
                # Use whatever tokens arrived before the error; fall back to
                # an error placeholder only if nothing was received at all.
                output = "".join(tokens) if tokens else f"[{role} agent unavailable: {exc}]"
        else:
            # Blocking path (gRPC streaming disabled or client lacks method).
            response = await client.generate(request)
            output = response.content
            logger.info(
                "[%s] iter=%d  role=%s  tokens=%d",
                session_id[:8], current_iteration, role, response.tokens_generated,
            )
    except Exception as exc:
        logger.error("[%s] Inference failed for role=%s: %s", session_id[:8], role, exc)
        output = f"[{role} agent unavailable: {exc}]"

    # Structured extraction: store compact key fields in history so downstream
    # agents' input token cost is lower.  Full output kept as last_output so
    # the convergence scorer can still read SCORE: annotations.
    stored_output = _optimizer.extract_for_history(role, output)

    history_entry = {
        "role": role,
        "iteration": current_iteration,
        "output": stored_output,       # compact version used by downstream agents
        "full_output": output,         # complete LLM response — preserved for training data
        "messages": [m.model_dump() for m in messages],  # full prompt sent to LLM — SFT training pair
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }

    updated_history = list(state.get("iteration_history", [])) + [history_entry]

    # Advance the chain index so the edge router moves to the next agent
    next_index = state.get("agent_chain_index", 0) + 1

    return {
        "iteration_history": updated_history,
        "last_output": output,  # full output — used by convergence scorer
        "agent_chain_index": next_index,
    }


# ── Concrete node functions (one per agent role) ──────────────────────────
# LangGraph requires that each node be a distinct callable.


async def planner_node(state: AgentState) -> dict:
    return await _run_agent_node("planner", state)


async def researcher_node(state: AgentState) -> dict:
    return await _run_agent_node("researcher", state)


async def coder_node(state: AgentState) -> dict:
    return await _run_agent_node("coder", state)


async def debugger_node(state: AgentState) -> dict:
    return await _run_agent_node("debugger", state)


async def reviewer_node(state: AgentState) -> dict:
    return await _run_agent_node("reviewer", state)


async def critic_node(state: AgentState) -> dict:
    return await _run_agent_node("critic", state)


async def review_critic_node(state: AgentState) -> dict:
    """
    Run reviewer then critic sequentially as a single chain step.

    Critic depends on reviewer's structured feedback (SCORE/ISSUES/SUGGESTIONS)
    to produce a meaningful verdict — they cannot run in parallel.
    Running them as one node reduces agent_chain length by one slot while
    preserving the semantic dependency: critic always sees reviewer's output.
    """
    # Reviewer runs first
    reviewer_delta = await _run_agent_node("reviewer", state)

    # Build an intermediate state so critic sees reviewer's history entry
    intermediate_state = {**state, "iteration_history": reviewer_delta["iteration_history"]}

    # Critic runs with reviewer's SCORE/ISSUES/SUGGESTIONS visible in history
    critic_delta = await _run_agent_node("critic", intermediate_state)

    return {
        "iteration_history": critic_delta["iteration_history"],
        "last_output": critic_delta["last_output"],
        "agent_chain_index": state.get("agent_chain_index", 0) + 1,
    }


async def consensus_node(state: AgentState) -> dict:
    """Final synthesis node — produces the definitive answer for the user."""
    delta = await _run_agent_node("consensus", state)
    # The consensus output is the final answer
    consensus_output = delta.get("last_output", state.get("best_output", ""))
    delta["final_output"] = consensus_output
    delta["final_score"] = state.get("best_score", 0.0)
    return delta
