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
from shared.contracts.models import load_registry
from shared.contracts.agent_prompts import PLAN_STEP_SYSTEM_TEMPLATE
from ..clients.inference import get_inference_client
from ..config.settings import AGENT_PROMPTS, settings
from ..tools.runner import build_tool_schemas, execute_tool_call
from ..utils.inference_optimizer import InferenceOptimizer
from ..utils.scoring import validate_output
from .state import AgentState

# Module-level singleton — shared across all node invocations in this process.
# Flags read once at startup; change requires service restart (or env reload).
_optimizer = InferenceOptimizer(
    role_history_filter=settings.optimize_role_history_filter,
    structured_extraction=settings.optimize_structured_extraction,
)

logger = logging.getLogger(__name__)

ROLE_NAME_OVERHEAD_TOKENS = 20
LABEL_PADDING_TOKENS = 5

_TOOL_USING_ROLES = {"coder"}
_MAX_TOOL_STEPS = 8


def _default_backend() -> str:
    try:
        return load_registry().default_backend
    except Exception:
        return "primary"


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
    hints                = state.get("system_hints", "")
    few_shots            = state.get("few_shots", [])
    repo_snippets        = state.get("repo_snippets", [])
    user_prompt          = state.get("user_prompt", "")
    current_iteration    = state.get("current_iteration", 1)
    conversation_history = state.get("conversation_history", [])

    # Separate tool-loop entries (coder+tool_calls pairs and tool results) so
    # they can be injected as proper OpenAI messages rather than as prose labels.
    raw_history = state.get("iteration_history", [])
    tool_loop_entries = [
        h for h in raw_history
        if h.get("role") == "tool"
        or (h.get("role") == role and "tool_calls" in h)
    ]
    regular_raw = [h for h in raw_history if h not in tool_loop_entries]
    iteration_history = _optimizer.filter_history(role, regular_raw)

    model_context_len = settings.model_context_len
    role_max_tokens   = settings.role_max_tokens.get(role, 1024)

    # Build the two non-negotiable messages first so we know their token cost.
    system_msg = _system_message(role, hints)
    user_msg   = Message(role="user", content=user_prompt)

    fixed_tokens = (
        _estimate_tokens(system_msg.content or "")
        + _estimate_tokens(user_msg.content or "")
        + ROLE_NAME_OVERHEAD_TOKENS
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

    repo_budget_cap    = int(remaining_budget * settings.context_budget_repo_ratio)
    fewshot_budget_cap = int(remaining_budget * max(0.0, 1.0 - settings.context_budget_history_ratio - settings.context_budget_repo_ratio))
    fewshot_used       = 0
    repo_used          = 0

    messages: List[Message] = [system_msg]

    # ── Plan-step context (extra system message — never concatenated into base prompt) ──
    current_step_id = state.get("current_step_id")
    if current_step_id and role in ("coder", "debugger"):
        plan_dict = state.get("plan") or {}
        step_desc = next(
            (s["description"] for s in plan_dict.get("steps", []) if s["id"] == current_step_id),
            "",
        )
        if step_desc:
            messages.append(Message(
                role="system",
                content=PLAN_STEP_SYSTEM_TEMPLATE.format(
                    current_step_id=current_step_id,
                    current_step_description=step_desc,
                ),
            ))

    # ── Few-shot block ────────────────────────────────────────────────────
    if few_shots and fewshot_budget_cap > 0:
        lines = ["Here are relevant past solutions for reference:"]
        used  = _estimate_tokens("\n".join(lines))
        # [:4] is a defensive guard; upstream retrieval already caps at settings.max_few_shots
        for i, ex in enumerate(few_shots[:4], 1):
            problem  = str(ex.get("problem",  ""))[:settings.fewshot_problem_max_chars]
            solution = str(ex.get("solution", ""))[:settings.fewshot_solution_max_chars]
            score    = ex.get("score", 0.0)
            entry    = (
                f"\nExample {i} (quality score {score:.2f}):\n"
                f"Task: {problem}\nSolution: {solution}"
            )
            entry_tokens = _estimate_tokens(entry)
            if used + entry_tokens > fewshot_budget_cap:
                break
            lines.append(entry)
            used += entry_tokens
        if len(lines) > 1:
            messages.append(Message(role="user",      content="\n".join(lines)))
            messages.append(Message(role="assistant", content="Understood. I'll use these examples as reference."))
            fewshot_used = used

    # ── Repo snippets block ───────────────────────────────────────────────
    if repo_snippets and repo_budget_cap > 0 and role in ("coder", "debugger", "reviewer"):
        repo_lines = ["Relevant code from the repository:"]
        used       = _estimate_tokens("\n".join(repo_lines))
        # [:5] is a defensive guard; upstream retrieval already caps at settings.max_repo_snippets
        for snip in repo_snippets[:5]:
            entry        = f"[{snip.get('file', '?')}]: {snip.get('snippet', '')[:settings.repo_snippet_max_chars]}"
            entry_tokens = _estimate_tokens(entry)
            if used + entry_tokens > repo_budget_cap:
                break
            repo_lines.append(entry)
            used += entry_tokens
        if len(repo_lines) > 1:
            messages.append(Message(role="user",      content="\n".join(repo_lines)))
            messages.append(Message(role="assistant", content="I'll keep this repo context in mind."))
            repo_used = used

    # History gets base allocation plus any surplus freed by repo/fewshot blocks.
    history_budget   = remaining_budget - fewshot_used - repo_used
    conv_hist_budget = int(history_budget * settings.context_budget_conv_history_ratio)
    iter_hist_budget = history_budget - conv_hist_budget

    # ── Conversation history block (cross-turn turns loaded from Redis) ───
    if conversation_history and conv_hist_budget > 0:
        conv_lines  = ["Prior conversation context:"]
        eligible: List[str] = []
        used        = 0
        recent_conv = conversation_history[-settings.max_conversation_turns:]
        for h in reversed(recent_conv):
            h_role    = h.get("role", "agent")
            output    = str(h.get("output") or h.get("full_output", ""))
            label     = f"[prev turn — {h_role}]: "
            available = conv_hist_budget - used - _estimate_tokens(label) - LABEL_PADDING_TOKENS
            if available <= 0:
                break
            if _estimate_tokens(output) > available:
                char_budget = available * settings.chars_per_token
                output      = "…" + output[-char_budget:]
            eligible.append(label + output)
            used += _estimate_tokens(label + output)
        if eligible:
            conv_lines.extend(reversed(eligible))
            messages.append(Message(role="user",      content="\n".join(conv_lines)))
            messages.append(Message(role="assistant", content="I have the prior conversation context."))

    # ── History block (recency-biased, newest-first) ──────────────────────
    if iteration_history and iter_hist_budget > 0:
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
            available = iter_hist_budget - used - _estimate_tokens(label) - LABEL_PADDING_TOKENS
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

    # ── User task ─────────────────────────────────────────────────────────
    # Must precede any tool-loop entries: OpenAI conversation order is
    # user → assistant(tool_calls) → tool(result) → assistant(next).
    # Appending user AFTER the tool turns confuses the model into treating
    # the original request as a fresh turn and re-emitting the same tool call.
    messages.append(user_msg)

    # ── Tool-loop messages (OpenAI assistant+tool pairs, chronological) ──
    if tool_loop_entries and role in _TOOL_USING_ROLES:
        for entry in tool_loop_entries:
            if "tool_calls" in entry:
                messages.append(Message(role="assistant", tool_calls=entry["tool_calls"]))
            else:
                messages.append(Message(
                    role="tool",
                    tool_call_id=entry.get("tool_call_id", ""),
                    content=entry.get("output", ""),
                    name=entry.get("name"),
                ))

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

    request_kwargs: Dict[str, Any] = dict(
        backend=settings.role_backend.get(role) or _default_backend(),
        role=role,
        messages=messages,
        adapter_name=settings.role_adapter.get(role),
        max_tokens=settings.role_max_tokens.get(role, 1024),
        session_id=session_id,
    )
    if role in _TOOL_USING_ROLES:
        request_kwargs["tools"] = build_tool_schemas()

    request = GenerateRequest(**request_kwargs)

    response = None
    try:
        client = get_inference_client()
        q = get_token_queue(session_id)

        # Tool-using roles always use the blocking path so response.tool_calls is available.
        if q is not None and hasattr(client, "generate_stream") and role not in _TOOL_USING_ROLES:
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
                output = "".join(tokens) if tokens else f"[{role} agent unavailable: {exc}]"
        else:
            response = await client.generate(request)
            output = response.content
            logger.info(
                "[%s] iter=%d  role=%s  tokens=%d",
                session_id[:8], current_iteration, role, response.tokens_generated,
            )
    except Exception as exc:
        logger.error("[%s] Inference failed for role=%s: %s", session_id[:8], role, exc)
        output = f"[{role} agent unavailable: {exc}]"

    # Post-generation validator: prevent downstream agents seeing garbage output.
    # full_output retains the original so the convergence scorer still operates
    # on the real text (leakage detection, SCORE: extraction, etc.).
    # Skip validation when the response carries native tool_calls — a tool-call
    # turn legitimately has empty/short content (the payload lives in tool_calls,
    # not text), so prose-oriented checks like short_text don't apply.
    has_tool_calls = response is not None and response.tool_calls
    if has_tool_calls:
        display_output = output
    else:
        _valid, _reasons = validate_output(output, role)
        if not _valid:
            logger.warning(
                "[%s] iter=%d  role=%s  validation_failed reasons=%s",
                session_id[:8], current_iteration, role, _reasons,
            )
            display_output = f"[VALIDATOR_FAIL:{';'.join(_reasons)}]"
        else:
            display_output = output

    # Structured extraction: store compact key fields in history so downstream
    # agents' input token cost is lower.  Full output kept as last_output so
    # the convergence scorer can still read SCORE: annotations.
    stored_output = _optimizer.extract_for_history(role, display_output)

    history_entry = {
        "role": role,
        "iteration": current_iteration,
        "output": stored_output,       # compact/sentinel version seen by downstream agents
        "full_output": output,         # original LLM response — scorer and training use this
        "messages": [m.model_dump() for m in messages],  # full prompt sent to LLM — SFT training pair
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }

    updated_history = list(state.get("iteration_history", [])) + [history_entry]

    # Detect native tool calls in the response for tool-using roles.
    native_tool_calls = response.tool_calls if response is not None else None
    if native_tool_calls:
        history_entry["tool_calls"] = native_tool_calls
        history_entry["output"] = ""
        return {
            "iteration_history": updated_history,
            "last_output": "",
            "pending_tool_calls": native_tool_calls,
            "tool_originating_role": role,
        }

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
    plan_dict = state.get("plan")

    if plan_dict:
        # Plan already exists — find the next pending step and mark in_progress.
        steps = plan_dict.get("steps", [])
        next_step = next((s for s in steps if s["status"] == "pending"), None)
        if next_step is None:
            # All steps done or blocked — let the pipeline continue.
            return await _run_agent_node("planner", state)

        import copy
        from datetime import datetime, timezone as _tz
        updated = copy.deepcopy(plan_dict)
        for s in updated["steps"]:
            if s["id"] == next_step["id"]:
                s["status"] = "in_progress"
                break
        updated["updated_at"] = datetime.now(_tz.utc).isoformat()
        step_id = next_step["id"]
        desc = next_step["description"]
        return {
            "plan": updated,
            "plan_changed": True,
            "current_step_id": step_id,
            "last_output": f"[planner] advancing to {step_id}: {desc}",
            "agent_chain_index": state.get("agent_chain_index", 0) + 1,
        }

    # No plan yet — run LLM to produce one, then try to parse it.
    delta = await _run_agent_node("planner", state)
    output = delta.get("last_output", "")
    parsed = _try_parse_plan_output(output)
    if parsed:
        try:
            from datetime import datetime as _dt, timezone as _tz
            from ..plans.schema import Plan
            now_iso = _dt.now(_tz.utc).isoformat()
            parsed.setdefault("created_at", now_iso)
            parsed.setdefault("updated_at", now_iso)
            plan = Plan.model_validate(parsed)
            delta["plan"] = plan.model_dump(mode="json")
            delta["plan_changed"] = True
        except Exception as exc:
            logger.warning("planner: plan parse succeeded but validation failed: %s", exc)
    return delta


def _try_parse_plan_output(output: str) -> dict | None:
    """Opportunistic JSON plan extraction from planner output. Returns None on failure."""
    import json as _json
    import re as _re

    # Try ```json ... ``` fence first
    m = _re.search(r"```json\s*(.*?)\s*```", output, _re.DOTALL)
    if m:
        try:
            return _json.loads(m.group(1))
        except _json.JSONDecodeError:
            pass

    # Try bare object with "goal" and "steps" keys
    m = _re.search(r"\{[^{}]*\"goal\"[^{}]*\"steps\".*\}", output, _re.DOTALL)
    if m:
        try:
            return _json.loads(m.group(0))
        except _json.JSONDecodeError:
            pass
    return None


async def researcher_node(state: AgentState) -> dict:
    return await _run_agent_node("researcher", state)


async def coder_node(state: AgentState) -> dict:
    delta = await _run_agent_node("coder", state)
    return _maybe_update_plan_step(state, delta)


async def debugger_node(state: AgentState) -> dict:
    delta = await _run_agent_node("debugger", state)
    return _maybe_update_plan_step(state, delta)


def _maybe_update_plan_step(state: AgentState, delta: dict) -> dict:
    """Opportunistically mark the current plan step done/blocked based on output."""
    import copy as _copy
    import re as _re
    from datetime import datetime, timezone as _tz

    step_id = state.get("current_step_id")
    plan_dict = state.get("plan")
    if not step_id or not plan_dict:
        return delta

    output = delta.get("last_output", "")
    blocked_m = _re.search(r"\[BLOCKED:([^\]]*)\]", output)

    updated = _copy.deepcopy(plan_dict)
    changed = False
    for step in updated.get("steps", []):
        if step["id"] == step_id:
            if blocked_m:
                step["status"] = "blocked"
                step["evidence"] = blocked_m.group(1).strip()
                changed = True
            elif step["status"] == "in_progress":
                step["status"] = "done"
                changed = True
            break

    if changed:
        updated["updated_at"] = datetime.now(_tz.utc).isoformat()
        delta["plan"] = updated
        delta["plan_changed"] = True
    return delta


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
    consensus_output = delta.get("last_output", state.get("best_output", ""))

    # Append remaining-work section when plan steps are still pending/blocked.
    plan_dict = state.get("plan")
    if plan_dict:
        remaining = [
            s for s in plan_dict.get("steps", [])
            if s["status"] in ("pending", "blocked")
        ]
        if remaining:
            lines = ["\n\n---\n**Remaining work:**"]
            for s in remaining:
                lines.append(f"- [{s['status']}] {s['id']}: {s['description']}")
            consensus_output = consensus_output + "\n".join(lines)

    delta["final_output"] = consensus_output
    delta["final_score"] = state.get("best_score", 0.0)
    return delta


async def tool_executor_node(state: AgentState) -> dict:
    """Execute pending OpenAI-format tool calls; record each result as a role='tool' history entry."""
    pending = state.get("pending_tool_calls", [])
    tool_steps_used = state.get("tool_steps_used", 0)
    session_id = state.get("session_id", "")
    current_iteration = state.get("current_iteration", 1)

    if tool_steps_used >= _MAX_TOOL_STEPS:
        exhausted_entry = {
            "role": "tool",
            "tool_call_id": "exhausted",
            "name": "budget",
            "iteration": current_iteration,
            "output": "Tool budget exhausted — finalise your answer without further tool calls.",
            "error": False,
            "tool_step": tool_steps_used,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }
        logger.warning("[%s] Tool budget exhausted (max=%d)", session_id[:8], _MAX_TOOL_STEPS)
        return {
            "iteration_history": list(state.get("iteration_history", [])) + [exhausted_entry],
            "pending_tool_calls": [],
        }

    tool_entries = []
    tool_names = []
    for call in pending:
        r = execute_tool_call(call)
        tool_names.append(r["tool"])
        entry = {
            "role": "tool",
            "tool_call_id": call.get("id", ""),
            "name": r["tool"],
            "iteration": current_iteration,
            "output": r["output"],
            "error": r["error"],
            "tool_step": tool_steps_used + 1,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }
        tool_entries.append(entry)

    logger.info(
        "[%s] tool_executor: step=%d  tools=%s",
        session_id[:8], tool_steps_used + 1, tool_names,
    )

    combined = "\n".join(e["output"] for e in tool_entries)
    return {
        "iteration_history": list(state.get("iteration_history", [])) + tool_entries,
        "pending_tool_calls": [],
        "tool_steps_used": tool_steps_used + 1,
        "last_output": combined,
    }
