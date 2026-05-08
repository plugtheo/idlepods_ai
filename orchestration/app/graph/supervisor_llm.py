"""
LLM Supervisor — inference-driven agent dispatch.

Implements the Supervisor protocol using a single routing inference call per
turn.  Forced dispatches (R1 pending-tool, R1.5 post-tool, R2a in-progress
step, R2b pending-step-planner) bypass the LLM entirely; the LLM is only
consulted when the legal target set has more than one element (R3/R4/R5
territory).

Fallback chain on any failure:
  LLM call fails / no tool call / invalid pick → DeterministicSupervisor
  Budget exhausted → DeterministicSupervisor

All fallbacks are logged with the reason so ops can diagnose routing quality.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from ..clients.inference import get_inference_client
from .signal_extractor import extract_signals, format_signals_for_prompt
from .state import AgentState
from .supervisor import (
    DeterministicSupervisor,
    SupervisorDecision,
    _VALID_WORKER_ROLES,
    _legal_targets,
)
from .supervisor_safety import is_token_budget_exhausted, token_spend_total

logger = logging.getLogger(__name__)


# ── Budget tracker ────────────────────────────────────────────────────────────


def _llm_calls_used(state: AgentState) -> int:
    """Count actual LLM inference calls made by this supervisor in the current run."""
    decisions = state.get("supervisor_decisions") or []
    return sum(
        1 for d in decisions
        if d.get("metadata", {}).get("llm_called") is True
    )


# ── Prompt and tool-schema builders ──────────────────────────────────────────


def _build_supervisor_prompt(state: AgentState, targets: frozenset) -> str:
    """Build the system prompt for the LLM routing call."""
    from ..config.settings import settings as _settings

    lines: List[str] = [
        "You are a supervisor deciding which specialist agent runs next.",
        "Choose the agent that will make the most progress given the current state.",
        "",
    ]

    # User task
    user_prompt = (state.get("user_prompt") or "")[:300]
    lines.append(f"## User Task\n{user_prompt}\n")

    # Iteration progress
    current_iter = state.get("current_iteration", 1)
    max_iter = state.get("max_iterations", _settings.default_max_iterations)
    best_score = state.get("best_score", 0.0)
    lines.append(
        f"## Progress\n"
        f"Iteration {current_iter}/{max_iter}  |  best_score={best_score:.2f}/1.00\n"
    )

    # Plan summary
    plan = state.get("plan")
    if plan:
        steps: List[Dict[str, Any]] = plan.get("steps") or []
        done_n = sum(1 for s in steps if s.get("status") == "done")
        in_prog_n = sum(1 for s in steps if s.get("status") == "in_progress")
        pending_n = sum(1 for s in steps if s.get("status") == "pending")
        blocked_n = sum(1 for s in steps if s.get("status") == "blocked")
        lines.append(
            f"## Plan ({len(steps)} steps: {done_n} done, {in_prog_n} in_progress, "
            f"{pending_n} pending, {blocked_n} blocked)"
        )
        for s in steps:
            desc = (s.get("description") or "")[:80]
            lines.append(
                f"  [{s.get('status', '?')}] {s.get('id', '?')}: {desc} "
                f"(owner: {s.get('owner_role', '?')})"
            )
        lines.append("")

    # Recent agent work
    history: List[Dict[str, Any]] = state.get("iteration_history") or []
    window = _settings.pipeline_supervisor_history_window
    recent = history[-window:] if len(history) > window else history
    if recent:
        lines.append("## Recent Agent Work (newest last)")
        for h in recent:
            role = h.get("role", "?")
            output = (h.get("output") or h.get("full_output") or "")[:200]
            lines.append(f"  [iter {h.get('iteration', '?')}] {role}: {output}")
        lines.append("")

    # Output signals extracted from the last agent's text
    signals = extract_signals(state)
    signal_section = format_signals_for_prompt(signals)
    if signal_section:
        lines.append(signal_section)
        lines.append("")

    # Available next agents
    targets_str = ", ".join(sorted(targets))
    lines.append(f"## Valid Next Agents\n{targets_str}\n")
    lines.append("Call route_to() with the best next agent and a brief reason.")

    return "\n".join(lines)


def _build_route_to_tool(targets: frozenset) -> dict:
    """Build the OpenAI-compatible route_to tool definition."""
    return {
        "type": "function",
        "function": {
            "name": "route_to",
            "description": "Select which specialist agent to dispatch next.",
            "parameters": {
                "type": "object",
                "properties": {
                    "next_node": {
                        "type": "string",
                        "enum": sorted(targets),
                        "description": "The agent or node to route to next.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why this agent should run next (1-2 sentences).",
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence in this routing decision (0.0–1.0).",
                    },
                },
                "required": ["next_node", "reason", "confidence"],
            },
        },
    }


# ── Tool-call parser ──────────────────────────────────────────────────────────


def _parse_route_to_call(
    tc: Any,
    targets: frozenset,
    session_id: str,
) -> Optional[SupervisorDecision]:
    """
    Parse a single OpenAI-format tool call dict into a SupervisorDecision.

    Returns None if the call cannot be parsed or the chosen node is not legal.
    """
    try:
        if isinstance(tc, dict):
            fn = tc.get("function", {})
            name = fn.get("name", "")
            args_raw = fn.get("arguments", "{}")
        else:
            fn = getattr(tc, "function", None)
            name = getattr(fn, "name", "") if fn else ""
            args_raw = getattr(fn, "arguments", "{}") if fn else "{}"

        if name != "route_to":
            logger.warning(
                "[%s] llm_supervisor.parse unexpected_tool=%r; ignoring",
                session_id, name,
            )
            return None

        args: Dict[str, Any] = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        next_node = str(args.get("next_node", "")).strip()
        reason = str(args.get("reason", "llm_routing")).strip()
        confidence = float(args.get("confidence", 0.0))

        if next_node not in targets:
            logger.warning(
                "[%s] llm_supervisor.parse invalid_target=%r valid=%s",
                session_id, next_node, sorted(targets),
            )
            return None

        logger.info(
            "[%s] llm_supervisor.parse next_node=%r confidence=%.2f reason=%r",
            session_id, next_node, confidence, reason[:120],
        )
        return SupervisorDecision(
            next_node=next_node,
            reason=reason[:200],
            rule="R_llm",
            metadata={"confidence": confidence, "llm_called": True},
        )

    except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
        logger.warning(
            "[%s] llm_supervisor.parse failed exc=%r", session_id, exc,
        )
        return None


# ── LLMSupervisor ─────────────────────────────────────────────────────────────


class LLMSupervisor:
    """
    Inference-driven supervisor.  Calls the Inference Service once per ambiguous
    routing decision.  Forced dispatches short-circuit before the LLM call.

    Fallback hierarchy (all logged at WARNING level):
      1. Single legal target  → shortcircuit, rule=R_forced
      2. Budget exhausted     → DeterministicSupervisor, rule suffix _budget_fallback
      3. Inference exception  → DeterministicSupervisor, rule suffix _exc_fallback
      4. No tool call         → DeterministicSupervisor, rule suffix _notc_fallback
      5. Invalid target pick  → DeterministicSupervisor, rule suffix _badpick_fallback
    """

    async def decide(self, state: AgentState) -> SupervisorDecision:
        session_id = (state.get("session_id") or "")[:8]

        # ── Compute valid targets ──────────────────────────────────────────
        targets = _legal_targets(state)
        logger.debug(
            "[%s] llm_supervisor.decide targets=%s iter=%d",
            session_id, sorted(targets), state.get("current_iteration", 1),
        )

        # ── Single target: no LLM call needed ─────────────────────────────
        if len(targets) == 1:
            target = next(iter(targets))
            logger.debug(
                "[%s] llm_supervisor.decide shortcircuit target=%r",
                session_id, target,
            )
            return SupervisorDecision(
                next_node=target,
                reason="single_valid_target",
                rule="R_forced",
                metadata={"shortcircuit": True, "llm_called": False},
            )

        # ── Budget checks ──────────────────────────────────────────────────
        from ..config.settings import settings as _settings
        decisions = state.get("supervisor_decisions") or []

        budget = _settings.pipeline_supervisor_llm_calls_per_run
        used = _llm_calls_used(state)
        if used >= budget:
            logger.warning(
                "[%s] llm_supervisor.decide budget_exhausted used=%d/%d; fallback",
                session_id, used, budget,
            )
            return await self._fallback(state, "budget_exhausted", "_budget_fallback")

        token_budget = _settings.pipeline_supervisor_token_budget_per_run
        if is_token_budget_exhausted(decisions, token_budget):
            logger.warning(
                "[%s] llm_supervisor.decide token_budget_exhausted spend=%d/%d; fallback",
                session_id, token_spend_total(decisions), token_budget,
            )
            return await self._fallback(state, "token_budget_exhausted", "_budget_fallback")

        # ── LLM inference call ─────────────────────────────────────────────
        _t0 = time.monotonic()
        try:
            decision = await self._call_llm(state, targets, session_id)
        except Exception as exc:
            _elapsed_ms = (time.monotonic() - _t0) * 1000
            logger.warning(
                "[%s] llm_supervisor.decide llm_exception exc=%r elapsed_ms=%.1f; fallback",
                session_id, exc, _elapsed_ms,
            )
            return await self._fallback(state, f"exc:{type(exc).__name__}", "_exc_fallback")

        _elapsed_ms = (time.monotonic() - _t0) * 1000
        logger.debug(
            "[%s] llm_supervisor.decide llm_call_done elapsed_ms=%.1f decision=%s",
            session_id, _elapsed_ms, decision,
        )

        if decision is not None:
            return decision

        # ── Fallback: LLM gave no usable decision ─────────────────────────
        return await self._fallback(state, "no_valid_decision", "_llm_fallback")

    async def _call_llm(
        self,
        state: AgentState,
        targets: frozenset,
        session_id: str,
    ) -> Optional[SupervisorDecision]:
        """
        Execute the inference call and parse the route_to() tool response.

        Returns a SupervisorDecision on success, None on parse failure or
        missing tool call.  Raises on network/HTTP errors so the caller can
        log and fall back.
        """
        from ..config.settings import settings as _settings
        from shared.contracts.inference import GenerateRequest, Message, ToolDefinition

        _t0 = time.monotonic()

        system_prompt = _build_supervisor_prompt(state, targets)
        tool_schema = _build_route_to_tool(targets)

        request = GenerateRequest(
            backend=_settings.pipeline_supervisor_backend,
            role="supervisor",
            messages=[
                Message(role="system", content=system_prompt),
                Message(
                    role="user",
                    content="Route to the most appropriate next agent given the current state.",
                ),
            ],
            max_tokens=_settings.pipeline_supervisor_max_tokens,
            temperature=0.0,
            session_id=state.get("session_id"),
            tools=[ToolDefinition(**tool_schema)],
        )

        client = get_inference_client()
        response = await client.generate(request)
        _elapsed_ms = (time.monotonic() - _t0) * 1000

        logger.info(
            "[%s] llm_supervisor._call_llm elapsed_ms=%.1f has_tool_calls=%s "
            "tokens_generated=%d content_preview=%r",
            session_id, _elapsed_ms,
            bool(response.tool_calls),
            response.tokens_generated,
            (response.content or "")[:80],
        )

        if not response.tool_calls:
            logger.warning(
                "[%s] llm_supervisor._call_llm no tool_calls returned; "
                "content=%r",
                session_id, (response.content or "")[:200],
            )
            return None

        tc = response.tool_calls[0]
        parsed = _parse_route_to_call(tc, targets, session_id)
        if parsed is None:
            return None

        # Attach token count so supervisor_safety can track the per-run token spend.
        return SupervisorDecision(
            next_node=parsed.next_node,
            reason=parsed.reason,
            rule=parsed.rule,
            metadata={
                **(parsed.metadata or {}),
                "tokens_generated": response.tokens_generated,
            },
        )

    @staticmethod
    async def _fallback(
        state: AgentState,
        reason: str,
        rule_suffix: str,
    ) -> SupervisorDecision:
        """Run DeterministicSupervisor and tag the result with a fallback marker."""
        det = DeterministicSupervisor()
        dec = await det.decide(state)
        return SupervisorDecision(
            next_node=dec.next_node,
            reason=dec.reason,
            rule=dec.rule + rule_suffix,
            metadata={
                **(dec.metadata or {}),
                "llm_called": False,
                "fallback_reason": reason,
            },
        )
