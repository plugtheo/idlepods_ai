"""
Hybrid Supervisor — safety-first, plan-driven with selective LLM routing.

Deterministic safety guards apply to all routing decisions except R4/R5
(no-plan open-set territory).  When targets is the full open set, an LLM
inference call selects the best next agent.  All other branches — forced
dispatches (R1/R1.5/R2a/R2b) and the evaluate-vs-converge gate (R3) — are
resolved by deterministic rules with no LLM involvement.

Rule assignment:
  R1/R1.5/R2a/R2b   single legal target   → shortcircuit, rule=R_forced
  R3                evaluate-vs-converge  → DeterministicSupervisor, rule tagged _hybrid_guard
  R4/R5 (open set)  LLM call             → on failure/budget → DeterministicSupervisor

Fallback chain for R4/R5 LLM call:
  Budget exhausted          → DeterministicSupervisor, rule suffix _budget_fallback
  Inference exception       → DeterministicSupervisor, rule suffix _exc_fallback
  No tool call / bad pick   → DeterministicSupervisor, rule suffix _llm_fallback

All fallbacks and guard delegations are logged at WARNING with the reason
so ops can diagnose routing quality and LLM skip rate.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from ..clients.inference import get_inference_client
from .state import AgentState
from .supervisor import (
    DeterministicSupervisor,
    SupervisorDecision,
    _legal_targets,
)
from .supervisor_llm import (
    _build_route_to_tool,
    _build_supervisor_prompt,
    _llm_calls_used,
    _parse_route_to_call,
)
from .supervisor_safety import is_token_budget_exhausted, token_spend_total

logger = logging.getLogger(__name__)

# R3 target set: both elements must be present for the deterministic guard to fire.
_R3_TARGETS = frozenset({"review_critic", "check_convergence"})


class HybridSupervisor:
    """
    Safety-first hybrid supervisor.

    The LLM is only consulted for R4/R5 open-set routing (no plan), where
    contextual awareness of task intent and agent history matters most.
    All plan-driven and tool-executor paths use deterministic rules.

    Fallback hierarchy (all logged at WARNING level):
      1. Single legal target  → shortcircuit, rule=R_forced, no LLM
      2. R3 targets           → DeterministicSupervisor, rule+_hybrid_guard, no LLM
      3. Budget exhausted     → DeterministicSupervisor, rule+_budget_fallback
      4. Inference exception  → DeterministicSupervisor, rule+_exc_fallback
      5. No tool call / bad pick → DeterministicSupervisor, rule+_llm_fallback
    """

    async def decide(self, state: AgentState) -> SupervisorDecision:
        session_id = (state.get("session_id") or "")[:8]
        targets = _legal_targets(state)

        logger.debug(
            "[%s] hybrid_supervisor.decide targets=%s iter=%d",
            session_id, sorted(targets), state.get("current_iteration", 1),
        )

        # ── Single target: forced dispatch (R1/R1.5/R2a/R2b) — no LLM ───────
        if len(targets) == 1:
            target = next(iter(targets))
            logger.debug(
                "[%s] hybrid_supervisor.decide shortcircuit target=%r",
                session_id, target,
            )
            return SupervisorDecision(
                next_node=target,
                reason="single_valid_target",
                rule="R_forced",
                metadata={"shortcircuit": True, "llm_called": False},
            )

        # ── R3: evaluate-vs-converge — deterministic guard, no LLM ───────────
        if targets == _R3_TARGETS:
            logger.debug(
                "[%s] hybrid_supervisor.decide r3_terminal_guard targets=%s; "
                "delegating to deterministic",
                session_id, sorted(targets),
            )
            return await self._deterministic_guard(state, "r3_terminal_guard", "_hybrid_guard")

        # ── R4/R5: open target set — attempt LLM routing ──────────────────────
        from ..config.settings import settings as _settings
        decisions = state.get("supervisor_decisions") or []

        budget = _settings.pipeline_supervisor_llm_calls_per_run
        used = _llm_calls_used(state)
        if used >= budget:
            logger.warning(
                "[%s] hybrid_supervisor.decide budget_exhausted used=%d/%d; fallback",
                session_id, used, budget,
            )
            return await self._deterministic_guard(state, "budget_exhausted", "_budget_fallback")

        token_budget = _settings.pipeline_supervisor_token_budget_per_run
        if is_token_budget_exhausted(decisions, token_budget):
            logger.warning(
                "[%s] hybrid_supervisor.decide token_budget_exhausted spend=%d/%d; fallback",
                session_id, token_spend_total(decisions), token_budget,
            )
            return await self._deterministic_guard(state, "token_budget_exhausted", "_budget_fallback")

        _t0 = time.monotonic()
        try:
            decision = await self._call_llm(state, targets, session_id)
        except Exception as exc:
            _elapsed_ms = (time.monotonic() - _t0) * 1000
            logger.warning(
                "[%s] hybrid_supervisor.decide llm_exception exc=%r elapsed_ms=%.1f; fallback",
                session_id, exc, _elapsed_ms,
            )
            return await self._deterministic_guard(
                state, f"exc:{type(exc).__name__}", "_exc_fallback"
            )

        _elapsed_ms = (time.monotonic() - _t0) * 1000
        logger.debug(
            "[%s] hybrid_supervisor.decide llm_call_done elapsed_ms=%.1f decision=%s",
            session_id, _elapsed_ms, decision,
        )

        if decision is not None:
            return decision

        return await self._deterministic_guard(state, "no_valid_decision", "_llm_fallback")

    async def _call_llm(
        self,
        state: AgentState,
        targets: frozenset,
        session_id: str,
    ) -> Optional[SupervisorDecision]:
        """
        Execute the routing inference call and parse the route_to() tool response.

        Tags decisions with rule="R_hybrid_llm" to distinguish from LLMSupervisor's
        "R_llm" in supervisor_decisions history.  Returns None on parse failure or
        missing tool call.  Raises on network/HTTP errors so the caller falls back.
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
            "[%s] hybrid_supervisor._call_llm elapsed_ms=%.1f has_tool_calls=%s "
            "tokens_generated=%d content_preview=%r",
            session_id, _elapsed_ms,
            bool(response.tool_calls),
            response.tokens_generated,
            (response.content or "")[:80],
        )

        if not response.tool_calls:
            logger.warning(
                "[%s] hybrid_supervisor._call_llm no tool_calls returned; content=%r",
                session_id, (response.content or "")[:200],
            )
            return None

        tc = response.tool_calls[0]
        parsed = _parse_route_to_call(tc, targets, session_id)
        if parsed is None:
            return None

        # Retag rule for hybrid attribution; attach token count for budget tracking.
        return SupervisorDecision(
            next_node=parsed.next_node,
            reason=parsed.reason,
            rule="R_hybrid_llm",
            metadata={
                **(parsed.metadata or {}),
                "tokens_generated": response.tokens_generated,
            },
        )

    @staticmethod
    async def _deterministic_guard(
        state: AgentState,
        reason: str,
        rule_suffix: str = "",
    ) -> SupervisorDecision:
        """
        Delegate to DeterministicSupervisor and annotate the result with the guard reason.

        The rule_suffix (e.g. '_hybrid_guard', '_budget_fallback') is appended to the
        DeterministicSupervisor rule so callers can distinguish guard-delegated decisions
        from native deterministic ones in the supervisor_decisions log.
        """
        det = DeterministicSupervisor()
        dec = await det.decide(state)
        rule = dec.rule + rule_suffix if rule_suffix else dec.rule
        return SupervisorDecision(
            next_node=dec.next_node,
            reason=dec.reason,
            rule=rule,
            metadata={
                **(dec.metadata or {}),
                "llm_called": False,
                "guard_reason": reason,
            },
        )
