"""
Supervisor — plan-aware agent dispatch for the supervisor pipeline.

Replaces static agent_chain[index] advancement with a strategy-driven decision
that reads plan state, tool-call state, and iteration history to decide
which agent node runs next.

Supervisor protocol:
  Every concrete supervisor must implement ``async decide(state) → SupervisorDecision``.

Implementations (selected via settings.pipeline_supervisor_strategy):
  deterministic  — pure rule-based, no LLM call (default).
  llm            — single inference call per ambiguous transition (R3/R4/R5).
  hybrid         — deterministic safety guards + LLM only for R4/R5 open-set routing.

Rule priority in DeterministicSupervisor.decide():
  R1   — pending_tool_calls → tool_executor (preserve ReAct)
  R1.5 — just returned from tool_executor → resume originating role
  R2a  — plan has an in_progress step → dispatch to its owner_role
  R2b  — plan has a pending step → route to planner (advances step to in_progress)
  R3   — plan all terminal → evaluate (review_critic) then converge
  R4   — no plan → use static agent_chain as template hint
  R5   — chain exhausted → evaluate then converge
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from .state import AgentState

logger = logging.getLogger(__name__)

# Roles the supervisor is permitted to dispatch to directly.
# "consensus" is excluded — it is reached via finalize, never via supervisor.
_VALID_WORKER_ROLES = frozenset({
    "planner", "researcher", "coder", "debugger",
    "reviewer", "critic", "review_critic",
})

_EVALUATOR_ROLES = frozenset({"reviewer", "critic", "review_critic"})


@dataclass
class SupervisorDecision:
    next_node: str
    reason: str
    rule: str
    metadata: Optional[Dict[str, Any]] = None


@runtime_checkable
class Supervisor(Protocol):
    """
    Strategy interface for all supervisor implementations.

    Each implementation must be stateless across calls — all routing context
    is derived from the AgentState snapshot passed to decide().
    """

    async def decide(self, state: AgentState) -> SupervisorDecision: ...


class DeterministicSupervisor:
    """
    Rule-based supervisor.  Pure function — no I/O, no LLM calls.

    All routing decisions are deterministic given the same AgentState snapshot.
    """

    async def decide(self, state: AgentState) -> SupervisorDecision:
        session_id = (state.get("session_id") or "")[:8]
        history: List[Dict[str, Any]] = state.get("iteration_history") or []
        current_iter: int = state.get("current_iteration", 1)

        # ── R1: pending tool call → tool_executor (preempts all other routing) ──
        if state.get("pending_tool_calls"):
            return SupervisorDecision("tool_executor", "tool_call_pending", "R1")

        # ── R1.5: just returned from tool_executor → resume originating role ──
        # Detect by checking whether the most recent history entry is a tool result.
        if history and history[-1].get("role") == "tool":
            originator = state.get("tool_originating_role", "")
            if originator in _VALID_WORKER_ROLES:
                return SupervisorDecision(
                    originator,
                    "tool_result_consume",
                    "R1.5",
                    {"originator": originator},
                )
            logger.warning(
                "[%s] supervisor R1.5: tool_originating_role=%r not in valid roles; "
                "falling through to plan rules",
                session_id, originator,
            )

        plan = state.get("plan")

        # ── R2: plan-driven dispatch ──────────────────────────────────────────
        if plan:
            steps: List[Dict[str, Any]] = plan.get("steps") or []

            # R2a: a step is already in_progress → dispatch its worker
            in_prog = next((s for s in steps if s.get("status") == "in_progress"), None)
            if in_prog:
                owner = (in_prog.get("owner_role") or "").strip()
                if owner in _VALID_WORKER_ROLES:
                    return SupervisorDecision(
                        owner,
                        "plan_step_in_progress",
                        "R2a",
                        {"step_id": in_prog["id"]},
                    )
                logger.warning(
                    "[%s] supervisor R2a: step %s has invalid owner_role=%r; falling through",
                    session_id, in_prog.get("id"), in_prog.get("owner_role"),
                )

            # R2b: pending steps remain → planner advances the next one to in_progress
            if any(s.get("status") == "pending" for s in steps):
                return SupervisorDecision(
                    "planner",
                    "plan_advance_next_step",
                    "R2b",
                )

            # R3: all steps are terminal (done or blocked)
            if all(s.get("status") in ("done", "blocked") for s in steps):
                if not self._evaluator_ran_this_iter(history, current_iter):
                    blocked = [s["id"] for s in steps if s.get("status") == "blocked"]
                    return SupervisorDecision(
                        "review_critic",
                        "plan_complete_evaluate",
                        "R3",
                        {"blocked_steps": blocked},
                    )
                return SupervisorDecision(
                    "check_convergence",
                    "plan_complete_converged",
                    "R3",
                )

        # ── R4: no plan — use the static agent_chain template hint ───────────
        chain: List[str] = state.get("agent_chain") or []
        chain_index: int = state.get("agent_chain_index", 0)
        if chain and chain_index < len(chain):
            role = chain[chain_index]
            if role in _VALID_WORKER_ROLES:
                return SupervisorDecision(
                    role,
                    "chain_template_dispatch",
                    "R4",
                    {"chain_index": chain_index},
                )
            logger.warning(
                "[%s] supervisor R4: chain[%d]=%r is not a valid worker role; skipping",
                session_id, chain_index, role,
            )

        # ── R5: chain exhausted or empty — evaluate then converge ─────────────
        if not self._evaluator_ran_this_iter(history, current_iter):
            return SupervisorDecision(
                "review_critic",
                "chain_complete_evaluate",
                "R5",
            )
        return SupervisorDecision(
            "check_convergence",
            "chain_complete_converged",
            "R5",
        )

    @staticmethod
    def _evaluator_ran_this_iter(history: List[Dict[str, Any]], iteration: int) -> bool:
        return any(
            h.get("iteration") == iteration and h.get("role") in _EVALUATOR_ROLES
            for h in history
        )


def _legal_targets(state: AgentState) -> frozenset:
    """
    Return the frozenset of valid next_node values for the current state.

    Single-element sets indicate a forced dispatch (R1, R1.5, R2a, R2b).
    Multi-element sets are the cases where LLM routing adds value over
    the deterministic rules (R3 evaluate-vs-converge, R4/R5 no-plan territory).
    """
    # R1: pending tool calls — forced
    if state.get("pending_tool_calls"):
        return frozenset({"tool_executor"})

    # R1.5: just returned from tool executor — forced to resume originator
    history: List[Dict[str, Any]] = state.get("iteration_history") or []
    if history and history[-1].get("role") == "tool":
        originator = state.get("tool_originating_role", "")
        if originator in _VALID_WORKER_ROLES:
            return frozenset({originator})

    plan = state.get("plan")
    if plan:
        steps: List[Dict[str, Any]] = plan.get("steps") or []

        # R2a: step already in_progress — forced to its owner
        in_prog = next((s for s in steps if s.get("status") == "in_progress"), None)
        if in_prog:
            owner = (in_prog.get("owner_role") or "").strip()
            if owner in _VALID_WORKER_ROLES:
                return frozenset({owner})

        # R2b: pending steps remain — planner must advance (forced)
        if any(s.get("status") == "pending" for s in steps):
            return frozenset({"planner"})

        # R3: all terminal — evaluate or converge (LLM may decide)
        if all(s.get("status") in ("done", "blocked") for s in steps):
            return frozenset({"review_critic", "check_convergence"})

    # R4/R5: no plan or chain-based routing — open set for LLM to choose from
    return frozenset(_VALID_WORKER_ROLES | {"check_convergence"})


def get_supervisor() -> Supervisor:
    """
    Factory that returns a Supervisor implementation selected by settings.

    Reads settings.pipeline_supervisor_strategy at call time so tests can
    override the setting without needing to reset a module-level singleton.
    Lazy imports are used for llm/hybrid to avoid circular imports at module load time.
    """
    from ..config.settings import settings
    strategy = settings.pipeline_supervisor_strategy
    if strategy == "deterministic":
        impl = DeterministicSupervisor()
        logger.debug("supervisor.factory strategy=%s impl=%s", strategy, type(impl).__name__)
        return impl
    if strategy == "llm":
        from .supervisor_llm import LLMSupervisor
        impl = LLMSupervisor()
        logger.debug("supervisor.factory strategy=%s impl=%s", strategy, type(impl).__name__)
        return impl
    if strategy == "hybrid":
        from .supervisor_hybrid import HybridSupervisor
        impl = HybridSupervisor()
        logger.debug("supervisor.factory strategy=%s impl=%s", strategy, type(impl).__name__)
        return impl
    logger.warning(
        "supervisor.factory strategy=%r unknown; falling back to deterministic",
        strategy,
    )
    return DeterministicSupervisor()


def get_supervisor_for_session(session_id: str = "") -> Supervisor:
    """
    Factory that applies rollout gating before returning a Supervisor.

    When pipeline_supervisor_rollout_pct < 100, uses an MD5 hash of session_id
    to determine whether this session receives the configured strategy or falls
    back to DeterministicSupervisor.  At rollout_pct=100 (default) all sessions
    get the configured strategy.
    """
    from ..config.settings import settings
    from .supervisor_safety import is_in_rollout

    rollout_pct = settings.pipeline_supervisor_rollout_pct
    if not is_in_rollout(session_id, rollout_pct):
        logger.info(
            "supervisor.rollout session=%r not_in_rollout pct=%d; using deterministic",
            session_id[:8], rollout_pct,
        )
        return DeterministicSupervisor()

    return get_supervisor()


async def supervisor_anchor(state: AgentState) -> dict:
    """
    Supervisor node — logs the dispatch decision and appends it to supervisor_decisions.

    This is a state-anchor node.  The routing decision itself is re-read by
    supervisor_decide (the conditional edge function) from supervisor_decisions[-1]
    so we compute it exactly once per supervisor invocation.

    Safety bounds applied (in order):
      1. Hard decision cap: if existing decisions >= max, force check_convergence.
      2. Role-loop detection: if same worker role >= max_consecutive, override to
         check_convergence.  Applied AFTER the supervisor computes its decision.
    """
    from ..config.settings import settings as _settings
    from .supervisor_safety import detect_role_loop, is_hard_decision_cap_reached

    _t0 = time.monotonic()

    full_session_id: str = state.get("session_id") or ""
    session_id: str = full_session_id[:8]
    existing: List[Dict[str, Any]] = list(state.get("supervisor_decisions") or [])

    # ── Safety: hard decision cap ─────────────────────────────────────────────
    max_decisions = _settings.pipeline_supervisor_max_decisions_per_run
    if is_hard_decision_cap_reached(existing, max_decisions):
        logger.error(
            "[%s] supervisor hard_decision_cap decisions=%d/%d; forcing check_convergence",
            session_id, len(existing), max_decisions,
        )
        decision = SupervisorDecision(
            "check_convergence",
            "hard_decision_cap",
            "R_safety",
            metadata={"hard_cap": True, "cap_value": max_decisions, "llm_called": False},
        )
        supervisor = DeterministicSupervisor()  # for impl logging below
    else:
        supervisor = get_supervisor_for_session(full_session_id)
        decision = await supervisor.decide(state)

        # ── Safety: role-loop detection ───────────────────────────────────────
        max_consec = _settings.pipeline_supervisor_max_consecutive_same_role
        loop_reason = detect_role_loop(
            existing, decision.next_node, max_consec, _VALID_WORKER_ROLES,
        )
        if loop_reason:
            logger.warning(
                "[%s] supervisor role_loop role=%r reason=%s; overriding to check_convergence",
                session_id, decision.next_node, loop_reason,
            )
            decision = SupervisorDecision(
                "check_convergence",
                loop_reason,
                decision.rule + "_loop_override",
                metadata={
                    **(decision.metadata or {}),
                    "loop_override": True,
                    "original_next_node": decision.next_node,
                },
            )

    _decide_ms = (time.monotonic() - _t0) * 1000

    plan = state.get("plan") or {}
    steps: List[Dict[str, Any]] = plan.get("steps") or []

    done_count = sum(1 for s in steps if s.get("status") == "done")
    in_prog_count = sum(1 for s in steps if s.get("status") == "in_progress")
    pending_count = sum(1 for s in steps if s.get("status") == "pending")
    blocked_count = sum(1 for s in steps if s.get("status") == "blocked")

    logger.info(
        "[%s] supervisor strategy=%s impl=%s iter=%d step_id=%s plan_steps=%d "
        "done=%d in_progress=%d pending=%d blocked=%d pending_tool=%s "
        "next=%s rule=%s reason=%s decide_ms=%.1f metadata=%s",
        session_id,
        _settings.pipeline_supervisor_strategy,
        type(supervisor).__name__,
        state.get("current_iteration", 1),
        state.get("current_step_id"),
        len(steps),
        done_count, in_prog_count, pending_count, blocked_count,
        bool(state.get("pending_tool_calls")),
        decision.next_node, decision.rule, decision.reason,
        _decide_ms, decision.metadata,
    )

    entry: Dict[str, Any] = {
        "iteration": state.get("current_iteration", 1),
        "next_node": decision.next_node,
        "reason": decision.reason,
        "rule": decision.rule,
        "metadata": {
            **(decision.metadata or {}),
            "strategy": _settings.pipeline_supervisor_strategy,
            "impl": type(supervisor).__name__,
            "decide_ms": round(_decide_ms, 2),
        },
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    all_decisions = existing + [entry]

    # Emit routing telemetry after every decision; escalate to INFO every 10.
    from .supervisor_metrics import compute_routing_stats, log_routing_stats
    stats = compute_routing_stats(all_decisions)
    telemetry_level = "info" if len(all_decisions) % 10 == 0 else "debug"
    log_routing_stats(stats, session_id, at_level=telemetry_level)

    return {"supervisor_decisions": all_decisions}


def supervisor_decide(state: AgentState) -> str:
    """
    Conditional edge function.  Reads the decision already logged by supervisor_anchor
    rather than recomputing it, so the result is always consistent.
    """
    decisions: List[Dict[str, Any]] = state.get("supervisor_decisions") or []
    if decisions:
        return decisions[-1]["next_node"]
    # Fallback: should not be reached in normal operation.
    logger.error("supervisor_decide called with empty supervisor_decisions; defaulting to check_convergence")
    return "check_convergence"
