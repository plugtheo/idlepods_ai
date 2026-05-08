"""
Supervisor — plan-aware agent dispatch for the supervisor pipeline.

Replaces static agent_chain[index] advancement with a rule-based decision
that reads plan state, tool-call state, and iteration history to decide
which agent node runs next.

Two implementations:
  DeterministicSupervisor  — pure rule-based, no LLM call (default, v1).
  LLMSupervisor            — deferred to v2; adds inference call per transition.

Rule priority in decide():
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
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

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


class DeterministicSupervisor:
    """
    Rule-based supervisor.  Pure function — no I/O, no LLM calls.

    All routing decisions are deterministic given the same AgentState snapshot.
    """

    def decide(self, state: AgentState) -> SupervisorDecision:
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


# Module-level singleton used by pipeline.py and edges.py.
_SUPERVISOR = DeterministicSupervisor()


def supervisor_anchor(state: AgentState) -> dict:
    """
    Supervisor node — logs the dispatch decision and appends it to supervisor_decisions.

    This is a state-anchor node.  The routing decision itself is re-read by
    _supervisor_decide (the conditional edge function) from supervisor_decisions[-1]
    so we compute it exactly once per supervisor invocation.
    """
    decision = _SUPERVISOR.decide(state)
    session_id = (state.get("session_id") or "")[:8]
    plan = state.get("plan") or {}
    steps: List[Dict[str, Any]] = plan.get("steps") or []

    done_count = sum(1 for s in steps if s.get("status") == "done")
    in_prog_count = sum(1 for s in steps if s.get("status") == "in_progress")
    pending_count = sum(1 for s in steps if s.get("status") == "pending")
    blocked_count = sum(1 for s in steps if s.get("status") == "blocked")

    logger.info(
        "[%s] supervisor iter=%d step_id=%s plan_steps=%d done=%d in_progress=%d "
        "pending=%d blocked=%d pending_tool=%s next=%s rule=%s reason=%s metadata=%s",
        session_id,
        state.get("current_iteration", 1),
        state.get("current_step_id"),
        len(steps),
        done_count, in_prog_count, pending_count, blocked_count,
        bool(state.get("pending_tool_calls")),
        decision.next_node, decision.rule, decision.reason, decision.metadata,
    )

    existing: List[Dict[str, Any]] = list(state.get("supervisor_decisions") or [])

    if len(existing) > 200:
        logger.warning(
            "[%s] supervisor_decisions log size=%d — possible routing loop",
            session_id, len(existing),
        )

    entry: Dict[str, Any] = {
        "iteration": state.get("current_iteration", 1),
        "next_node": decision.next_node,
        "reason": decision.reason,
        "rule": decision.rule,
        "metadata": decision.metadata,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    return {"supervisor_decisions": existing + [entry]}


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
