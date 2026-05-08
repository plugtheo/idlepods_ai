"""
Tests for DeterministicSupervisor and the supervisor pipeline anchors.

Covers all decision rules (R1–R5) plus the regression cases called out in the
revised plan: in_progress dispatch (A1), post-tool-executor return (A2), and
the _maybe_update_plan_step gate interaction (B1).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestration.app.graph.state import AgentState
from orchestration.app.graph.supervisor import (
    DeterministicSupervisor,
    supervisor_anchor,
    supervisor_decide,
)


# ─── helpers ─────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_plan(steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "goal": "test goal",
        "steps": steps,
        "created_at": _now(),
        "updated_at": _now(),
    }


def _step(
    step_id: str,
    status: str = "pending",
    owner_role: str = "coder",
) -> Dict[str, Any]:
    return {
        "id": step_id,
        "description": f"Do {step_id}",
        "status": status,
        "owner_role": owner_role,
        "evidence": "",
        "files_touched": [],
        "tools_used": [],
        "depends_on": [],
    }


def _history_entry(role: str, iteration: int = 1) -> Dict[str, Any]:
    return {
        "role": role,
        "iteration": iteration,
        "output": f"output from {role}",
        "full_output": f"output from {role}",
        "timestamp": _now(),
    }


def _make_state(**kwargs) -> AgentState:
    base: AgentState = {
        "session_id": "test-session",
        "task_id": "test-task",
        "user_prompt": "Test prompt",
        "agent_chain": ["planner", "coder", "review_critic"],
        "agent_chain_index": 0,
        "allowed_files": [],
        "file_fingerprints": {},
        "few_shots": [],
        "repo_snippets": [],
        "system_hints": "",
        "current_iteration": 1,
        "max_iterations": 3,
        "convergence_threshold": 0.85,
        "conversation_history": [],
        "iteration_history": [],
        "last_output": "",
        "iteration_scores": [],
        "best_score": 0.0,
        "best_output": "",
        "converged": False,
        "quality_converged": False,
        "final_output": "",
        "final_score": 0.0,
        "plan": None,
        "plan_changed": False,
        "current_step_id": None,
        "pending_tool_calls": [],
        "tool_steps_used": 0,
        "tool_originating_role": "",
        "supervisor_decisions": [],
    }
    base.update(kwargs)
    return base


_SUP = DeterministicSupervisor()


# ─── R1: pending_tool_calls → tool_executor ──────────────────────────────────

def test_r1_pending_tool_calls_routes_to_tool_executor():
    state = _make_state(pending_tool_calls=[{"id": "c1", "function": {"name": "read_file", "arguments": "{}"}}])
    decision = _SUP.decide(state)
    assert decision.next_node == "tool_executor"
    assert decision.rule == "R1"


def test_r1_empty_pending_tool_calls_falls_through():
    """Empty list must NOT trigger R1."""
    state = _make_state(
        pending_tool_calls=[],
        agent_chain=["coder", "review_critic"],
        agent_chain_index=0,
    )
    decision = _SUP.decide(state)
    assert decision.next_node != "tool_executor"


# ─── R1.5: post-tool-executor resume ─────────────────────────────────────────

def test_r15_after_tool_executor_routes_to_originator():
    """When last history entry is role='tool', supervisor resumes the originating role."""
    history = [_history_entry("coder"), _history_entry("tool")]
    state = _make_state(
        iteration_history=history,
        tool_originating_role="coder",
        pending_tool_calls=[],  # already cleared by tool_executor
    )
    decision = _SUP.decide(state)
    assert decision.next_node == "coder"
    assert decision.rule == "R1.5"
    assert decision.metadata["originator"] == "coder"


def test_r15_invalid_originator_falls_through():
    """If tool_originating_role is not a valid worker role, R1.5 falls through."""
    history = [_history_entry("tool")]
    state = _make_state(
        iteration_history=history,
        tool_originating_role="consensus",  # not a valid worker role
        pending_tool_calls=[],
        agent_chain=["coder"],
        agent_chain_index=0,
    )
    decision = _SUP.decide(state)
    # Should fall through to R4 (chain template)
    assert decision.rule == "R4"
    assert decision.next_node == "coder"


def test_r15_no_history_does_not_fire():
    """Empty history means R1.5 cannot detect tool_executor return."""
    state = _make_state(
        iteration_history=[],
        tool_originating_role="coder",
        pending_tool_calls=[],
        agent_chain=["coder"],
        agent_chain_index=0,
    )
    decision = _SUP.decide(state)
    assert decision.rule != "R1.5"


# ─── R2a: in_progress step dispatch ──────────────────────────────────────────

def test_r2a_dispatches_to_in_progress_step_owner():
    """Supervisor must route to the owner_role of an in_progress step (A1 regression)."""
    plan = _make_plan([
        _step("step-1", status="done", owner_role="coder"),
        _step("step-2", status="in_progress", owner_role="debugger"),
    ])
    state = _make_state(plan=plan)
    decision = _SUP.decide(state)
    assert decision.next_node == "debugger"
    assert decision.rule == "R2a"
    assert decision.metadata["step_id"] == "step-2"


def test_r2a_invalid_owner_role_falls_through_to_r2b():
    """When in_progress step has an invalid owner_role, R2a falls through to R2b
    (planner re-advances the next pending step)."""
    plan = _make_plan([
        _step("step-1", status="in_progress", owner_role="consensus"),  # invalid
        _step("step-2", status="pending", owner_role="coder"),
    ])
    state = _make_state(plan=plan)
    decision = _SUP.decide(state)
    # R2a falls through because owner_role invalid; R2b sees pending step-2
    assert decision.rule == "R2b"
    assert decision.next_node == "planner"


# ─── R2b: pending steps → planner advance ────────────────────────────────────

def test_r2b_routes_to_planner_when_pending_steps_remain():
    """When no in_progress step but pending steps exist, route to planner."""
    plan = _make_plan([
        _step("step-1", status="done", owner_role="researcher"),
        _step("step-2", status="pending", owner_role="coder"),
    ])
    state = _make_state(plan=plan)
    decision = _SUP.decide(state)
    assert decision.next_node == "planner"
    assert decision.rule == "R2b"


# ─── R3: all steps terminal → evaluate then converge ─────────────────────────

def test_r3_routes_to_review_critic_when_evaluator_not_run():
    """All steps terminal and evaluator hasn't run this iter → review_critic."""
    plan = _make_plan([
        _step("step-1", status="done"),
        _step("step-2", status="done"),
    ])
    state = _make_state(plan=plan, iteration_history=[])
    decision = _SUP.decide(state)
    assert decision.next_node == "review_critic"
    assert decision.rule == "R3"


def test_r3_routes_to_check_convergence_when_evaluator_ran():
    """All steps terminal and evaluator already ran this iter → check_convergence."""
    plan = _make_plan([_step("step-1", status="done")])
    history = [_history_entry("reviewer", iteration=1), _history_entry("critic", iteration=1)]
    state = _make_state(plan=plan, iteration_history=history, current_iteration=1)
    decision = _SUP.decide(state)
    assert decision.next_node == "check_convergence"
    assert decision.rule == "R3"


def test_r3_includes_blocked_steps_in_metadata():
    """Blocked steps are reported in metadata so logs are actionable."""
    plan = _make_plan([
        _step("step-1", status="done"),
        _step("step-2", status="blocked"),
    ])
    state = _make_state(plan=plan, iteration_history=[])
    decision = _SUP.decide(state)
    assert decision.next_node == "review_critic"
    assert "step-2" in (decision.metadata or {}).get("blocked_steps", [])


def test_r3_evaluator_ran_in_different_iter_does_not_count():
    """Evaluator from a prior iteration doesn't satisfy the current iteration check."""
    plan = _make_plan([_step("step-1", status="done")])
    # Reviewer ran in iteration 1, but current_iteration is 2
    history = [_history_entry("reviewer", iteration=1)]
    state = _make_state(plan=plan, iteration_history=history, current_iteration=2)
    decision = _SUP.decide(state)
    assert decision.next_node == "review_critic"
    assert decision.rule == "R3"


# ─── R4: static chain template fallback ──────────────────────────────────────

def test_r4_dispatches_chain_index_role_when_no_plan():
    state = _make_state(agent_chain=["researcher", "coder"], agent_chain_index=0)
    decision = _SUP.decide(state)
    assert decision.next_node == "researcher"
    assert decision.rule == "R4"
    assert decision.metadata["chain_index"] == 0


def test_r4_advances_with_chain_index():
    state = _make_state(agent_chain=["researcher", "coder", "review_critic"], agent_chain_index=2)
    decision = _SUP.decide(state)
    assert decision.next_node == "review_critic"
    assert decision.rule == "R4"


def test_r4_invalid_chain_role_falls_through_to_r5():
    """consensus in the chain is not a valid worker — should fall through to R5."""
    state = _make_state(agent_chain=["consensus"], agent_chain_index=0, iteration_history=[])
    decision = _SUP.decide(state)
    assert decision.rule == "R5"
    assert decision.next_node == "review_critic"


# ─── R5: chain exhausted → evaluate then converge ────────────────────────────

def test_r5_routes_to_review_critic_when_evaluator_not_run():
    state = _make_state(agent_chain=["coder"], agent_chain_index=1, iteration_history=[])
    decision = _SUP.decide(state)
    assert decision.next_node == "review_critic"
    assert decision.rule == "R5"


def test_r5_routes_to_check_convergence_when_evaluator_ran():
    history = [_history_entry("review_critic", iteration=1)]
    state = _make_state(
        agent_chain=["coder"], agent_chain_index=1,
        iteration_history=history, current_iteration=1,
    )
    decision = _SUP.decide(state)
    assert decision.next_node == "check_convergence"
    assert decision.rule == "R5"


def test_r5_empty_chain_falls_through():
    state = _make_state(agent_chain=[], agent_chain_index=0, iteration_history=[])
    decision = _SUP.decide(state)
    assert decision.rule == "R5"


# ─── R1 preempts plan rules ──────────────────────────────────────────────────

def test_r1_preempts_plan_rules():
    """Pending tool call must always win, even when a plan with in_progress step exists."""
    plan = _make_plan([_step("step-1", status="in_progress", owner_role="coder")])
    state = _make_state(
        plan=plan,
        pending_tool_calls=[{"id": "c1", "function": {"name": "read_file", "arguments": "{}"}}],
    )
    decision = _SUP.decide(state)
    assert decision.next_node == "tool_executor"
    assert decision.rule == "R1"


# ─── R1.5 preempts plan rules ────────────────────────────────────────────────

def test_r15_preempts_plan_rules():
    """Post-tool-executor return wins over in_progress plan rule."""
    plan = _make_plan([_step("step-1", status="in_progress", owner_role="debugger")])
    history = [_history_entry("tool")]
    state = _make_state(
        plan=plan,
        iteration_history=history,
        tool_originating_role="coder",
        pending_tool_calls=[],
    )
    decision = _SUP.decide(state)
    assert decision.next_node == "coder"
    assert decision.rule == "R1.5"


# ─── supervisor_anchor and supervisor_decide ─────────────────────────────────

def test_supervisor_anchor_appends_decision_to_state():
    state = _make_state(agent_chain=["researcher"], agent_chain_index=0)
    delta = supervisor_anchor(state)
    assert "supervisor_decisions" in delta
    decisions = delta["supervisor_decisions"]
    assert len(decisions) == 1
    assert decisions[0]["next_node"] == "researcher"
    assert decisions[0]["rule"] == "R4"
    assert "ts" in decisions[0]


def test_supervisor_anchor_accumulates_decisions():
    """Decisions from prior anchor calls are preserved."""
    existing = [{"iteration": 1, "next_node": "coder", "reason": "x", "rule": "R4", "metadata": None, "ts": _now()}]
    state = _make_state(
        supervisor_decisions=existing,
        agent_chain=["reviewer"],
        agent_chain_index=0,
    )
    delta = supervisor_anchor(state)
    assert len(delta["supervisor_decisions"]) == 2


def test_supervisor_decide_reads_last_decision():
    """Edge function reads supervisor_decisions[-1] without recomputing."""
    decisions = [
        {"next_node": "planner", "reason": "x", "rule": "R2b", "metadata": None, "ts": _now()},
        {"next_node": "coder", "reason": "y", "rule": "R2a", "metadata": None, "ts": _now()},
    ]
    state = _make_state(supervisor_decisions=decisions)
    assert supervisor_decide(state) == "coder"


def test_supervisor_decide_defaults_to_check_convergence_when_empty():
    state = _make_state(supervisor_decisions=[])
    assert supervisor_decide(state) == "check_convergence"


# ─── Full ReAct round-trip under supervisor ───────────────────────────────────

def test_react_loop_under_plan():
    """
    Simulate one ReAct round-trip:
    coder emits tool call → R1 → tool_executor
    tool_executor clears pending_tool_calls, adds tool history → R1.5 → coder
    coder finishes → R2a with in_progress step → coder again (step not done yet)

    After _maybe_update_plan_step marks step done (via separate node call), R3 kicks in.
    """
    plan = _make_plan([_step("step-1", status="in_progress", owner_role="coder")])

    # State after coder emitted a tool call
    state_with_tool = _make_state(
        plan=plan,
        current_step_id="step-1",
        pending_tool_calls=[{"id": "c1", "function": {"name": "read_file", "arguments": "{}"}}],
    )
    d1 = _SUP.decide(state_with_tool)
    assert d1.next_node == "tool_executor"  # R1

    # After tool_executor clears pending_tool_calls and appends tool history
    state_after_tool = _make_state(
        plan=plan,
        current_step_id="step-1",
        pending_tool_calls=[],
        iteration_history=[_history_entry("tool")],
        tool_originating_role="coder",
    )
    d2 = _SUP.decide(state_after_tool)
    assert d2.next_node == "coder"  # R1.5 — coder consumes tool result
    assert d2.rule == "R1.5"

    # After coder finishes (step still in_progress until _maybe_update_plan_step marks done)
    state_coder_done = _make_state(
        plan=plan,
        current_step_id="step-1",
        pending_tool_calls=[],
        iteration_history=[_history_entry("coder")],
    )
    d3 = _SUP.decide(state_coder_done)
    assert d3.next_node == "coder"  # R2a — step is still in_progress
    assert d3.rule == "R2a"


# ─── Multi-iteration plan-done regression ─────────────────────────────────────

def test_plan_all_done_and_evaluator_ran_routes_to_convergence():
    """After all steps done and evaluator ran, must not loop back to workers."""
    plan = _make_plan([
        _step("step-1", status="done"),
        _step("step-2", status="done"),
    ])
    history = [
        _history_entry("reviewer", iteration=1),
        _history_entry("critic", iteration=1),
    ]
    state = _make_state(plan=plan, iteration_history=history, current_iteration=1)
    decision = _SUP.decide(state)
    assert decision.next_node == "check_convergence"
    assert decision.rule == "R3"


# ─── _evaluator_ran_this_iter helper ──────────────────────────────────────────

def test_evaluator_ran_this_iter_positive():
    from orchestration.app.graph.supervisor import _EVALUATOR_ROLES
    sup = DeterministicSupervisor()
    history = [_history_entry("review_critic", iteration=2)]
    assert sup._evaluator_ran_this_iter(history, 2) is True


def test_evaluator_ran_this_iter_negative_wrong_iter():
    sup = DeterministicSupervisor()
    history = [_history_entry("reviewer", iteration=1)]
    assert sup._evaluator_ran_this_iter(history, 2) is False


def test_evaluator_ran_this_iter_negative_wrong_role():
    sup = DeterministicSupervisor()
    history = [_history_entry("coder", iteration=1)]
    assert sup._evaluator_ran_this_iter(history, 1) is False
