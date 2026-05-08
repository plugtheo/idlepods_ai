"""
Tests for LLMSupervisor and the shared _legal_targets helper.

Covers:
  - _legal_targets: all rule branches (R1, R1.5, R2a, R2b, R3, R4/R5)
  - LLMSupervisor.decide():
      * single-target shortcircuit (no LLM call)
      * successful LLM routing (tool call parsed)
      * invalid target returned by LLM → fallback
      * no tool calls in LLM response → fallback
      * LLM inference exception → fallback
      * budget exhausted → fallback
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestration.app.graph.state import AgentState
from orchestration.app.graph.supervisor import _legal_targets, _VALID_WORKER_ROLES
from orchestration.app.graph.supervisor_llm import (
    LLMSupervisor,
    _build_route_to_tool,
    _build_supervisor_prompt,
    _llm_calls_used,
    _parse_route_to_call,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_plan(steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"goal": "test", "steps": steps, "created_at": _now(), "updated_at": _now()}


def _step(step_id: str, status: str = "pending", owner_role: str = "coder") -> Dict[str, Any]:
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


def _make_state(**overrides) -> AgentState:
    base: AgentState = {
        "session_id": "test-session",
        "user_prompt": "Fix the authentication bug.",
        "current_iteration": 1,
        "max_iterations": 5,
        "best_score": 0.0,
        "agent_chain": ["planner", "coder", "reviewer"],
        "agent_chain_index": 0,
        "iteration_history": [],
        "supervisor_decisions": [],
    }
    base.update(overrides)  # type: ignore[arg-type]
    return base


def _tool_result_history() -> Dict[str, Any]:
    return {"role": "tool", "iteration": 1, "output": "tool output", "timestamp": _now()}


def _agent_history(role: str = "coder", iteration: int = 1) -> Dict[str, Any]:
    return {"role": role, "iteration": iteration, "output": f"{role} output", "timestamp": _now()}


def _supervisor_decision_llm(next_node: str = "coder") -> Dict[str, Any]:
    return {
        "iteration": 1,
        "next_node": next_node,
        "reason": "llm_routing",
        "rule": "R_llm",
        "metadata": {"llm_called": True, "strategy": "llm"},
        "ts": _now(),
    }


def _make_tool_call(next_node: str, reason: str = "test", confidence: float = 0.9) -> dict:
    import json
    return {
        "id": "call_001",
        "type": "function",
        "function": {
            "name": "route_to",
            "arguments": json.dumps(
                {"next_node": next_node, "reason": reason, "confidence": confidence}
            ),
        },
    }


# ─── _legal_targets ───────────────────────────────────────────────────────────

class TestLegalTargets:

    def test_r1_pending_tool_calls(self):
        state = _make_state(pending_tool_calls=[{"id": "c1", "function": {"name": "f"}}])
        assert _legal_targets(state) == frozenset({"tool_executor"})

    def test_r1_5_last_history_is_tool_with_valid_originator(self):
        state = _make_state(
            iteration_history=[_tool_result_history()],
            tool_originating_role="coder",
        )
        assert _legal_targets(state) == frozenset({"coder"})

    def test_r1_5_invalid_originator_falls_through(self):
        state = _make_state(
            iteration_history=[_tool_result_history()],
            tool_originating_role="unknown_role",
        )
        # Falls through to R4/R5 open set (no plan)
        result = _legal_targets(state)
        assert "check_convergence" in result
        assert result & _VALID_WORKER_ROLES  # non-empty intersection

    def test_r2a_step_in_progress(self):
        plan = _make_plan([_step("s1", "in_progress", "coder")])
        state = _make_state(plan=plan)
        assert _legal_targets(state) == frozenset({"coder"})

    def test_r2a_invalid_owner_role_falls_through(self):
        plan = _make_plan([_step("s1", "in_progress", "not_a_real_role")])
        state = _make_state(plan=plan)
        # Falls through — R2b may apply
        result = _legal_targets(state)
        assert result != frozenset({"not_a_real_role"})

    def test_r2b_pending_steps(self):
        plan = _make_plan([_step("s1", "pending"), _step("s2", "pending")])
        state = _make_state(plan=plan)
        assert _legal_targets(state) == frozenset({"planner"})

    def test_r3_all_terminal_done(self):
        plan = _make_plan([_step("s1", "done"), _step("s2", "done")])
        state = _make_state(plan=plan)
        assert _legal_targets(state) == frozenset({"review_critic", "check_convergence"})

    def test_r3_all_terminal_mixed_blocked(self):
        plan = _make_plan([_step("s1", "done"), _step("s2", "blocked")])
        state = _make_state(plan=plan)
        assert _legal_targets(state) == frozenset({"review_critic", "check_convergence"})

    def test_r4_no_plan_returns_open_set(self):
        state = _make_state()
        result = _legal_targets(state)
        assert "check_convergence" in result
        assert result.issuperset(_VALID_WORKER_ROLES)

    def test_r1_takes_priority_over_plan(self):
        plan = _make_plan([_step("s1", "in_progress", "coder")])
        state = _make_state(
            plan=plan,
            pending_tool_calls=[{"id": "c1"}],
        )
        assert _legal_targets(state) == frozenset({"tool_executor"})


# ─── helper utilities ─────────────────────────────────────────────────────────

class TestLlmCallsUsed:

    def test_counts_only_llm_called_true(self):
        state = _make_state(
            supervisor_decisions=[
                _supervisor_decision_llm("coder"),
                _supervisor_decision_llm("planner"),
                {"iteration": 1, "next_node": "coder", "reason": "r", "rule": "R1",
                 "metadata": {"llm_called": False}, "ts": _now()},
            ]
        )
        assert _llm_calls_used(state) == 2

    def test_zero_when_no_decisions(self):
        state = _make_state()
        assert _llm_calls_used(state) == 0

    def test_zero_when_all_deterministic(self):
        state = _make_state(
            supervisor_decisions=[
                {"metadata": {"llm_called": False}, "ts": _now()},
                {"metadata": {}, "ts": _now()},
            ]
        )
        assert _llm_calls_used(state) == 0


class TestParseRouteTo:

    def test_valid_tool_call(self):
        tc = _make_tool_call("coder", "coder should fix the bug", 0.95)
        targets = frozenset({"coder", "planner", "check_convergence"})
        decision = _parse_route_to_call(tc, targets, "test")
        assert decision is not None
        assert decision.next_node == "coder"
        assert decision.rule == "R_llm"
        assert decision.metadata["confidence"] == 0.95
        assert decision.metadata["llm_called"] is True

    def test_invalid_target_returns_none(self):
        tc = _make_tool_call("consensus")  # not in targets
        targets = frozenset({"coder", "planner"})
        decision = _parse_route_to_call(tc, targets, "test")
        assert decision is None

    def test_wrong_tool_name_returns_none(self):
        import json
        tc = {
            "id": "c1", "type": "function",
            "function": {"name": "other_tool", "arguments": "{}"},
        }
        decision = _parse_route_to_call(tc, frozenset({"coder"}), "test")
        assert decision is None

    def test_malformed_arguments_returns_none(self):
        tc = {"id": "c1", "type": "function",
              "function": {"name": "route_to", "arguments": "NOT_JSON"}}
        decision = _parse_route_to_call(tc, frozenset({"coder"}), "test")
        assert decision is None


class TestBuildRouteToTool:

    def test_enum_matches_targets(self):
        targets = frozenset({"coder", "planner", "check_convergence"})
        tool = _build_route_to_tool(targets)
        enum_vals = tool["function"]["parameters"]["properties"]["next_node"]["enum"]
        assert set(enum_vals) == targets

    def test_required_fields_present(self):
        tool = _build_route_to_tool(frozenset({"coder"}))
        required = tool["function"]["parameters"]["required"]
        assert set(required) == {"next_node", "reason", "confidence"}


class TestBuildSupervisorPrompt:

    def test_includes_user_prompt(self):
        state = _make_state(user_prompt="Fix the login bug in the auth module.")
        prompt = _build_supervisor_prompt(state, frozenset({"coder", "planner"}))
        assert "Fix the login bug" in prompt

    def test_includes_valid_agents(self):
        state = _make_state()
        targets = frozenset({"coder", "planner"})
        prompt = _build_supervisor_prompt(state, targets)
        assert "coder" in prompt
        assert "planner" in prompt

    def test_includes_plan_summary(self):
        plan = _make_plan([_step("s1", "done"), _step("s2", "pending")])
        state = _make_state(plan=plan)
        prompt = _build_supervisor_prompt(state, frozenset({"planner"}))
        assert "done" in prompt
        assert "pending" in prompt

    def test_includes_recent_history(self):
        state = _make_state(
            iteration_history=[_agent_history("coder", 1), _agent_history("reviewer", 1)]
        )
        prompt = _build_supervisor_prompt(state, frozenset({"coder", "reviewer"}))
        assert "coder" in prompt
        assert "reviewer" in prompt


# ─── LLMSupervisor.decide() ──────────────────────────────────────────────────

@pytest.mark.asyncio
class TestLLMSupervisorDecide:

    async def test_single_target_shortcircuit_no_llm_call(self):
        """R1: pending tool call → tool_executor without invoking LLM."""
        state = _make_state(pending_tool_calls=[{"id": "c1"}])
        supervisor = LLMSupervisor()
        with patch(
            "orchestration.app.graph.supervisor_llm.get_inference_client"
        ) as mock_client:
            decision = await supervisor.decide(state)
        mock_client.assert_not_called()
        assert decision.next_node == "tool_executor"
        assert decision.rule == "R_forced"
        assert decision.metadata["shortcircuit"] is True
        assert decision.metadata["llm_called"] is False

    async def test_successful_llm_routing(self):
        """Multi-target state → LLM picks a valid agent."""
        state = _make_state()  # no plan → open target set

        mock_response = MagicMock()
        mock_response.tool_calls = [_make_tool_call("coder", "coder should write the fix", 0.9)]
        mock_response.content = ""
        mock_response.tokens_generated = 10

        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value=mock_response)

        with patch(
            "orchestration.app.graph.supervisor_llm.get_inference_client",
            return_value=mock_client,
        ):
            decision = await LLMSupervisor().decide(state)

        assert decision.next_node == "coder"
        assert decision.rule == "R_llm"
        assert decision.metadata["llm_called"] is True
        assert decision.metadata["confidence"] == 0.9

    async def test_invalid_target_falls_back_to_deterministic(self):
        """LLM returns a node not in legal targets → fallback."""
        state = _make_state()

        mock_response = MagicMock()
        mock_response.tool_calls = [_make_tool_call("consensus")]  # never in open-set targets
        mock_response.content = ""
        mock_response.tokens_generated = 10

        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value=mock_response)

        with patch(
            "orchestration.app.graph.supervisor_llm.get_inference_client",
            return_value=mock_client,
        ):
            decision = await LLMSupervisor().decide(state)

        assert "_llm_fallback" in decision.rule
        assert decision.metadata["llm_called"] is False
        assert decision.metadata["fallback_reason"] == "no_valid_decision"

    async def test_no_tool_calls_falls_back_to_deterministic(self):
        """LLM response has no tool calls → fallback."""
        state = _make_state()

        mock_response = MagicMock()
        mock_response.tool_calls = []
        mock_response.content = "I think coder should run next."
        mock_response.tokens_generated = 8

        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value=mock_response)

        with patch(
            "orchestration.app.graph.supervisor_llm.get_inference_client",
            return_value=mock_client,
        ):
            decision = await LLMSupervisor().decide(state)

        assert "_llm_fallback" in decision.rule
        assert decision.metadata["llm_called"] is False

    async def test_inference_exception_falls_back_to_deterministic(self):
        """LLM call raises → fallback to deterministic."""
        state = _make_state()

        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(side_effect=RuntimeError("connection refused"))

        with patch(
            "orchestration.app.graph.supervisor_llm.get_inference_client",
            return_value=mock_client,
        ):
            decision = await LLMSupervisor().decide(state)

        assert "_exc_fallback" in decision.rule
        assert decision.metadata["llm_called"] is False
        assert "exc:" in decision.metadata["fallback_reason"]

    async def test_budget_exhausted_falls_back(self):
        """All budget slots consumed → skip LLM call, use deterministic."""
        # Fill budget with 8 llm_called=True decisions
        decisions = [_supervisor_decision_llm() for _ in range(8)]
        state = _make_state(supervisor_decisions=decisions)

        with patch(
            "orchestration.app.graph.supervisor_llm.get_inference_client"
        ) as mock_client:
            decision = await LLMSupervisor().decide(state)

        mock_client.assert_not_called()
        assert "_budget_fallback" in decision.rule
        assert decision.metadata["llm_called"] is False
        assert decision.metadata["fallback_reason"] == "budget_exhausted"

    async def test_partial_budget_still_calls_llm(self):
        """Budget not exhausted → LLM is called."""
        decisions = [_supervisor_decision_llm() for _ in range(7)]  # 7 of 8 used
        state = _make_state(supervisor_decisions=decisions)

        mock_response = MagicMock()
        mock_response.tool_calls = [_make_tool_call("coder")]
        mock_response.content = ""
        mock_response.tokens_generated = 5

        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value=mock_response)

        with patch(
            "orchestration.app.graph.supervisor_llm.get_inference_client",
            return_value=mock_client,
        ):
            decision = await LLMSupervisor().decide(state)

        assert decision.next_node == "coder"
        assert decision.rule == "R_llm"

    async def test_r3_plan_complete_llm_picks_between_evaluate_and_converge(self):
        """All plan steps done → LLM chooses between review_critic and check_convergence."""
        plan = _make_plan([_step("s1", "done"), _step("s2", "done")])
        state = _make_state(plan=plan)

        mock_response = MagicMock()
        mock_response.tool_calls = [_make_tool_call("review_critic")]
        mock_response.content = ""
        mock_response.tokens_generated = 5

        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value=mock_response)

        with patch(
            "orchestration.app.graph.supervisor_llm.get_inference_client",
            return_value=mock_client,
        ):
            decision = await LLMSupervisor().decide(state)

        assert decision.next_node == "review_critic"
        assert decision.rule == "R_llm"
