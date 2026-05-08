"""
Tests for HybridSupervisor.decide().

Covers:
  - R1/R1.5/R2a/R2b single-target shortcircuit — no LLM call, rule=R_forced
  - R3 evaluate-vs-converge — deterministic guard, no LLM, rule tagged _hybrid_guard
      * evaluator not yet run → review_critic
      * evaluator already ran → check_convergence
  - R4/R5 open-set — LLM consulted
      * successful LLM pick → rule=R_hybrid_llm
      * invalid target from LLM → fallback (rule+_llm_fallback)
      * no tool calls from LLM → fallback
      * inference exception → fallback (rule+_exc_fallback)
      * budget exhausted → fallback (rule+_budget_fallback), no LLM call
      * partial budget still calls LLM
  - get_inference_client NOT called in all non-R4/R5 paths
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestration.app.graph.state import AgentState
from orchestration.app.graph.supervisor import _legal_targets, _VALID_WORKER_ROLES
from orchestration.app.graph.supervisor_hybrid import HybridSupervisor, _R3_TARGETS


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
        "session_id": "hyb-session",
        "user_prompt": "Implement the feature.",
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


def _evaluator_history(role: str = "review_critic", iteration: int = 1) -> Dict[str, Any]:
    return {"role": role, "iteration": iteration, "output": f"{role} output", "timestamp": _now()}


def _llm_decision_entry(next_node: str = "coder") -> Dict[str, Any]:
    return {
        "iteration": 1,
        "next_node": next_node,
        "reason": "llm_routing",
        "rule": "R_llm",
        "metadata": {"llm_called": True},
        "ts": _now(),
    }


def _make_tool_call(next_node: str, reason: str = "test reason", confidence: float = 0.85) -> dict:
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


def _mock_llm_response(next_node: str) -> MagicMock:
    response = MagicMock()
    response.tool_calls = [_make_tool_call(next_node)]
    response.content = ""
    response.tokens_generated = 8
    return response


# ─── R3 sentinel ─────────────────────────────────────────────────────────────

class TestR3TargetsSentinel:
    """Verify _R3_TARGETS matches what _legal_targets emits for all-done plans."""

    def test_r3_sentinel_matches_legal_targets_output(self):
        plan = _make_plan([_step("s1", "done"), _step("s2", "done")])
        state = _make_state(plan=plan)
        assert _legal_targets(state) == _R3_TARGETS


# ─── HybridSupervisor.decide() ───────────────────────────────────────────────

@pytest.mark.asyncio
class TestHybridSupervisorDecide:

    async def test_r1_single_target_shortcircuit_no_llm(self):
        """Pending tool call → tool_executor shortcircuit, no inference call."""
        state = _make_state(pending_tool_calls=[{"id": "c1"}])
        supervisor = HybridSupervisor()

        with patch(
            "orchestration.app.graph.supervisor_hybrid.get_inference_client"
        ) as mock_client:
            decision = await supervisor.decide(state)

        mock_client.assert_not_called()
        assert decision.next_node == "tool_executor"
        assert decision.rule == "R_forced"
        assert decision.metadata["shortcircuit"] is True
        assert decision.metadata["llm_called"] is False

    async def test_r2a_in_progress_step_shortcircuit_no_llm(self):
        """In-progress plan step → owner shortcircuit, no inference call."""
        plan = _make_plan([_step("s1", "in_progress", "coder")])
        state = _make_state(plan=plan)
        supervisor = HybridSupervisor()

        with patch(
            "orchestration.app.graph.supervisor_hybrid.get_inference_client"
        ) as mock_client:
            decision = await supervisor.decide(state)

        mock_client.assert_not_called()
        assert decision.next_node == "coder"
        assert decision.rule == "R_forced"
        assert decision.metadata["llm_called"] is False

    async def test_r2b_pending_steps_shortcircuit_no_llm(self):
        """Pending steps remain → planner shortcircuit, no inference call."""
        plan = _make_plan([_step("s1", "pending")])
        state = _make_state(plan=plan)
        supervisor = HybridSupervisor()

        with patch(
            "orchestration.app.graph.supervisor_hybrid.get_inference_client"
        ) as mock_client:
            decision = await supervisor.decide(state)

        mock_client.assert_not_called()
        assert decision.next_node == "planner"
        assert decision.rule == "R_forced"
        assert decision.metadata["llm_called"] is False

    async def test_r3_evaluator_not_run_picks_review_critic_no_llm(self):
        """All steps done, evaluator not yet run → review_critic via deterministic guard."""
        plan = _make_plan([_step("s1", "done"), _step("s2", "done")])
        state = _make_state(plan=plan)
        supervisor = HybridSupervisor()

        with patch(
            "orchestration.app.graph.supervisor_hybrid.get_inference_client"
        ) as mock_client:
            decision = await supervisor.decide(state)

        mock_client.assert_not_called()
        assert decision.next_node == "review_critic"
        assert "_hybrid_guard" in decision.rule
        assert decision.metadata["llm_called"] is False
        assert decision.metadata["guard_reason"] == "r3_terminal_guard"

    async def test_r3_evaluator_already_ran_picks_check_convergence_no_llm(self):
        """All steps done, evaluator ran this iteration → check_convergence via deterministic guard."""
        plan = _make_plan([_step("s1", "done")])
        state = _make_state(
            plan=plan,
            current_iteration=2,
            iteration_history=[_evaluator_history("review_critic", iteration=2)],
        )
        supervisor = HybridSupervisor()

        with patch(
            "orchestration.app.graph.supervisor_hybrid.get_inference_client"
        ) as mock_client:
            decision = await supervisor.decide(state)

        mock_client.assert_not_called()
        assert decision.next_node == "check_convergence"
        assert "_hybrid_guard" in decision.rule
        assert decision.metadata["llm_called"] is False
        assert decision.metadata["guard_reason"] == "r3_terminal_guard"

    async def test_r3_blocked_steps_still_uses_deterministic_guard(self):
        """All steps done/blocked → still R3, still resolved deterministically."""
        plan = _make_plan([_step("s1", "done"), _step("s2", "blocked")])
        state = _make_state(plan=plan)
        supervisor = HybridSupervisor()

        with patch(
            "orchestration.app.graph.supervisor_hybrid.get_inference_client"
        ) as mock_client:
            decision = await supervisor.decide(state)

        mock_client.assert_not_called()
        assert decision.next_node == "review_critic"
        assert "_hybrid_guard" in decision.rule

    async def test_r4_open_set_llm_called_and_picks_valid_agent(self):
        """No plan → open target set → LLM is consulted and picks coder."""
        state = _make_state()

        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value=_mock_llm_response("coder"))

        with patch(
            "orchestration.app.graph.supervisor_hybrid.get_inference_client",
            return_value=mock_client,
        ):
            decision = await HybridSupervisor().decide(state)

        assert decision.next_node == "coder"
        assert decision.rule == "R_hybrid_llm"
        assert decision.metadata["llm_called"] is True

    async def test_r4_successful_llm_metadata_has_confidence(self):
        """Successful LLM pick carries confidence in metadata."""
        state = _make_state()

        response = MagicMock()
        response.tool_calls = [_make_tool_call("researcher", confidence=0.92)]
        response.content = ""
        response.tokens_generated = 6

        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value=response)

        with patch(
            "orchestration.app.graph.supervisor_hybrid.get_inference_client",
            return_value=mock_client,
        ):
            decision = await HybridSupervisor().decide(state)

        assert decision.next_node == "researcher"
        assert decision.metadata["confidence"] == 0.92

    async def test_r4_invalid_target_falls_back_to_deterministic(self):
        """LLM picks a node not in legal targets → _llm_fallback."""
        state = _make_state()

        response = MagicMock()
        response.tool_calls = [_make_tool_call("consensus")]  # never in open-set
        response.content = ""
        response.tokens_generated = 5

        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value=response)

        with patch(
            "orchestration.app.graph.supervisor_hybrid.get_inference_client",
            return_value=mock_client,
        ):
            decision = await HybridSupervisor().decide(state)

        assert "_llm_fallback" in decision.rule
        assert decision.metadata["llm_called"] is False
        assert decision.metadata["guard_reason"] == "no_valid_decision"

    async def test_r4_no_tool_calls_falls_back_to_deterministic(self):
        """LLM response has no tool calls → _llm_fallback."""
        state = _make_state()

        response = MagicMock()
        response.tool_calls = []
        response.content = "I think coder should go next."
        response.tokens_generated = 7

        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value=response)

        with patch(
            "orchestration.app.graph.supervisor_hybrid.get_inference_client",
            return_value=mock_client,
        ):
            decision = await HybridSupervisor().decide(state)

        assert "_llm_fallback" in decision.rule
        assert decision.metadata["llm_called"] is False

    async def test_r4_inference_exception_falls_back_to_deterministic(self):
        """LLM inference raises → _exc_fallback with exc type in guard_reason."""
        state = _make_state()

        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(side_effect=RuntimeError("timeout"))

        with patch(
            "orchestration.app.graph.supervisor_hybrid.get_inference_client",
            return_value=mock_client,
        ):
            decision = await HybridSupervisor().decide(state)

        assert "_exc_fallback" in decision.rule
        assert decision.metadata["llm_called"] is False
        assert "exc:" in decision.metadata["guard_reason"]
        assert "RuntimeError" in decision.metadata["guard_reason"]

    async def test_budget_exhausted_no_llm_call(self):
        """All budget slots consumed → deterministic, no inference call."""
        decisions = [_llm_decision_entry() for _ in range(8)]
        state = _make_state(supervisor_decisions=decisions)

        with patch(
            "orchestration.app.graph.supervisor_hybrid.get_inference_client"
        ) as mock_client:
            decision = await HybridSupervisor().decide(state)

        mock_client.assert_not_called()
        assert "_budget_fallback" in decision.rule
        assert decision.metadata["llm_called"] is False
        assert decision.metadata["guard_reason"] == "budget_exhausted"

    async def test_partial_budget_still_calls_llm(self):
        """7 of 8 budget slots used → LLM is still called."""
        decisions = [_llm_decision_entry() for _ in range(7)]
        state = _make_state(supervisor_decisions=decisions)

        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value=_mock_llm_response("coder"))

        with patch(
            "orchestration.app.graph.supervisor_hybrid.get_inference_client",
            return_value=mock_client,
        ):
            decision = await HybridSupervisor().decide(state)

        mock_client.generate.assert_called_once()
        assert decision.next_node == "coder"
        assert decision.rule == "R_hybrid_llm"

    async def test_r3_does_not_consume_llm_budget(self):
        """R3 decisions are resolved deterministically and must not consume LLM budget."""
        plan = _make_plan([_step("s1", "done")])
        # Load budget to verify R3 is still deterministic even with budget remaining.
        decisions = [_llm_decision_entry() for _ in range(4)]
        state = _make_state(plan=plan, supervisor_decisions=decisions)
        supervisor = HybridSupervisor()

        with patch(
            "orchestration.app.graph.supervisor_hybrid.get_inference_client"
        ) as mock_client:
            decision = await supervisor.decide(state)

        mock_client.assert_not_called()
        # R3 guard — evaluator not run yet
        assert decision.next_node == "review_critic"
        assert "_hybrid_guard" in decision.rule
