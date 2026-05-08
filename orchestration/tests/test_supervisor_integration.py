"""
Phase 6 — End-to-end integration tests for the supervisor pipeline.

Exercises supervisor_anchor + supervisor_decide together, covering:
  - All three strategies (deterministic, llm, hybrid) via supervisor_anchor
  - Safety bounds applied by supervisor_anchor:
      * Hard decision cap forces check_convergence before calling supervisor
      * Role-loop detection overrides a worker-role decision
  - Rollout gating: sessions outside the window fall back to deterministic
  - Routing metrics emitted by supervisor_anchor (log_routing_stats called)
  - Token budget tracking stored in LLM decision metadata
  - supervisor_decide reads the correct next_node from supervisor_decisions[-1]

Patching strategy: settings is a singleton imported lazily inside functions,
so we use patch.object(real_settings, 'attribute') rather than module-path
patching, which would silently fail to intercept lazy imports.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestration.app.config.settings import settings as _real_settings
from orchestration.app.graph.state import AgentState
from orchestration.app.graph.supervisor import (
    _VALID_WORKER_ROLES,
    supervisor_anchor,
    supervisor_decide,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_plan(steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"goal": "test", "steps": steps, "created_at": _now(), "updated_at": _now()}


def _step(step_id: str, status: str = "pending", owner_role: str = "coder") -> Dict[str, Any]:
    return {
        "id": step_id, "description": f"Do {step_id}", "status": status,
        "owner_role": owner_role, "evidence": "", "files_touched": [],
        "tools_used": [], "depends_on": [],
    }


def _make_state(**overrides) -> AgentState:
    base: AgentState = {
        "session_id": "integ-session-001",
        "user_prompt": "Build the authentication module.",
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


def _forced_decision(next_node: str = "planner") -> Dict[str, Any]:
    return {
        "iteration": 1, "next_node": next_node, "reason": "forced",
        "rule": "R_forced",
        "metadata": {"llm_called": False, "shortcircuit": True},
        "ts": _now(),
    }


def _make_tool_call_payload(next_node: str) -> dict:
    return {
        "id": "call_001", "type": "function",
        "function": {
            "name": "route_to",
            "arguments": json.dumps({"next_node": next_node, "reason": "test", "confidence": 0.9}),
        },
    }


def _mock_llm_response(next_node: str, tokens: int = 42) -> MagicMock:
    r = MagicMock()
    r.tool_calls = [_make_tool_call_payload(next_node)]
    r.content = ""
    r.tokens_generated = tokens
    return r


# ─── anchor + decide consistency ─────────────────────────────────────────────

@pytest.mark.asyncio
class TestAnchorDecideConsistency:

    async def test_decide_reads_what_anchor_wrote(self):
        """supervisor_decide returns exactly the next_node written by supervisor_anchor."""
        state = _make_state(
            pending_tool_calls=[{"id": "c1", "function": {"name": "f", "arguments": "{}"}}],
        )
        delta = await supervisor_anchor(state)
        updated_state = {**state, **delta}
        assert supervisor_decide(updated_state) == "tool_executor"

    async def test_anchor_appends_required_fields(self):
        """Each decision entry must have iteration, next_node, rule, reason, ts, metadata."""
        state = _make_state(agent_chain=["coder"], agent_chain_index=0)
        delta = await supervisor_anchor(state)
        entry = delta["supervisor_decisions"][-1]
        for key in ("iteration", "next_node", "rule", "reason", "ts", "metadata"):
            assert key in entry, f"missing key: {key}"

    async def test_anchor_accumulates_decisions(self):
        """Existing decisions are preserved when anchor appends a new one."""
        existing = [_forced_decision("planner")]
        state = _make_state(supervisor_decisions=existing, agent_chain=["coder"], agent_chain_index=0)
        delta = await supervisor_anchor(state)
        assert len(delta["supervisor_decisions"]) == 2

    async def test_metadata_contains_impl_and_decide_ms(self):
        state = _make_state(agent_chain=["planner"], agent_chain_index=0)
        delta = await supervisor_anchor(state)
        meta = delta["supervisor_decisions"][-1]["metadata"]
        assert "impl" in meta
        assert "decide_ms" in meta


# ─── Strategy: deterministic ─────────────────────────────────────────────────

@pytest.mark.asyncio
class TestDeterministicStrategyIntegration:

    async def test_r1_routes_to_tool_executor(self):
        state = _make_state(pending_tool_calls=[{"id": "c1"}])
        delta = await supervisor_anchor(state)
        assert delta["supervisor_decisions"][-1]["next_node"] == "tool_executor"

    async def test_r2a_routes_to_step_owner(self):
        plan = _make_plan([_step("s1", "in_progress", "debugger")])
        state = _make_state(plan=plan)
        delta = await supervisor_anchor(state)
        assert delta["supervisor_decisions"][-1]["next_node"] == "debugger"

    async def test_r2b_routes_to_planner_for_pending_steps(self):
        plan = _make_plan([_step("s1", "pending")])
        state = _make_state(plan=plan)
        delta = await supervisor_anchor(state)
        assert delta["supervisor_decisions"][-1]["next_node"] == "planner"


# ─── Strategy: LLM ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestLLMStrategyIntegration:

    async def test_llm_decision_appended_with_r_llm_rule(self):
        """No plan → open set → LLM is called and decision has rule=R_llm."""
        state = _make_state()

        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value=_mock_llm_response("coder"))

        with patch.object(_real_settings, "pipeline_supervisor_strategy", "llm"), \
             patch("orchestration.app.graph.supervisor_llm.get_inference_client",
                   return_value=mock_client):
            delta = await supervisor_anchor(state)

        entry = delta["supervisor_decisions"][-1]
        assert entry["next_node"] == "coder"
        assert entry["rule"] == "R_llm"

    async def test_tokens_generated_stored_in_metadata(self):
        """tokens_generated from the LLM response must be recorded in decision metadata."""
        state = _make_state()

        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value=_mock_llm_response("coder", tokens=77))

        with patch.object(_real_settings, "pipeline_supervisor_strategy", "llm"), \
             patch("orchestration.app.graph.supervisor_llm.get_inference_client",
                   return_value=mock_client):
            delta = await supervisor_anchor(state)

        meta = delta["supervisor_decisions"][-1]["metadata"]
        assert meta.get("tokens_generated") == 77


# ─── Strategy: hybrid ────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestHybridStrategyIntegration:

    async def test_r3_resolved_deterministically_no_llm(self):
        """All plan steps done → HybridSupervisor uses deterministic guard, no LLM call."""
        plan = _make_plan([_step("s1", "done"), _step("s2", "done")])
        state = _make_state(plan=plan)

        with patch.object(_real_settings, "pipeline_supervisor_strategy", "hybrid"), \
             patch("orchestration.app.graph.supervisor_hybrid.get_inference_client") as mock_inf:
            delta = await supervisor_anchor(state)

        mock_inf.assert_not_called()
        entry = delta["supervisor_decisions"][-1]
        assert entry["next_node"] == "review_critic"
        assert "_hybrid_guard" in entry["rule"]

    async def test_r4_llm_called_and_decision_tagged_r_hybrid_llm(self):
        """No plan → HybridSupervisor calls LLM → rule=R_hybrid_llm."""
        state = _make_state()

        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value=_mock_llm_response("researcher"))

        with patch.object(_real_settings, "pipeline_supervisor_strategy", "hybrid"), \
             patch("orchestration.app.graph.supervisor_hybrid.get_inference_client",
                   return_value=mock_client):
            delta = await supervisor_anchor(state)

        entry = delta["supervisor_decisions"][-1]
        assert entry["next_node"] == "researcher"
        assert entry["rule"] == "R_hybrid_llm"


# ─── Safety: hard decision cap ────────────────────────────────────────────────

@pytest.mark.asyncio
class TestHardDecisionCap:

    async def test_cap_reached_forces_check_convergence_no_llm_call(self):
        """When existing decisions >= max_decisions, anchor forces convergence immediately."""
        cap = _real_settings.pipeline_supervisor_max_decisions_per_run
        existing = [_forced_decision() for _ in range(cap)]
        state = _make_state(supervisor_decisions=existing)

        with patch("orchestration.app.graph.supervisor_llm.get_inference_client") as mock_inf, \
             patch.object(_real_settings, "pipeline_supervisor_strategy", "llm"):
            delta = await supervisor_anchor(state)

        mock_inf.assert_not_called()
        entry = delta["supervisor_decisions"][-1]
        assert entry["next_node"] == "check_convergence"
        assert entry["rule"] == "R_safety"
        assert entry["metadata"]["hard_cap"] is True

    async def test_cap_exceeded_by_one_still_forces_convergence(self):
        """Cap + 1 existing decisions also forces convergence (>= check)."""
        cap = _real_settings.pipeline_supervisor_max_decisions_per_run
        existing = [_forced_decision() for _ in range(cap + 1)]
        state = _make_state(supervisor_decisions=existing)

        delta = await supervisor_anchor(state)
        assert delta["supervisor_decisions"][-1]["next_node"] == "check_convergence"
        assert delta["supervisor_decisions"][-1]["rule"] == "R_safety"

    async def test_one_below_cap_calls_supervisor_normally(self):
        """cap - 1 existing decisions should call the supervisor normally."""
        cap = _real_settings.pipeline_supervisor_max_decisions_per_run
        existing = [_forced_decision() for _ in range(cap - 1)]
        state = _make_state(supervisor_decisions=existing, agent_chain=["coder"], agent_chain_index=0)

        delta = await supervisor_anchor(state)
        # Normal routing — should NOT produce R_safety
        entry = delta["supervisor_decisions"][-1]
        assert entry["rule"] != "R_safety"


# ─── Safety: role-loop detection ─────────────────────────────────────────────

@pytest.mark.asyncio
class TestRoleLoopDetection:

    async def test_role_loop_overrides_to_check_convergence(self):
        """4 prior coder decisions + proposed coder (from chain) = 5 → loop override."""
        max_consec = _real_settings.pipeline_supervisor_max_consecutive_same_role
        # Fill max_consec - 1 prior decisions so proposed makes it exactly max_consec.
        existing = [
            {"next_node": "coder", "rule": "R_llm",
             "metadata": {"llm_called": False}, "ts": _now()}
            for _ in range(max_consec - 1)
        ]
        # State that would naturally route to coder (chain dispatch)
        state = _make_state(supervisor_decisions=existing, agent_chain=["coder"], agent_chain_index=0)

        delta = await supervisor_anchor(state)
        entry = delta["supervisor_decisions"][-1]
        assert entry["next_node"] == "check_convergence"
        assert "_loop_override" in entry["rule"]
        assert entry["metadata"]["loop_override"] is True
        assert entry["metadata"]["original_next_node"] == "coder"

    async def test_no_loop_one_below_threshold(self):
        """max_consec - 2 prior decisions + proposed = max_consec - 1 → no override."""
        max_consec = _real_settings.pipeline_supervisor_max_consecutive_same_role
        existing = [
            {"next_node": "coder", "rule": "R_llm",
             "metadata": {"llm_called": False}, "ts": _now()}
            for _ in range(max_consec - 2)
        ]
        state = _make_state(supervisor_decisions=existing, agent_chain=["coder"], agent_chain_index=0)

        delta = await supervisor_anchor(state)
        entry = delta["supervisor_decisions"][-1]
        assert entry["metadata"].get("loop_override") is not True

    async def test_loop_detection_disabled_at_zero(self):
        """max_consecutive_same_role=0 disables loop detection even with many repeats."""
        existing = [
            {"next_node": "coder", "rule": "R4",
             "metadata": {"llm_called": False}, "ts": _now()}
            for _ in range(20)
        ]
        state = _make_state(supervisor_decisions=existing, agent_chain=["coder"], agent_chain_index=0)

        with patch.object(_real_settings, "pipeline_supervisor_max_consecutive_same_role", 0):
            delta = await supervisor_anchor(state)

        entry = delta["supervisor_decisions"][-1]
        assert entry["metadata"].get("loop_override") is not True


# ─── Rollout gating ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestRolloutGating:

    async def test_0pct_rollout_always_uses_deterministic(self):
        """rollout_pct=0 means all sessions get DeterministicSupervisor regardless of strategy."""
        state = _make_state()

        with patch.object(_real_settings, "pipeline_supervisor_strategy", "llm"), \
             patch.object(_real_settings, "pipeline_supervisor_rollout_pct", 0), \
             patch("orchestration.app.graph.supervisor_llm.get_inference_client") as mock_inf:
            delta = await supervisor_anchor(state)

        mock_inf.assert_not_called()
        meta = delta["supervisor_decisions"][-1]["metadata"]
        assert meta.get("impl") == "DeterministicSupervisor"

    async def test_100pct_rollout_default_uses_configured_strategy(self):
        """Default rollout_pct=100 uses the configured strategy for all sessions."""
        state = _make_state()

        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value=_mock_llm_response("researcher"))

        with patch.object(_real_settings, "pipeline_supervisor_strategy", "llm"), \
             patch("orchestration.app.graph.supervisor_llm.get_inference_client",
                   return_value=mock_client):
            delta = await supervisor_anchor(state)

        mock_client.generate.assert_called_once()
        assert delta["supervisor_decisions"][-1]["rule"] == "R_llm"


# ─── Metrics emitted ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestMetricsEmitted:

    async def test_log_routing_stats_called_after_anchor(self):
        """supervisor_anchor must call log_routing_stats on every decision."""
        state = _make_state(agent_chain=["planner"], agent_chain_index=0)

        with patch("orchestration.app.graph.supervisor_metrics.log_routing_stats") as mock_log:
            await supervisor_anchor(state)

        mock_log.assert_called_once()

    async def test_metrics_escalate_to_info_at_decision_10(self):
        """log_routing_stats must be called at_level='info' when total decisions reach 10."""
        # 9 existing → anchor adds 1 → total = 10 → INFO
        existing = [_forced_decision() for _ in range(9)]
        state = _make_state(
            supervisor_decisions=existing,
            agent_chain=["coder"], agent_chain_index=0,
        )

        with patch.object(_real_settings, "pipeline_supervisor_max_consecutive_same_role", 0), \
             patch("orchestration.app.graph.supervisor_metrics.log_routing_stats") as mock_log:
            await supervisor_anchor(state)

        mock_log.assert_called_once()
        _args, kwargs = mock_log.call_args
        assert kwargs.get("at_level") == "info"

    async def test_metrics_at_debug_below_milestone(self):
        """log_routing_stats uses at_level='debug' when total decisions is not a multiple of 10."""
        existing = [_forced_decision() for _ in range(4)]  # 4 → total = 5 after anchor
        state = _make_state(
            supervisor_decisions=existing,
            agent_chain=["coder"], agent_chain_index=0,
        )

        with patch.object(_real_settings, "pipeline_supervisor_max_consecutive_same_role", 0), \
             patch("orchestration.app.graph.supervisor_metrics.log_routing_stats") as mock_log:
            await supervisor_anchor(state)

        mock_log.assert_called_once()
        _args, kwargs = mock_log.call_args
        assert kwargs.get("at_level") == "debug"
