"""
Pipeline integration tests — supervisor and legacy full-graph runs.

Both pipelines are driven with a mocked inference client (no real gRPC/HTTP).
Configuration choices that keep tests fast and deterministic:
  - convergence_threshold=0.0  → any score satisfies convergence
  - max_iterations=1           → belt-and-suspenders against runaway loops
  - chain=["coder","review_critic"] → 2-agent short chain (skip_consensus path)

Supervisor routing trace for the base chain (no plan, R4/R5 path):
  START→supervisor[R4→coder]→coder→supervisor[R4→review_critic]
  →review_critic(runs reviewer+critic)→supervisor[R5→check_convergence]
  →check_convergence→finalize(skip_consensus=True)→END
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from shared.contracts.inference import GenerateResponse


# ── helpers ───────────────────────────────────────────────────────────────────

_LONG_OUTPUT = "x" * 60   # passes short_text validator (>= 30 chars)
_CODER_OUTPUT = "def add(a, b):\n    return a + b  # clean and correct implementation"


def _make_response(content: str = _LONG_OUTPUT) -> GenerateResponse:
    return GenerateResponse(
        content=content,
        backend="primary",
        role="coder",
        tokens_generated=20,
    )


def _mock_client(content: str = _LONG_OUTPUT) -> MagicMock:
    client = MagicMock()
    client.generate = AsyncMock(return_value=_make_response(content))
    return client


def _base_state(**overrides) -> dict:
    state: dict = {
        "session_id": "integ-test-session",
        "task_id": "integ-test-task",
        "user_prompt": "write a function that adds two numbers",
        "agent_chain": ["coder", "review_critic"],
        "agent_chain_index": 0,
        "few_shots": [],
        "repo_snippets": [],
        "system_hints": "",
        "current_iteration": 1,
        "max_iterations": 1,
        "convergence_threshold": 0.0,
        "iteration_history": [],
        "conversation_history": [],
        "last_output": "",
        "iteration_scores": [],
        "best_score": 0.0,
        "best_output": "",
        "converged": False,
        "quality_converged": False,
        "final_output": "",
        "final_score": 0.0,
        "pending_tool_calls": [],
        "tool_steps_used": 0,
        "tool_originating_role": "",
        "plan": None,
        "plan_changed": False,
        "current_step_id": None,
        "supervisor_decisions": [],
    }
    state.update(overrides)
    return state


# ── Item 1: supervisor pipeline integration ───────────────────────────────────


@pytest.mark.asyncio
class TestSupervisorPipelineIntegration:
    def _build(self):
        from orchestration.app.graph.pipeline import _build_supervisor_pipeline
        return _build_supervisor_pipeline()

    async def test_compiles_without_error(self):
        graph = self._build()
        assert graph is not None
        assert hasattr(graph, "ainvoke")

    async def test_runs_to_completion(self):
        """Graph must set final_output for a simple 2-agent chain."""
        graph = self._build()
        with patch(
            "orchestration.app.graph.nodes.get_inference_client",
            return_value=_mock_client(_CODER_OUTPUT),
        ):
            result = await graph.ainvoke(_base_state(), config={"recursion_limit": 200})

        assert result.get("final_output"), "final_output must be non-empty after graph completes"

    async def test_supervisor_decisions_logged(self):
        """supervisor_decisions must accumulate at least one entry per run."""
        graph = self._build()
        with patch(
            "orchestration.app.graph.nodes.get_inference_client",
            return_value=_mock_client(),
        ):
            result = await graph.ainvoke(_base_state(), config={"recursion_limit": 200})

        decisions = result.get("supervisor_decisions", [])
        assert len(decisions) > 0

    async def test_first_decision_dispatches_coder_via_R4(self):
        """With chain=['coder','review_critic'] and no plan, first dispatch must be coder (R4)."""
        graph = self._build()
        with patch(
            "orchestration.app.graph.nodes.get_inference_client",
            return_value=_mock_client(),
        ):
            result = await graph.ainvoke(_base_state(), config={"recursion_limit": 200})

        first = result["supervisor_decisions"][0]
        assert first["next_node"] == "coder"
        assert first["rule"] == "R4"

    async def test_evaluator_visited_before_convergence(self):
        """Supervisor must route to review_critic (R5) before check_convergence."""
        graph = self._build()
        with patch(
            "orchestration.app.graph.nodes.get_inference_client",
            return_value=_mock_client(),
        ):
            result = await graph.ainvoke(_base_state(), config={"recursion_limit": 200})

        visited = [d["next_node"] for d in result["supervisor_decisions"]]
        assert "review_critic" in visited
        assert "check_convergence" in visited
        # review_critic must appear before check_convergence
        assert visited.index("review_critic") < visited.index("check_convergence")

    async def test_final_check_convergence_decision_uses_R5(self):
        """The decision that routes to check_convergence must carry rule R5."""
        graph = self._build()
        with patch(
            "orchestration.app.graph.nodes.get_inference_client",
            return_value=_mock_client(),
        ):
            result = await graph.ainvoke(_base_state(), config={"recursion_limit": 200})

        conv_decisions = [
            d for d in result["supervisor_decisions"]
            if d["next_node"] == "check_convergence"
        ]
        assert conv_decisions, "no decision routed to check_convergence"
        assert conv_decisions[-1]["rule"] == "R5"

    async def test_iteration_history_contains_both_roles(self):
        """History must record coder, reviewer, and critic entries (review_critic = reviewer+critic)."""
        graph = self._build()
        with patch(
            "orchestration.app.graph.nodes.get_inference_client",
            return_value=_mock_client(),
        ):
            result = await graph.ainvoke(_base_state(), config={"recursion_limit": 200})

        roles_in_history = {e["role"] for e in result.get("iteration_history", [])}
        assert "coder" in roles_in_history
        # review_critic_node runs reviewer then critic internally
        assert "reviewer" in roles_in_history
        assert "critic" in roles_in_history


# ── Item 2: legacy pipeline integration ──────────────────────────────────────


@pytest.mark.asyncio
class TestLegacyPipelineIntegration:
    def _build(self):
        from orchestration.app.graph.pipeline import _build_legacy_pipeline
        return _build_legacy_pipeline()

    async def test_compiles_without_error(self):
        graph = self._build()
        assert graph is not None
        assert hasattr(graph, "ainvoke")

    async def test_runs_to_completion(self):
        """Legacy graph must set final_output for a simple 2-agent chain."""
        graph = self._build()
        with patch(
            "orchestration.app.graph.nodes.get_inference_client",
            return_value=_mock_client(_CODER_OUTPUT),
        ):
            result = await graph.ainvoke(_base_state(), config={"recursion_limit": 200})

        assert result.get("final_output"), "final_output must be non-empty"

    async def test_chain_index_advances_through_chain(self):
        """Legacy pipeline must advance agent_chain_index past all chain roles."""
        graph = self._build()
        with patch(
            "orchestration.app.graph.nodes.get_inference_client",
            return_value=_mock_client(),
        ):
            result = await graph.ainvoke(_base_state(), config={"recursion_limit": 200})

        # coder increments to 1, review_critic (=reviewer+critic) to 2+
        assert result.get("agent_chain_index", 0) >= 2

    async def test_no_supervisor_decisions_in_legacy_mode(self):
        """supervisor_decisions must stay empty in the legacy pipeline."""
        graph = self._build()
        with patch(
            "orchestration.app.graph.nodes.get_inference_client",
            return_value=_mock_client(),
        ):
            result = await graph.ainvoke(_base_state(), config={"recursion_limit": 200})

        assert result.get("supervisor_decisions", []) == []


# ── Item 3 (parity): both pipelines, same prompt ─────────────────────────────


@pytest.mark.asyncio
class TestPipelineParity:
    async def test_both_produce_final_output(self):
        """Supervisor and legacy must both complete with a non-empty final_output."""
        from orchestration.app.graph.pipeline import (
            _build_legacy_pipeline,
            _build_supervisor_pipeline,
        )

        sup_graph = _build_supervisor_pipeline()
        leg_graph = _build_legacy_pipeline()
        state = _base_state()

        with patch(
            "orchestration.app.graph.nodes.get_inference_client",
            return_value=_mock_client(_CODER_OUTPUT),
        ):
            sup_result = await sup_graph.ainvoke(state, config={"recursion_limit": 200})
            leg_result = await leg_graph.ainvoke(state, config={"recursion_limit": 200})

        assert sup_result.get("final_output"), "supervisor: missing final_output"
        assert leg_result.get("final_output"), "legacy: missing final_output"

    async def test_both_call_coder_for_same_prompt(self):
        """Both pipelines must invoke the coder role for the same user prompt."""
        from orchestration.app.graph.pipeline import (
            _build_legacy_pipeline,
            _build_supervisor_pipeline,
        )

        sup_roles: list[str] = []
        leg_roles: list[str] = []

        sup_graph = _build_supervisor_pipeline()
        leg_graph = _build_legacy_pipeline()

        sup_client = MagicMock()
        sup_client.generate = AsyncMock(
            side_effect=lambda req: (sup_roles.append(req.role) or _make_response())
        )
        leg_client = MagicMock()
        leg_client.generate = AsyncMock(
            side_effect=lambda req: (leg_roles.append(req.role) or _make_response())
        )

        state = _base_state()
        with patch("orchestration.app.graph.nodes.get_inference_client", return_value=sup_client):
            await sup_graph.ainvoke(state, config={"recursion_limit": 200})
        with patch("orchestration.app.graph.nodes.get_inference_client", return_value=leg_client):
            await leg_graph.ainvoke(state, config={"recursion_limit": 200})

        assert "coder" in sup_roles, f"supervisor did not call coder; called: {sup_roles}"
        assert "coder" in leg_roles, f"legacy did not call coder; called: {leg_roles}"

    async def test_both_run_evaluator_before_converging(self):
        """Both pipelines must run an evaluator role before the convergence node."""
        from orchestration.app.graph.pipeline import (
            _build_legacy_pipeline,
            _build_supervisor_pipeline,
        )

        evaluator_roles = {"reviewer", "critic", "review_critic"}

        sup_graph = _build_supervisor_pipeline()
        leg_graph = _build_legacy_pipeline()
        state = _base_state()

        with patch(
            "orchestration.app.graph.nodes.get_inference_client",
            return_value=_mock_client(),
        ):
            sup_result = await sup_graph.ainvoke(state, config={"recursion_limit": 200})
            leg_result = await leg_graph.ainvoke(state, config={"recursion_limit": 200})

        for label, result in [("supervisor", sup_result), ("legacy", leg_result)]:
            roles = {e["role"] for e in result.get("iteration_history", [])}
            assert roles & evaluator_roles, f"{label}: no evaluator in history; roles={roles}"
