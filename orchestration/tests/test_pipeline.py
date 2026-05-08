"""
Tests for the orchestration pipeline builder.

Covers:
- build_pipeline() returns a compiled graph without errors
- _recursion_limit() calculation
- _finalize_state() returns complete final output
"""
import pytest
from unittest.mock import patch, MagicMock


class TestBuildPipeline:
    def test_pipeline_compiles_without_error(self):
        """build_pipeline() should succeed with LangGraph installed."""
        from services.orchestration.app.graph.pipeline import build_pipeline
        pipeline = build_pipeline()
        assert pipeline is not None

    def test_pipeline_has_ainvoke(self):
        """Compiled graph must have ainvoke for async execution."""
        from services.orchestration.app.graph.pipeline import build_pipeline
        pipeline = build_pipeline()
        assert hasattr(pipeline, "ainvoke")


class TestRecursionLimit:
    def test_default_limit(self):
        from services.orchestration.app.graph.pipeline import _recursion_limit
        # Default 5 iterations, 7 nodes → should be well above 35
        limit = _recursion_limit(max_iterations=5, chain_length=7)
        assert limit > 35

    def test_single_iteration(self):
        from services.orchestration.app.graph.pipeline import _recursion_limit
        limit = _recursion_limit(max_iterations=1, chain_length=2)
        assert limit >= 2

    def test_scales_with_iterations(self):
        from services.orchestration.app.graph.pipeline import _recursion_limit
        limit_5 = _recursion_limit(max_iterations=5, chain_length=4)
        limit_10 = _recursion_limit(max_iterations=10, chain_length=4)
        assert limit_10 > limit_5


class TestUpdateLoopState:
    """Regression tests for _update_loop_state, including the B2 plan-step reset fix."""

    @pytest.mark.asyncio
    async def test_b2_resets_last_done_step_to_pending_in_supervisor_mode(self):
        """
        B2 fix: in supervisor mode, _update_loop_state must reset the last 'done'
        plan step to 'pending' so iteration ≥ 2 produces work-dispatching decisions.
        """
        from unittest.mock import patch
        from orchestration.app.graph.pipeline import _update_loop_state

        plan = {
            "steps": [
                {"id": "s1", "status": "done", "owner": "coder"},
                {"id": "s2", "status": "done", "owner": "reviewer"},
            ],
            "updated_at": "2025-01-01T00:00:00+00:00",
        }
        state = {
            "current_iteration": 1,
            "iteration_history": [],
            "iteration_scores": [],
            "best_score": 0.0,
            "best_output": "",
            "last_output": "some output",
            "plan": plan,
            "agent_chain_index": 2,
        }
        with patch("orchestration.app.config.settings.settings") as mock_s:
            mock_s.pipeline_use_supervisor = True
            delta = await _update_loop_state(state)

        assert delta["current_iteration"] == 2
        assert delta["agent_chain_index"] == 0
        # Last done step (s2) must be reset to pending
        updated_steps = {s["id"]: s["status"] for s in delta["plan"]["steps"]}
        assert updated_steps["s1"] == "done"
        assert updated_steps["s2"] == "pending"
        assert delta.get("plan_changed") is True

    @pytest.mark.asyncio
    async def test_update_loop_does_not_touch_plan_in_legacy_mode(self):
        """In legacy mode, plan must not be modified by _update_loop_state."""
        from unittest.mock import patch
        from orchestration.app.graph.pipeline import _update_loop_state

        plan = {
            "steps": [{"id": "s1", "status": "done", "owner": "coder"}],
            "updated_at": "2025-01-01T00:00:00+00:00",
        }
        state = {
            "current_iteration": 1,
            "iteration_history": [],
            "iteration_scores": [],
            "best_score": 0.0,
            "best_output": "",
            "last_output": "output",
            "plan": plan,
            "agent_chain_index": 1,
        }
        with patch("orchestration.app.config.settings.settings") as mock_s:
            mock_s.pipeline_use_supervisor = False
            delta = await _update_loop_state(state)

        assert "plan" not in delta
        assert delta.get("plan_changed") is None

    @pytest.mark.asyncio
    async def test_update_loop_skips_plan_reset_when_no_done_steps(self):
        """If no steps are done, plan must not be modified (nothing to reset)."""
        from unittest.mock import patch
        from orchestration.app.graph.pipeline import _update_loop_state

        plan = {
            "steps": [{"id": "s1", "status": "pending", "owner": "coder"}],
            "updated_at": "2025-01-01T00:00:00+00:00",
        }
        state = {
            "current_iteration": 1,
            "iteration_history": [],
            "iteration_scores": [],
            "best_score": 0.0,
            "best_output": "",
            "last_output": "output",
            "plan": plan,
            "agent_chain_index": 0,
        }
        with patch("orchestration.app.config.settings.settings") as mock_s:
            mock_s.pipeline_use_supervisor = True
            delta = await _update_loop_state(state)

        assert "plan" not in delta


class TestSupervisorPipelineBuild:
    def test_supervisor_pipeline_compiles(self):
        """_build_supervisor_pipeline() must return a compiled graph."""
        from orchestration.app.graph.pipeline import _build_supervisor_pipeline
        graph = _build_supervisor_pipeline()
        assert graph is not None
        assert hasattr(graph, "ainvoke")

    def test_recursion_limit_supervisor_mode(self):
        """Supervisor recursion limit must scale with max_iter and max_steps."""
        from unittest.mock import patch
        from orchestration.app.graph.pipeline import _recursion_limit
        with patch("orchestration.app.config.settings.settings") as s:
            s.pipeline_use_supervisor = True
            s.pipeline_supervisor_max_steps = 8
            limit_3 = _recursion_limit(max_iterations=3, chain_length=0)
            limit_6 = _recursion_limit(max_iterations=6, chain_length=0)
        assert limit_6 > limit_3

    def test_recursion_limit_supervisor_greater_than_legacy(self):
        """Supervisor limit must be larger than legacy for same iteration count."""
        from unittest.mock import patch
        from orchestration.app.graph.pipeline import _recursion_limit
        leg_limit = _recursion_limit(max_iterations=5, chain_length=4)
        with patch("orchestration.app.config.settings.settings") as s:
            s.pipeline_use_supervisor = True
            s.pipeline_supervisor_max_steps = 8
            sup_limit = _recursion_limit(max_iterations=5, chain_length=0)
        assert sup_limit > leg_limit
