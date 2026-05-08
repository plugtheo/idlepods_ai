"""
Tests for the Orchestration Service /v1/run route.

Context builder, pipeline, and experience recorder are all mocked.
"""
import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from shared.contracts.context import BuiltContext
from shared.contracts.orchestration import OrchestrationResponse


def _fake_built_context():
    return BuiltContext(few_shots=[], repo_snippets=[], system_hints="")


def _fake_final_state():
    return {
        "session_id": "sess-123",
        "final_output": "def binary_search(): pass",
        "final_score": 0.90,
        "converged": True,
        "quality_converged": True,
        "current_iteration": 2,
        "agent_chain": ["planner", "coder", "reviewer"],
        "iteration_history": [
            {"iteration": 1, "role": "coder", "output": "def binary_search(): pass"}
        ],
        "best_score": 0.90,
    }


@pytest.fixture
def orchestration_client():
    from services.orchestration.app.main import app
    return TestClient(app)


class TestRunRoute:
    @pytest.fixture(autouse=True)
    def mock_all_clients(self):
        import services.orchestration.app.routes.run  # ensure module is imported before patching
        with (
            patch(
                "services.orchestration.app.routes.run.builder.build",
                new_callable=AsyncMock,
                return_value=_fake_built_context(),
            ),
            patch(
                "services.orchestration.app.routes.run._PIPELINE"
            ) as mock_pipeline,
            patch(
                "services.orchestration.app.routes.run.recorder.record",
                new_callable=AsyncMock,
            ),
            patch(
                "services.orchestration.app.routes.run.session_store.get_session",
                new_callable=AsyncMock,
                return_value=[],
            ) as mock_get_session,
            patch(
                "services.orchestration.app.routes.run.session_store.save_session",
                new_callable=AsyncMock,
            ) as mock_save_session,
        ):
            mock_pipeline.ainvoke = AsyncMock(return_value=_fake_final_state())
            yield mock_get_session, mock_save_session

    def test_health(self, orchestration_client):
        resp = orchestration_client.get("/health")
        assert resp.status_code == 200

    def test_run_returns_200(self, orchestration_client):
        resp = orchestration_client.post(
            "/v1/run",
            json={"prompt": "write a binary search function"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["output"] == "def binary_search(): pass"
        assert body["converged"] is True

    def test_run_response_schema(self, orchestration_client):
        resp = orchestration_client.post(
            "/v1/run",
            json={"prompt": "plan a new feature"},
        )
        body = resp.json()
        required_fields = {"session_id", "output", "success", "confidence", "iterations", "converged", "agent_steps"}
        assert required_fields.issubset(body.keys())

    def test_run_with_explicit_chain(self, orchestration_client):
        resp = orchestration_client.post(
            "/v1/run",
            json={
                "prompt": "fix the bug",
                "agent_chain": ["debugger", "reviewer"],
                "max_iterations": 3,
            },
        )
        assert resp.status_code == 200

    def test_missing_prompt_returns_422(self, orchestration_client):
        resp = orchestration_client.post("/v1/run", json={})
        assert resp.status_code == 422

    def test_context_failure_still_runs(self, orchestration_client):
        """If context build fails, pipeline should still proceed with empty context."""
        with patch(
            "services.orchestration.app.routes.run.builder.build",
            new_callable=AsyncMock,
            side_effect=Exception("chromadb unavailable"),
        ):
            resp = orchestration_client.post(
                "/v1/run",
                json={"prompt": "do something"},
            )
        assert resp.status_code == 200

    def test_run_loads_session_history(self, orchestration_client, mock_all_clients):
        mock_get_session, _ = mock_all_clients
        orchestration_client.post(
            "/v1/run",
            json={"prompt": "write a sort function", "task_id": "task-multi-1"},
        )
        mock_get_session.assert_called_once_with("task-multi-1")

    def test_run_saves_session_history(self, orchestration_client, mock_all_clients):
        _, mock_save_session = mock_all_clients
        orchestration_client.post(
            "/v1/run",
            json={"prompt": "write a sort function", "task_id": "task-multi-2"},
        )
        mock_save_session.assert_called_once()
        call_args = mock_save_session.call_args
        saved_task_id, saved_history, saved_ttl = call_args.args
        assert saved_task_id == "task-multi-2"
        assert len(saved_history) == 1
        assert saved_history[0]["role"] == "coder"


class TestStreamRoute:
    """SSE smoke test — verifies the supervisor pipeline path through the stream route."""

    def _fake_prepare_result(self, session_id="sse-test"):
        """Return the tuple that _prepare_pipeline_run normally produces."""
        initial_state = {
            "session_id": session_id,
            "task_id": session_id,
            "user_prompt": "write f",
            "agent_chain": ["coder", "review_critic"],
            "agent_chain_index": 0,
            "few_shots": [], "repo_snippets": [], "system_hints": "",
            "current_iteration": 1, "max_iterations": 1, "convergence_threshold": 0.0,
            "conversation_history": [], "iteration_history": [],
            "last_output": "", "iteration_scores": [],
            "best_score": 0.0, "best_output": "",
            "converged": False, "quality_converged": False,
            "final_output": "", "final_score": 0.0,
            "plan": None, "plan_changed": False, "current_step_id": None,
            "pending_tool_calls": [], "tool_steps_used": 0, "tool_originating_role": "",
            "supervisor_decisions": [],
        }
        return (session_id, "general", "simple", ["coder", "review_critic"],
                initial_state, 200, True, True)

    def _make_astream(self):
        """Async generator simulating one supervisor cycle: coder → review_critic → done."""

        async def _gen(*args, **kwargs):
            yield {"supervisor": {"supervisor_decisions": [
                {"next_node": "coder", "rule": "R4", "reason": "chain_template_dispatch",
                 "metadata": None, "ts": "2026-01-01T00:00:00+00:00", "iteration": 1}
            ]}}
            yield {"coder": {
                "iteration_history": [{"role": "coder", "iteration": 1,
                                        "output": "def f(): pass", "full_output": "def f(): pass",
                                        "timestamp": "2026-01-01T00:00:00+00:00",
                                        "failed": False, "validator_failed": False}],
                "last_output": "def f(): pass", "agent_chain_index": 1,
            }}
            yield {"supervisor": {"supervisor_decisions": [
                {"next_node": "review_critic", "rule": "R5", "reason": "chain_complete_evaluate",
                 "metadata": None, "ts": "2026-01-01T00:00:00+00:00", "iteration": 1}
            ]}}
            yield {"review_critic": {
                "iteration_history": [
                    {"role": "reviewer", "iteration": 1, "output": "SCORE: 0.90",
                     "full_output": "SCORE: 0.90", "timestamp": "2026-01-01T00:00:00+00:00",
                     "failed": False, "validator_failed": False},
                    {"role": "critic", "iteration": 1, "output": "Good",
                     "full_output": "Good", "timestamp": "2026-01-01T00:00:00+00:00",
                     "failed": False, "validator_failed": False},
                ],
                "last_output": "SCORE: 0.90", "agent_chain_index": 2,
            }}
            yield {"check_convergence": {}}
            yield {"finalize": {
                "converged": True, "quality_converged": True, "skip_consensus": True,
                "final_output": "def f(): pass", "best_score": 0.90,
                "best_output": "def f(): pass", "iteration_scores": [0.90],
            }}

        return _gen

    @pytest.mark.asyncio
    async def test_supervisor_stream_emits_start_and_done_events(self):
        """Stream route must emit 'start', 'agent_complete', and 'done' events
        when routed through the supervisor pipeline."""
        mock_pipeline = MagicMock()
        mock_pipeline.astream = self._make_astream()

        with (
            patch(
                "services.orchestration.app.routes.run._prepare_pipeline_run",
                new_callable=AsyncMock,
                return_value=self._fake_prepare_result(),
            ),
            patch("services.orchestration.app.routes.run._get_pipeline", return_value=mock_pipeline),
            patch("services.orchestration.app.routes.run.recorder.record", new_callable=AsyncMock),
            patch("services.orchestration.app.routes.run.session_store.save_session", new_callable=AsyncMock),
        ):
            from services.orchestration.app.main import app
            from httpx import AsyncClient, ASGITransport

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                collected_lines: list[str] = []
                async with ac.stream("POST", "/v1/run/stream", json={"prompt": "write f"}) as resp:
                    assert resp.status_code == 200
                    async for line in resp.aiter_lines():
                        if line.startswith("data:"):
                            collected_lines.append(line)

        event_types = []
        for line in collected_lines:
            try:
                payload = json.loads(line[len("data:"):].strip())
                event_types.append(payload.get("type"))
            except json.JSONDecodeError:
                pass

        assert "start" in event_types, f"'start' event missing; got: {event_types}"
        assert "done" in event_types, f"'done' event missing; got: {event_types}"
        assert "agent_complete" in event_types, f"'agent_complete' event missing; got: {event_types}"
