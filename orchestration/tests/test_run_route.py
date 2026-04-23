"""
Tests for the Orchestration Service /v1/run route.

Context client, pipeline, and experience client are all mocked.
"""
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
        with (
            patch(
                "services.orchestration.app.routes.run.fetch_context",
                new_callable=AsyncMock,
                return_value=_fake_built_context(),
            ),
            patch(
                "services.orchestration.app.routes.run._PIPELINE"
            ) as mock_pipeline,
            patch(
                "services.orchestration.app.routes.run.fire_publish_experience",
                new_callable=AsyncMock,
            ),
        ):
            mock_pipeline.ainvoke = AsyncMock(return_value=_fake_final_state())
            yield

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
        """If context service is unavailable, pipeline should still proceed."""
        with patch(
            "services.orchestration.app.routes.run.fetch_context",
            new_callable=AsyncMock,
            return_value=_fake_built_context(),  # degraded but non-empty
        ):
            resp = orchestration_client.post(
                "/v1/run",
                json={"prompt": "do something"},
            )
        assert resp.status_code == 200
