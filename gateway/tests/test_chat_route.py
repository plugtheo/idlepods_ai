"""
Tests for the Gateway /v1/chat route.

Uses FastAPI TestClient with the full gateway app.
The orchestration client is mocked to avoid real HTTP calls.
"""
import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient


def _make_orch_response(**kwargs):
    from shared.contracts.orchestration import OrchestrationResponse
    defaults = {
        "session_id": "sess-test",
        "output": "Here is the result.",
        "success": True,
        "confidence": 0.9,
        "iterations": 2,
        "best_score": 0.9,
        "converged": True,
        "agent_steps": [],
    }
    defaults.update(kwargs)
    return OrchestrationResponse(**defaults)


@pytest.fixture
def gateway_client():
    """Gateway TestClient with auth disabled and mocked orchestration client."""
    with (
        patch("services.gateway.app.middleware.auth.settings") as auth_settings,
        patch("services.gateway.app.config.settings.settings") as gw_settings,
    ):
        auth_settings.api_key = ""
        gw_settings.api_key = ""
        gw_settings.debug = False
        gw_settings.port = 8080

        from services.gateway.app.main import app
        yield TestClient(app)


class TestChatRoute:
    @pytest.fixture(autouse=True)
    def mock_run_pipeline(self):
        """Replace the real orchestration HTTP call with a mock."""
        with patch(
            "services.gateway.app.routes.chat.run_pipeline",
            new_callable=AsyncMock,
            return_value=_make_orch_response(),
        ) as mock:
            yield mock

    def test_health_endpoint(self, gateway_client):
        resp = gateway_client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_chat_returns_200(self, gateway_client):
        resp = gateway_client.post(
            "/v1/chat",
            json={"prompt": "Write a quicksort function"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["output"] == "Here is the result."
        assert body["success"] is True

    def test_chat_response_schema(self, gateway_client):
        resp = gateway_client.post(
            "/v1/chat",
            json={"prompt": "debug this code"},
        )
        body = resp.json()
        assert "session_id" in body
        assert "output" in body
        assert "success" in body
        assert "confidence" in body
        assert "iterations" in body
        assert "converged" in body

    def test_chat_with_session_id(self, gateway_client):
        resp = gateway_client.post(
            "/v1/chat",
            json={"prompt": "plan a project", "session_id": "my-session"},
        )
        assert resp.status_code == 200

    def test_missing_prompt_returns_422(self, gateway_client):
        resp = gateway_client.post("/v1/chat", json={})
        assert resp.status_code == 422

    def test_orchestration_error_returns_502(self, gateway_client, mock_run_pipeline):
        mock_run_pipeline.side_effect = Exception("connection refused")
        resp = gateway_client.post(
            "/v1/chat",
            json={"prompt": "do something"},
        )
        assert resp.status_code == 502
