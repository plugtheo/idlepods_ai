"""
Tests for the Experience Service /v1/experience route.

JSONL store, vector store, and training client are mocked.
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient


def _event_payload(session_id="sess-exp"):
    return {
        "session_id": session_id,
        "prompt": "write a REST API",
        "final_output": "from fastapi import FastAPI\napp = FastAPI()",
        "agent_chain": ["planner", "coder", "reviewer"],
        "contributions": [
            {
                "role": "coder",
                "output": "from fastapi import FastAPI",
                "quality_score": 0.85,
                "iteration": 1,
            }
        ],
        "final_score": 0.85,
        "iterations": 2,
        "converged": True,
    }


@pytest.fixture
def experience_client():
    from services.experience.app.main import app
    return TestClient(app)


class TestRecordRoute:
    @pytest.fixture(autouse=True)
    def mock_storage(self):
        with (
            patch(
                "services.experience.app.routes.record.append_experience",
                new_callable=AsyncMock,
            ),
            patch(
                "services.experience.app.routes.record.upsert_experience",
                new_callable=AsyncMock,
            ),
            patch(
                "services.experience.app.routes.record.fire_notify_training",
            ),
        ):
            yield

    def test_health(self, experience_client):
        resp = experience_client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_record_experience_returns_200(self, experience_client):
        resp = experience_client.post("/v1/experience", json=_event_payload())
        assert resp.status_code == 200
        body = resp.json()
        assert body["stored"] is True
        assert body["session_id"] == "sess-exp"

    def test_record_missing_fields_returns_422(self, experience_client):
        resp = experience_client.post(
            "/v1/experience",
            json={"session_id": "s1"},  # missing required fields
        )
        assert resp.status_code == 422

    def test_record_storage_error_returns_500(self, experience_client):
        with patch(
            "services.experience.app.routes.record.append_experience",
            new_callable=AsyncMock,
            side_effect=OSError("disk full"),
        ):
            resp = experience_client.post("/v1/experience", json=_event_payload())
        assert resp.status_code == 500

    def test_record_triggers_training_notification(self, experience_client):
        """fire_notify_training should be called once per record."""
        with patch(
            "services.experience.app.routes.record.fire_notify_training"
        ) as mock_notify:
            resp = experience_client.post("/v1/experience", json=_event_payload())
        assert resp.status_code == 200
        mock_notify.assert_called_once()
