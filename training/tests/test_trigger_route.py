"""
Tests for the Training Service /v1/training/trigger route.

experience_reader and trainer_launcher are mocked.
"""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient


def _diverse_records(n=60):
    return [
        {"prompt": f"unique prompt {i}", "final_score": 0.5 + i * 0.005}
        for i in range(n)
    ]


def _trigger_payload(capability="coding", count=60):
    return {"capability": capability, "new_experience_count": count}


@pytest.fixture
def training_client():
    from services.training.app.main import app
    return TestClient(app)


class TestTriggerRoute:
    @pytest.fixture(autouse=True)
    def reset_training_state(self):
        """Ensure _training_running is False before each test."""
        from services.training.app.utils import trainer_launcher
        trainer_launcher._training_running = False
        yield
        trainer_launcher._training_running = False

    def test_health(self, training_client):
        resp = training_client.get("/health")
        assert resp.status_code == 200

    def test_triggers_when_criteria_met(self, training_client):
        with (
            patch(
                "services.training.app.routes.trigger.load_experiences",
                return_value=_diverse_records(60),
            ),
            patch(
                "services.training.app.routes.trigger.check_diversity",
                return_value=(True, "criteria met: n=60, spread=0.30, diversity=0.80"),
            ),
            patch(
                "services.training.app.routes.trigger.to_training_records",
                return_value=[{"problem": "x", "solution": "y", "evaluation": 0.8}],
            ),
            patch(
                "services.training.app.routes.trigger.launch_training",
                new_callable=AsyncMock,
            ) as mock_launch,
            patch(
                "services.training.app.routes.trigger.is_training_running",
                return_value=False,
            ),
        ):
            resp = training_client.post(
                "/v1/training/trigger",
                json=_trigger_payload(),
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["triggered"] is True

    def test_not_triggered_when_diversity_fails(self, training_client):
        with (
            patch(
                "services.training.app.routes.trigger.load_experiences",
                return_value=[{"prompt": "p", "final_score": 0.7}] * 10,
            ),
            patch(
                "services.training.app.routes.trigger.check_diversity",
                return_value=(False, "too few experiences: 10 < 50"),
            ),
            patch(
                "services.training.app.routes.trigger.is_training_running",
                return_value=False,
            ),
        ):
            resp = training_client.post(
                "/v1/training/trigger",
                json=_trigger_payload(),
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["triggered"] is False
        assert "too few" in body["reason"]

    def test_not_triggered_when_training_already_running(self, training_client):
        with patch(
            "services.training.app.routes.trigger.is_training_running",
            return_value=True,
        ):
            resp = training_client.post(
                "/v1/training/trigger",
                json=_trigger_payload(),
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["triggered"] is False
        assert "in progress" in body["reason"]

    def test_missing_payload_returns_422(self, training_client):
        resp = training_client.post("/v1/training/trigger", json={})
        assert resp.status_code == 422

    def test_status_endpoint(self, training_client):
        resp = training_client.get("/v1/training/status")
        assert resp.status_code == 200
        assert "running" in resp.json()
