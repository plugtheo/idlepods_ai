"""
Tests for the Inference Service /v1/generate route.

Uses FastAPI TestClient. The backend is mocked to avoid real model calls.
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient

from shared.contracts.inference import GenerateResponse


def _make_response(**kwargs):
    defaults = {
        "content": "generated output",
        "backend": "primary",
        "role": "coder",
        "tokens_generated": 15,
        "session_id": "sess-1",
    }
    defaults.update(kwargs)
    return GenerateResponse(**defaults)


@pytest.fixture
def inference_client():
    from services.inference.app.main import app
    return TestClient(app)


class TestGenerateRoute:
    @pytest.fixture(autouse=True)
    def mock_backend(self):
        from shared.contracts.models import BackendEntry, ModelsRegistry

        entry = BackendEntry(served_url="http://vllm-primary:8000", model_id="Qwen/Qwen3-14B", backend_type="local_vllm")
        registry = ModelsRegistry(default_backend="primary", backends={"primary": entry})

        mock_be = MagicMock()
        mock_be.generate = AsyncMock(return_value=_make_response())
        mock_be.health = AsyncMock(return_value={"backend": "local_vllm", "url": "http://vllm-primary:8000", "status": "ok"})
        mock_be.list_adapters = AsyncMock(return_value=[])

        with patch("services.inference.app.routes.generate.load_registry", return_value=registry), \
             patch("services.inference.app.routes.generate.get_backend", return_value=mock_be):
            yield mock_be

    def test_health(self, inference_client):
        resp = inference_client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_generate_returns_200(self, inference_client):
        resp = inference_client.post(
            "/v1/generate",
            json={
                "backend": "primary",
                "role": "coder",
                "messages": [{"role": "user", "content": "write a function"}],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["content"] == "generated output"
        assert body["role"] == "coder"

    def test_generate_invalid_payload_returns_422(self, inference_client):
        resp = inference_client.post(
            "/v1/generate",
            json={"role": "coder"},  # missing model_family and messages
        )
        assert resp.status_code == 422

    def test_generate_bad_backend_returns_400(self, inference_client, mock_backend):
        mock_backend.generate = AsyncMock(side_effect=ValueError("Unknown backend 'llama'"))
        resp = inference_client.post(
            "/v1/generate",
            json={
                "backend": "llama",
                "role": "coder",
                "messages": [{"role": "user", "content": "x"}],
            },
        )
        assert resp.status_code == 400
        assert "Unknown backend" in resp.json()["detail"]

    def test_generate_backend_error_returns_502(self, inference_client, mock_backend):
        mock_backend.generate = AsyncMock(side_effect=Exception("vLLM crashed"))
        resp = inference_client.post(
            "/v1/generate",
            json={
                "backend": "primary",
                "role": "coder",
                "messages": [{"role": "user", "content": "x"}],
            },
        )
        assert resp.status_code == 502
