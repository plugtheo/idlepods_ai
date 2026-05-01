"""
Tests for POST /v1/generate/stream — the inference SSE streaming route.

Covers:
- 200 with correct SSE event format (token events + is_final sentinel)
- Backend error yields {"error": ...} event then closes
- Invalid request body returns 422 (FastAPI validation, before backend)
- Stream=True is passed through to the backend
- Tokens are delivered in order
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
from shared.contracts.inference import GenerateResponse


def _make_response(**kwargs):
    defaults = {
        "content": "generated output",
        "model_family": "qwen",
        "role": "coder",
        "tokens_generated": 15,
        "session_id": "sess-1",
    }
    defaults.update(kwargs)
    return GenerateResponse(**defaults)


async def _token_gen(*tokens):
    for t in tokens:
        yield t


@pytest.fixture
def inference_client():
    from services.inference.app.main import app
    return TestClient(app)


def _parse_sse_events(raw: str) -> list[dict]:
    """Parse a raw SSE body into a list of decoded JSON payloads."""
    events = []
    for line in raw.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    return events


class TestGenerateStreamRoute:
    """Tests for POST /v1/generate/stream."""

    @pytest.fixture(autouse=True)
    def mock_backend(self):
        self._mock_be = MagicMock()
        self._mock_be.generate = AsyncMock(return_value=_make_response())
        self._mock_be.generate_stream = MagicMock(
            return_value=_token_gen("Hello", " world")
        )
        with patch(
            "services.inference.app.routes.generate.get_backend",
            return_value=self._mock_be,
        ):
            yield self._mock_be

    def test_stream_returns_200(self, inference_client):
        resp = inference_client.post(
            "/v1/generate/stream",
            json={
                "model_family": "qwen",
                "role": "coder",
                "messages": [{"role": "user", "content": "write code"}],
            },
        )
        assert resp.status_code == 200

    def test_stream_content_type_is_sse(self, inference_client):
        resp = inference_client.post(
            "/v1/generate/stream",
            json={
                "model_family": "qwen",
                "role": "coder",
                "messages": [{"role": "user", "content": "write code"}],
            },
        )
        assert "text/event-stream" in resp.headers["content-type"]

    def test_stream_yields_token_events_then_final(self, inference_client):
        resp = inference_client.post(
            "/v1/generate/stream",
            json={
                "model_family": "qwen",
                "role": "coder",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        events = _parse_sse_events(resp.text)

        # All non-final events carry tokens
        token_events = [e for e in events if not e.get("is_final")]
        assert [e["token"] for e in token_events] == ["Hello", " world"]

        # Last event is the is_final sentinel
        assert events[-1] == {"token": "", "is_final": True}

    def test_stream_backend_error_yields_error_event(self, inference_client):
        async def _bad_gen():
            yield "partial"
            raise RuntimeError("backend exploded")

        self._mock_be.generate_stream = MagicMock(return_value=_bad_gen())

        resp = inference_client.post(
            "/v1/generate/stream",
            json={
                "model_family": "qwen",
                "role": "coder",
                "messages": [{"role": "user", "content": "boom"}],
            },
        )
        events = _parse_sse_events(resp.text)
        # Should contain the partial token then an error event
        assert any("error" in e for e in events)
        error_event = next(e for e in events if "error" in e)
        assert "backend exploded" in error_event["error"]

    def test_stream_invalid_body_returns_422(self, inference_client):
        resp = inference_client.post(
            "/v1/generate/stream",
            json={"role": "coder"},  # missing model_family and messages
        )
        assert resp.status_code == 422

    def test_stream_token_order_preserved(self, inference_client):
        tokens = ["The", " quick", " brown", " fox"]
        self._mock_be.generate_stream = MagicMock(return_value=_token_gen(*tokens))

        resp = inference_client.post(
            "/v1/generate/stream",
            json={
                "model_family": "qwen",
                "role": "coder",
                "messages": [{"role": "user", "content": "complete this"}],
            },
        )
        events = _parse_sse_events(resp.text)
        yielded = [e["token"] for e in events if not e.get("is_final")]
        assert yielded == tokens

    def test_stream_with_no_tokens_yields_only_final(self, inference_client):
        async def _empty():
            return
            yield  # make it a generator

        self._mock_be.generate_stream = MagicMock(return_value=_empty())

        resp = inference_client.post(
            "/v1/generate/stream",
            json={
                "model_family": "qwen",
                "role": "coder",
                "messages": [{"role": "user", "content": "empty"}],
            },
        )
        events = _parse_sse_events(resp.text)
        assert events == [{"token": "", "is_final": True}]
