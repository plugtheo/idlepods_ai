"""
Tests for LocalVLLMBackend.

Covers:
- Successful generation with adapter (adapter available in registry)
- Successful generation, adapter falls back to base model (not registered)
- Generation with no adapter uses base model directly
- Unknown family at factory level raises ValueError
- HTTP error from vLLM propagates
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from shared.contracts.inference import GenerateRequest, GenerateResponse, Message

_BACKEND_URL = "http://vllm-primary:8000"
_BACKEND_MODEL = "Qwen/Qwen3-14B"


def _make_request(backend="primary", role="coder", adapter=None):
    return GenerateRequest(
        backend=backend,
        role=role,
        messages=[Message(role="user", content="write code")],
        adapter_name=adapter,
        session_id="sess-1",
    )


def _make_chat_completions_response(content="def hello(): pass", tokens=10):
    """Mock for /v1/chat/completions (base model path)."""
    return MagicMock(
        json=MagicMock(
            return_value={
                "choices": [{"message": {"content": content}}],
                "usage": {"completion_tokens": tokens},
            }
        ),
        raise_for_status=MagicMock(),
    )


def _make_completions_response(content="def hello(): pass", tokens=10):
    """Mock for /v1/completions (adapter path)."""
    return MagicMock(
        json=MagicMock(
            return_value={
                "choices": [{"text": content}],
                "usage": {"completion_tokens": tokens},
            }
        ),
        raise_for_status=MagicMock(),
    )


def _make_backend(mock_client):
    """Return a LocalVLLMBackend with the httpx client replaced by mock_client."""
    from services.inference.app.backends.local_vllm import LocalVLLMBackend
    from shared.contracts.models import BackendEntry
    entry = BackendEntry(served_url=_BACKEND_URL, model_id=_BACKEND_MODEL)
    backend = LocalVLLMBackend("primary", entry)
    backend._client = mock_client
    backend._registry._client = mock_client
    backend._registry._fetched_at = float("inf")  # prevent HTTP refresh
    return backend


@pytest.mark.asyncio
class TestLocalVLLMBackend:
    async def test_generate_with_available_adapter(self):
        """When adapter is registered, request uses the adapter model name."""
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=_make_completions_response("print('hi')", tokens=5))

        backend = _make_backend(mock_client)
        backend._registry._known = {f"{_BACKEND_MODEL}/coding_lora"}

        resp = await backend.generate(_make_request(adapter="coding_lora"))

        assert isinstance(resp, GenerateResponse)
        assert resp.content == "print('hi')"
        assert resp.tokens_generated == 5
        assert resp.backend == "primary"

        payload_sent = mock_client.post.call_args[1]["json"]
        assert payload_sent["model"] == "coding_lora"

    async def test_generate_falls_back_to_base_when_adapter_missing(self):
        """When adapter is NOT registered, base model is used and no error raised."""
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=_make_chat_completions_response("result"))

        backend = _make_backend(mock_client)
        backend._registry._known = set()

        resp = await backend.generate(_make_request(adapter="coding_lora"))

        assert resp.content == "result"
        payload_sent = mock_client.post.call_args[1]["json"]
        assert "/coding_lora" not in payload_sent["model"]

    async def test_generate_no_adapter(self):
        """When no adapter requested, base model is used directly."""
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=_make_chat_completions_response("answer"))

        backend = _make_backend(mock_client)

        resp = await backend.generate(_make_request(adapter=None))

        assert resp.content == "answer"

    async def test_unknown_family_raises_at_factory(self):
        """get_backend raises ValueError for an unknown model family."""
        import services.inference.app.backends.factory as factory_mod
        factory_mod._backends.clear()

        with pytest.raises(ValueError):
            factory_mod.get_backend("nonexistent_backend_xyz")

    async def test_http_error_propagates(self):
        import httpx
        from services.inference.app.backends.base import InferenceError

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))

        backend = _make_backend(mock_client)
        backend._registry._known = set()

        with pytest.raises(Exception):
            await backend.generate(_make_request())
