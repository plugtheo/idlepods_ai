"""
Tests for LocalVLLMBackend.

Covers:
- Successful generation with adapter (adapter available)
- Successful generation, adapter falls back to base model (not registered)
- Unknown model_family raises ValueError
- HTTP error from vLLM propagates
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from shared.contracts.inference import GenerateRequest, GenerateResponse, Message


def _make_request(family="deepseek", role="coder", adapter=None):
    return GenerateRequest(
        model_family=family,
        role=role,
        messages=[Message(role="user", content="write code")],
        adapter_name=adapter,
        session_id="sess-1",
    )


def _make_vllm_response(content="def hello(): pass", tokens=10):
    return MagicMock(
        json=MagicMock(
            return_value={
                "choices": [{"message": {"content": content}}],
                "usage": {"completion_tokens": tokens},
            }
        ),
        raise_for_status=MagicMock(),
    )


@pytest.mark.asyncio
class TestLocalVLLMBackend:
    async def test_generate_with_available_adapter(self):
        """When adapter is registered, request uses qualified model name."""
        from services.inference.app.backends.local_vllm import LocalVLLMBackend, _REGISTRIES

        # Make the registry report the adapter as available
        _REGISTRIES["deepseek"]._known = {
            "deepseek-ai/deepseek-coder-6.7b-instruct/coding_lora"
        }
        _REGISTRIES["deepseek"]._fetched_at = float("inf")  # never refresh

        vllm_resp = _make_vllm_response("print('hi')", tokens=5)
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=vllm_resp)

        with patch("services.inference.app.backends.local_vllm.httpx.AsyncClient", return_value=mock_client):
            backend = LocalVLLMBackend()
            resp = await backend.generate(_make_request(adapter="coding_lora"))

        assert isinstance(resp, GenerateResponse)
        assert resp.content == "print('hi')"
        assert resp.tokens_generated == 5
        assert resp.model_family == "deepseek"

        # _resolve_model() returns the bare adapter_name; vLLM uses it as the model field
        payload_sent = mock_client.post.call_args[1]["json"]
        assert payload_sent["model"] == "coding_lora"

    async def test_generate_falls_back_to_base_when_adapter_missing(self):
        """When adapter is NOT registered, base model is used and no error raised."""
        from services.inference.app.backends.local_vllm import LocalVLLMBackend, _REGISTRIES

        # Registry empty — adapter not available
        _REGISTRIES["deepseek"]._known = set()
        _REGISTRIES["deepseek"]._fetched_at = float("inf")

        vllm_resp = _make_vllm_response("result")
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=vllm_resp)

        with patch("services.inference.app.backends.local_vllm.httpx.AsyncClient", return_value=mock_client):
            backend = LocalVLLMBackend()
            resp = await backend.generate(_make_request(adapter="coding_lora"))

        assert resp.content == "result"
        # Model should be the base model (no /adapter suffix)
        payload_sent = mock_client.post.call_args[1]["json"]
        assert "/coding_lora" not in payload_sent["model"]

    async def test_generate_no_adapter(self):
        """When no adapter requested, base model is used directly."""
        from services.inference.app.backends.local_vllm import LocalVLLMBackend

        vllm_resp = _make_vllm_response("answer")
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=vllm_resp)

        with patch("services.inference.app.backends.local_vllm.httpx.AsyncClient", return_value=mock_client):
            backend = LocalVLLMBackend()
            resp = await backend.generate(_make_request(adapter=None))

        assert resp.content == "answer"

    async def test_unknown_family_raises_value_error(self):
        from services.inference.app.backends.local_vllm import LocalVLLMBackend

        backend = LocalVLLMBackend()
        with pytest.raises(ValueError, match="Unknown model_family"):
            await backend.generate(_make_request(family="llama"))

    async def test_http_error_propagates(self):
        import httpx
        from services.inference.app.backends.local_vllm import LocalVLLMBackend, _REGISTRIES

        _REGISTRIES["mistral"]._known = set()
        _REGISTRIES["mistral"]._fetched_at = float("inf")

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))

        with patch("services.inference.app.backends.local_vllm.httpx.AsyncClient", return_value=mock_client):
            backend = LocalVLLMBackend()
            with pytest.raises(httpx.ConnectError):
                await backend.generate(_make_request(family="mistral", role="planner"))
