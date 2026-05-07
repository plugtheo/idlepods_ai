"""Tests for LocalVLLMBackend adapter routing via /v1/chat/completions."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from shared.contracts.inference import GenerateRequest, Message


def _make_request(adapter_name="coding_lora", tools=None):
    return GenerateRequest(
        backend="primary",
        role="coder",
        messages=[
            Message(role="system", content="You are a coder."),
            Message(role="user", content="Write hello."),
        ],
        adapter_name=adapter_name,
        max_tokens=128,
        session_id="test-session",
        tools=tools,
    )


def _make_backend():
    from shared.contracts.models import BackendEntry
    entry = BackendEntry(
        model_id="test-model",
        served_url="http://vllm:8000",
    )
    from inference.app.backends.local_vllm import LocalVLLMBackend
    return LocalVLLMBackend("primary", entry)


def _mock_chat_response(content="hello", tool_calls=None):
    msg = {"content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {
        "choices": [{"message": msg, "finish_reason": "stop"}],
        "usage": {"completion_tokens": 5},
    }


@pytest.mark.asyncio
async def test_adapter_always_uses_chat_completions():
    """All adapters route to /v1/chat/completions regardless of any legacy config."""
    backend = _make_backend()
    backend._registry.adapter_available = AsyncMock(return_value=True)

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = _mock_chat_response()
    mock_post_coro = AsyncMock(return_value=mock_resp)
    backend._client.post = mock_post_coro

    await backend.generate(_make_request())

    calls = mock_post_coro.call_args_list
    assert len(calls) == 1
    call_url = calls[0].args[0] if calls[0].args else ""
    assert "/v1/chat/completions" in call_url, \
        f"Expected /v1/chat/completions in URL, got: {call_url}"


@pytest.mark.asyncio
async def test_adapter_propagates_tools():
    """tools= from request are forwarded to /v1/chat/completions payload."""
    from shared.contracts.inference import ToolDefinition
    tool = ToolDefinition(type="function", function={
        "name": "read_file",
        "description": "Read a file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
    })

    backend = _make_backend()
    backend._registry.adapter_available = AsyncMock(return_value=True)

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = _mock_chat_response()
    mock_post_coro = AsyncMock(return_value=mock_resp)
    backend._client.post = mock_post_coro

    await backend.generate(_make_request(tools=[tool]))

    calls = mock_post_coro.call_args_list
    assert calls
    payload = calls[0].kwargs.get("json", {}) or (calls[0].args[1] if len(calls[0].args) > 1 else {})
    assert "tools" in payload, "tools must be forwarded to the chat/completions payload"
