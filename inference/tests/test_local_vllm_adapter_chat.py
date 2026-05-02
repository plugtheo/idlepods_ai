"""Tests for LocalVLLMBackend adapter routing via tool_call_style."""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inference.app.backends import local_vllm as _lv_module
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
    backend = LocalVLLMBackend("primary", entry)
    return backend


def _mock_chat_response(content="hello", tool_calls=None):
    msg = {"content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {
        "choices": [{"message": msg, "finish_reason": "stop"}],
        "usage": {"completion_tokens": 5},
    }


def _mock_completions_response(text="hello"):
    return {
        "choices": [{"text": text, "finish_reason": "stop"}],
        "usage": {"completion_tokens": 5},
    }


@pytest.mark.asyncio
async def test_openai_native_adapter_uses_chat_completions():
    """tool_call_style=openai_native → /v1/chat/completions, _build_adapter_prompt NOT called."""
    _lv_module._adapter_tool_call_style_cache["coding_lora"] = "openai_native"

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = _mock_chat_response()

    backend = _make_backend()
    # Mock the registry so adapter is available
    backend._registry.adapter_available = AsyncMock(return_value=True)

    with patch.object(backend._client, "post", return_value=mock_resp) as mock_post, \
         patch.object(_lv_module, "_build_adapter_prompt") as mock_bap:
        mock_post_coro = AsyncMock(return_value=mock_resp)
        backend._client.post = mock_post_coro
        await backend.generate(_make_request())

    calls = mock_post_coro.call_args_list
    assert len(calls) == 1
    url = calls[0][0][0] if calls[0][0] else calls[0][1].get("url", "")
    # Extract from positional arg
    call_url = calls[0].args[0] if calls[0].args else ""
    assert "/v1/chat/completions" in call_url, \
        f"Expected /v1/chat/completions in URL, got: {call_url}"
    mock_bap.assert_not_called()

    # Clean up cache
    del _lv_module._adapter_tool_call_style_cache["coding_lora"]


@pytest.mark.asyncio
async def test_none_style_adapter_uses_legacy_completions():
    """tool_call_style=none → /v1/completions + _build_adapter_prompt called."""
    _lv_module._adapter_tool_call_style_cache["coding_lora"] = "none"

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = _mock_completions_response("def hello(): pass")

    backend = _make_backend()
    backend._registry.adapter_available = AsyncMock(return_value=True)

    mock_post_coro = AsyncMock(return_value=mock_resp)
    backend._client.post = mock_post_coro

    await backend.generate(_make_request())

    calls = mock_post_coro.call_args_list
    assert len(calls) == 1
    call_url = calls[0].args[0] if calls[0].args else ""
    assert "/v1/completions" in call_url and "/v1/chat/completions" not in call_url, \
        f"Expected /v1/completions, got: {call_url}"

    # Clean up cache
    del _lv_module._adapter_tool_call_style_cache["coding_lora"]


@pytest.mark.asyncio
async def test_openai_native_adapter_propagates_tools():
    """tools= from request are forwarded to /v1/chat/completions payload."""
    _lv_module._adapter_tool_call_style_cache["coding_lora"] = "openai_native"

    from shared.contracts.inference import ToolDefinition
    tool = ToolDefinition(type="function", function={
        "name": "read_file",
        "description": "Read a file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
    })

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = _mock_chat_response()

    backend = _make_backend()
    backend._registry.adapter_available = AsyncMock(return_value=True)
    mock_post_coro = AsyncMock(return_value=mock_resp)
    backend._client.post = mock_post_coro

    await backend.generate(_make_request(tools=[tool]))

    calls = mock_post_coro.call_args_list
    assert calls
    payload = calls[0].kwargs.get("json", {}) or (calls[0].args[1] if len(calls[0].args) > 1 else {})
    assert "tools" in payload, "tools must be forwarded to the chat/completions payload"

    del _lv_module._adapter_tool_call_style_cache["coding_lora"]
