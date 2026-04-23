"""
Tests for APIBackend (LiteLLM).

Covers:
- Default model used when no role override configured
- Per-role model override applied correctly
- Role without override falls back to default model
- Response parsed correctly from litellm response
"""
import pytest
import os
from unittest.mock import AsyncMock, MagicMock, patch

from shared.contracts.inference import GenerateRequest, GenerateResponse, Message


def _make_litellm_response(content="generated text", tokens=20):
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage.completion_tokens = tokens
    return resp


def _make_request(role="coder", family="anthropic"):
    return GenerateRequest(
        model_family=family,
        role=role,
        messages=[Message(role="user", content="do something")],
        session_id="sess-api",
    )


def _make_settings(api_model="claude-3-5-haiku-20241022", overrides=None, api_key="test-key", provider="anthropic"):
    s = MagicMock()
    s.api_model = api_model
    s.role_model_overrides = overrides or {}
    s.api_key = api_key
    s.api_provider = provider
    return s


@pytest.mark.asyncio
class TestAPIBackend:
    async def test_uses_default_model_when_no_override(self):
        mock_settings = _make_settings()
        lt_resp = _make_litellm_response("the answer")

        with (
            patch("services.inference.app.backends.api.settings", mock_settings),
            patch("litellm.acompletion", new_callable=AsyncMock, return_value=lt_resp) as mock_acomp,
        ):
            from services.inference.app.backends.api import APIBackend
            backend = APIBackend()
            resp = await backend.generate(_make_request(role="coder"))

        assert resp.content == "the answer"
        assert resp.tokens_generated == 20

    async def test_per_role_override_used(self):
        mock_settings = _make_settings(
            overrides={"coder": "ft:gpt-4o:acme:coding:xyz"},
        )
        lt_resp = _make_litellm_response("coded output")

        with (
            patch("services.inference.app.backends.api.settings", mock_settings),
            patch("litellm.acompletion", new_callable=AsyncMock, return_value=lt_resp) as mock_acomp,
        ):
            from services.inference.app.backends.api import APIBackend
            backend = APIBackend()
            resp = await backend.generate(_make_request(role="coder"))

        called_model = mock_acomp.call_args[1]["model"]
        assert called_model == "ft:gpt-4o:acme:coding:xyz"
        assert resp.content == "coded output"

    async def test_role_without_override_uses_default(self):
        mock_settings = _make_settings(
            api_model="claude-3-5-haiku-20241022",
            overrides={"coder": "ft:gpt-4o:acme:coding:xyz"},
        )
        lt_resp = _make_litellm_response("planned")

        with (
            patch("services.inference.app.backends.api.settings", mock_settings),
            patch("litellm.acompletion", new_callable=AsyncMock, return_value=lt_resp) as mock_acomp,
        ):
            from services.inference.app.backends.api import APIBackend
            backend = APIBackend()
            resp = await backend.generate(_make_request(role="planner"))

        called_model = mock_acomp.call_args[1]["model"]
        assert called_model == "claude-3-5-haiku-20241022"

    async def test_response_fields_populated(self):
        mock_settings = _make_settings()
        lt_resp = _make_litellm_response("output", tokens=42)

        with (
            patch("services.inference.app.backends.api.settings", mock_settings),
            patch("litellm.acompletion", new_callable=AsyncMock, return_value=lt_resp),
        ):
            from services.inference.app.backends.api import APIBackend
            backend = APIBackend()
            resp = await backend.generate(_make_request(role="reviewer"))

        assert isinstance(resp, GenerateResponse)
        assert resp.role == "reviewer"
        assert resp.session_id == "sess-api"
        assert resp.tokens_generated == 42
