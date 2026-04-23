"""
Tests for generate_stream() on inference backends.

Covers:
- LocalVLLMBackend.generate_stream(): yields tokens from vLLM SSE
- LocalVLLMBackend.generate_stream(): handles [DONE] sentinel correctly
- LocalVLLMBackend.generate_stream(): unknown family raises ValueError
- LocalVLLMBackend.generate_stream(): skips blank delta.content
- APIBackend.generate_stream(): yields tokens from litellm stream
- APIBackend.generate_stream(): skips None/empty deltas
- InferenceBackend base fallback: default generate_stream yields full content
"""

import sys
import types

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from typing import AsyncIterator

from shared.contracts.inference import GenerateRequest, GenerateResponse, Message

# ── Stub litellm into sys.modules so patch("litellm.acompletion", …) works
# without actually installing the package.
if "litellm" not in sys.modules:
    _litellm_stub = types.ModuleType("litellm")
    _litellm_stub.acompletion = None  # placeholder attribute; overridden per test
    sys.modules["litellm"] = _litellm_stub


def _make_request(family="deepseek", role="coder", adapter=None):
    return GenerateRequest(
        model_family=family,
        role=role,
        messages=[Message(role="user", content="write code")],
        adapter_name=adapter,
        session_id="sess-stream",
    )


# ────────────────────────────────────────────────────────────────────────────
# Helpers for building async line iterators (mocks vLLM SSE transport)
# ────────────────────────────────────────────────────────────────────────────

async def _sse_lines(*lines):
    """Async generator that yields pre-built SSE lines."""
    for line in lines:
        yield line


def _make_streaming_mock_client(lines):
    """
    Build a mock httpx AsyncClient whose .stream() method returns an async
    context manager that yields the given SSE text lines from .aiter_lines().

    The returned object is used as the persistent self._client on
    LocalVLLMBackend — no outer context manager needed.
    """
    response_mock = MagicMock()
    response_mock.raise_for_status = MagicMock()
    response_mock.aiter_lines = MagicMock(return_value=_sse_lines(*lines))

    stream_ctx = AsyncMock()
    stream_ctx.__aenter__ = AsyncMock(return_value=response_mock)
    stream_ctx.__aexit__ = AsyncMock(return_value=False)

    client_mock = MagicMock()
    client_mock.stream = MagicMock(return_value=stream_ctx)

    return client_mock


# ────────────────────────────────────────────────────────────────────────────
# LocalVLLMBackend streaming
# ────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestLocalVLLMBackendStream:

    def _patch_registry(self):
        """Force the adapter registry to report nothing known, TTL never expired."""
        from services.inference.app.backends.local_vllm import _REGISTRIES
        for reg in _REGISTRIES.values():
            reg._known = set()
            reg._fetched_at = float("inf")

    async def test_yields_tokens_from_sse_stream(self):
        """Tokens from vLLM SSE delta.content are yielded in order."""
        from services.inference.app.backends.local_vllm import LocalVLLMBackend
        self._patch_registry()

        sse_lines = [
            'data: {"choices": [{"delta": {"content": "def "}}]}',
            'data: {"choices": [{"delta": {"content": "hello"}}]}',
            'data: {"choices": [{"delta": {"content": "():"}}]}',
            "data: [DONE]",
        ]
        client_mock = _make_streaming_mock_client(sse_lines)

        with patch("services.inference.app.backends.local_vllm.httpx.AsyncClient", return_value=client_mock):
            backend = LocalVLLMBackend()
            tokens = [t async for t in backend.generate_stream(_make_request())]

        assert tokens == ["def ", "hello", "():"]

    async def test_stops_at_done_sentinel(self):
        """Lines after [DONE] are not yielded."""
        from services.inference.app.backends.local_vllm import LocalVLLMBackend
        self._patch_registry()

        sse_lines = [
            'data: {"choices": [{"delta": {"content": "hi"}}]}',
            "data: [DONE]",
            'data: {"choices": [{"delta": {"content": "should not appear"}}]}',
        ]
        client_mock = _make_streaming_mock_client(sse_lines)

        with patch("services.inference.app.backends.local_vllm.httpx.AsyncClient", return_value=client_mock):
            backend = LocalVLLMBackend()
            tokens = [t async for t in backend.generate_stream(_make_request())]

        assert "should not appear" not in tokens
        assert tokens == ["hi"]

    async def test_skips_empty_delta_content(self):
        """Chunks with empty or missing delta.content are not yielded."""
        from services.inference.app.backends.local_vllm import LocalVLLMBackend
        self._patch_registry()

        sse_lines = [
            'data: {"choices": [{"delta": {}}]}',                        # no content key
            'data: {"choices": [{"delta": {"content": ""}}]}',           # empty string
            'data: {"choices": [{"delta": {"content": "token"}}]}',      # real token
            "data: [DONE]",
        ]
        client_mock = _make_streaming_mock_client(sse_lines)

        with patch("services.inference.app.backends.local_vllm.httpx.AsyncClient", return_value=client_mock):
            backend = LocalVLLMBackend()
            tokens = [t async for t in backend.generate_stream(_make_request())]

        assert tokens == ["token"]

    async def test_skips_non_data_lines(self):
        """Lines that don't start with 'data: ' are ignored."""
        from services.inference.app.backends.local_vllm import LocalVLLMBackend
        self._patch_registry()

        sse_lines = [
            ": keep-alive",                                              # SSE comment
            "",                                                          # blank line
            'data: {"choices": [{"delta": {"content": "ok"}}]}',
            "data: [DONE]",
        ]
        client_mock = _make_streaming_mock_client(sse_lines)

        with patch("services.inference.app.backends.local_vllm.httpx.AsyncClient", return_value=client_mock):
            backend = LocalVLLMBackend()
            tokens = [t async for t in backend.generate_stream(_make_request())]

        assert tokens == ["ok"]

    async def test_unknown_family_raises_value_error(self):
        from services.inference.app.backends.local_vllm import LocalVLLMBackend

        backend = LocalVLLMBackend()
        with pytest.raises(ValueError, match="Unknown model_family"):
            async for _ in backend.generate_stream(_make_request(family="llama")):
                pass

    async def test_stream_request_has_stream_true(self):
        """The payload sent to vLLM must include stream=True."""
        from services.inference.app.backends.local_vllm import LocalVLLMBackend
        self._patch_registry()

        sse_lines = ["data: [DONE]"]
        client_mock = _make_streaming_mock_client(sse_lines)

        with patch("services.inference.app.backends.local_vllm.httpx.AsyncClient", return_value=client_mock):
            backend = LocalVLLMBackend()
            _ = [t async for t in backend.generate_stream(_make_request())]

        # Retrieve the stream() call args from the persistent client mock
        stream_call_kwargs = client_mock.stream.call_args
        payload = stream_call_kwargs[1]["json"]
        assert payload["stream"] is True


# ────────────────────────────────────────────────────────────────────────────
# APIBackend streaming
# ────────────────────────────────────────────────────────────────────────────

def _make_litellm_settings():
    s = MagicMock()
    s.api_model = "claude-3-5-haiku-20241022"
    s.role_model_overrides = {}
    s.api_key = "test-key"
    s.api_provider = "anthropic"
    return s


async def _async_litellm_chunks(*tokens):
    """Async generator yielding fake litellm stream chunks."""
    for token in tokens:
        choice = MagicMock()
        choice.delta.content = token
        chunk = MagicMock()
        chunk.choices = [choice]
        yield chunk


@pytest.mark.asyncio
class TestAPIBackendStream:

    async def test_yields_tokens_from_litellm_stream(self):
        """Tokens from litellm acompletion stream are yielded in order."""
        mock_settings = _make_litellm_settings()

        async_gen = _async_litellm_chunks("Hello", " world", "!")

        with (
            patch("services.inference.app.backends.api.settings", mock_settings),
            patch("litellm.acompletion", new=AsyncMock(return_value=async_gen)),
        ):
            from services.inference.app.backends.api import APIBackend
            backend = APIBackend()
            tokens = [t async for t in backend.generate_stream(_make_request())]

        assert tokens == ["Hello", " world", "!"]

    async def test_skips_none_delta_content(self):
        """Chunks with None delta.content are not yielded."""
        mock_settings = _make_litellm_settings()

        async_gen = _async_litellm_chunks(None, "real", None, " token")

        with (
            patch("services.inference.app.backends.api.settings", mock_settings),
            patch("litellm.acompletion", new=AsyncMock(return_value=async_gen)),
        ):
            from services.inference.app.backends.api import APIBackend
            backend = APIBackend()
            tokens = [t async for t in backend.generate_stream(_make_request())]

        assert tokens == ["real", " token"]

    async def test_skips_empty_string_delta(self):
        """Chunks with empty string delta.content are not yielded."""
        mock_settings = _make_litellm_settings()

        async_gen = _async_litellm_chunks("", "content", "")

        with (
            patch("services.inference.app.backends.api.settings", mock_settings),
            patch("litellm.acompletion", new=AsyncMock(return_value=async_gen)),
        ):
            from services.inference.app.backends.api import APIBackend
            backend = APIBackend()
            tokens = [t async for t in backend.generate_stream(_make_request())]

        assert tokens == ["content"]

    async def test_stream_true_passed_to_litellm(self):
        """stream=True must be included in the litellm call."""
        mock_settings = _make_litellm_settings()

        async_gen = _async_litellm_chunks("x")
        mock_acomp = AsyncMock(return_value=async_gen)

        with (
            patch("services.inference.app.backends.api.settings", mock_settings),
            patch("litellm.acompletion", new=mock_acomp),
        ):
            from services.inference.app.backends.api import APIBackend
            backend = APIBackend()
            _ = [t async for t in backend.generate_stream(_make_request())]

        assert mock_acomp.call_args[1]["stream"] is True

    async def test_role_override_applied_in_stream(self):
        """Per-role model override is also used for streaming calls."""
        mock_settings = _make_litellm_settings()
        mock_settings.role_model_overrides = {"coder": "ft:gpt-4o:org:coder:v1"}

        async_gen = _async_litellm_chunks("code")
        mock_acomp = AsyncMock(return_value=async_gen)

        with (
            patch("services.inference.app.backends.api.settings", mock_settings),
            patch("litellm.acompletion", new=mock_acomp),
        ):
            from services.inference.app.backends.api import APIBackend
            backend = APIBackend()
            _ = [t async for t in backend.generate_stream(_make_request(role="coder"))]

        assert mock_acomp.call_args[1]["model"] == "ft:gpt-4o:org:coder:v1"


# ────────────────────────────────────────────────────────────────────────────
# Base backend fallback
# ────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestBaseBackendStreamFallback:
    """The default generate_stream() buffers via generate() and yields full content."""

    async def test_fallback_yields_full_content_as_single_chunk(self):
        from services.inference.app.backends.base import InferenceBackend

        class _ConcreteBackend(InferenceBackend):
            async def generate(self, request):
                return GenerateResponse(
                    content="full response text",
                    model_family="deepseek",
                    role="coder",
                    tokens_generated=3,
                    session_id=None,
                )

        backend = _ConcreteBackend()
        chunks = [t async for t in backend.generate_stream(_make_request())]
        assert chunks == ["full response text"]

    async def test_fallback_yields_exactly_one_chunk(self):
        from services.inference.app.backends.base import InferenceBackend

        class _ConcreteBackend(InferenceBackend):
            async def generate(self, request):
                return GenerateResponse(
                    content="abc def ghi",
                    model_family="deepseek",
                    role="coder",
                    tokens_generated=3,
                    session_id=None,
                )

        backend = _ConcreteBackend()
        chunks = [t async for t in backend.generate_stream(_make_request())]
        assert len(chunks) == 1
