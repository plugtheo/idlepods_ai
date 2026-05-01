"""
Tests for generate_stream() on inference backends.

Covers:
- LocalVLLMBackend.generate_stream(): yields tokens from vLLM SSE
- LocalVLLMBackend.generate_stream(): handles [DONE] sentinel correctly
- LocalVLLMBackend.generate_stream(): skips blank delta.content
- LocalVLLMBackend.generate_stream(): skips non-data lines
- LocalVLLMBackend.generate_stream(): payload includes stream=True
- InferenceBackend base fallback: default generate_stream yields full content
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from shared.contracts.inference import GenerateRequest, GenerateResponse, Message

_QWEN_URL = "http://vllm-qwen:8000"
_QWEN_MODEL = "Qwen/Qwen3-14B"


def _make_request(family="qwen", role="coder", adapter=None):
    return GenerateRequest(
        model_family=family,
        role=role,
        messages=[Message(role="user", content="write code")],
        adapter_name=adapter,
        session_id="sess-stream",
    )


# ─── SSE transport helpers ────────────────────────────────────────────────────

async def _sse_lines(*lines):
    for line in lines:
        yield line


def _make_streaming_mock_client(lines):
    """
    Build a mock httpx AsyncClient whose .stream() method returns an async
    context manager that yields the given SSE text lines from .aiter_lines().
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


def _make_backend(client_mock):
    """Return a LocalVLLMBackend with the httpx client replaced by client_mock."""
    from services.inference.app.backends.local_vllm import LocalVLLMBackend
    backend = LocalVLLMBackend("qwen", _QWEN_URL, _QWEN_MODEL)
    backend._client = client_mock
    backend._registry._client = client_mock
    backend._registry._known = set()
    backend._registry._fetched_at = float("inf")  # prevent HTTP refresh
    return backend


# ─── LocalVLLMBackend streaming ───────────────────────────────────────────────

@pytest.mark.asyncio
class TestLocalVLLMBackendStream:

    async def test_yields_tokens_from_sse_stream(self):
        sse_lines = [
            'data: {"choices": [{"delta": {"content": "def "}}]}',
            'data: {"choices": [{"delta": {"content": "hello"}}]}',
            'data: {"choices": [{"delta": {"content": "():"}}]}',
            "data: [DONE]",
        ]
        backend = _make_backend(_make_streaming_mock_client(sse_lines))
        tokens = [t async for t in backend.generate_stream(_make_request())]
        assert tokens == ["def ", "hello", "():"]

    async def test_stops_at_done_sentinel(self):
        sse_lines = [
            'data: {"choices": [{"delta": {"content": "hi"}}]}',
            "data: [DONE]",
            'data: {"choices": [{"delta": {"content": "should not appear"}}]}',
        ]
        backend = _make_backend(_make_streaming_mock_client(sse_lines))
        tokens = [t async for t in backend.generate_stream(_make_request())]
        assert tokens == ["hi"]

    async def test_skips_empty_delta_content(self):
        sse_lines = [
            'data: {"choices": [{"delta": {}}]}',
            'data: {"choices": [{"delta": {"content": ""}}]}',
            'data: {"choices": [{"delta": {"content": "token"}}]}',
            "data: [DONE]",
        ]
        backend = _make_backend(_make_streaming_mock_client(sse_lines))
        tokens = [t async for t in backend.generate_stream(_make_request())]
        assert tokens == ["token"]

    async def test_skips_non_data_lines(self):
        sse_lines = [
            ": keep-alive",
            "",
            'data: {"choices": [{"delta": {"content": "ok"}}]}',
            "data: [DONE]",
        ]
        backend = _make_backend(_make_streaming_mock_client(sse_lines))
        tokens = [t async for t in backend.generate_stream(_make_request())]
        assert tokens == ["ok"]

    async def test_stream_request_has_stream_true(self):
        """The payload sent to vLLM must include stream=True."""
        client_mock = _make_streaming_mock_client(["data: [DONE]"])
        backend = _make_backend(client_mock)
        _ = [t async for t in backend.generate_stream(_make_request())]

        stream_call_kwargs = client_mock.stream.call_args
        payload = stream_call_kwargs[1]["json"]
        assert payload["stream"] is True


# ─── Base backend fallback ────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestBaseBackendStreamFallback:
    """The default generate_stream() buffers via generate() and yields full content."""

    async def test_fallback_yields_full_content_as_single_chunk(self):
        from services.inference.app.backends.base import InferenceBackend

        class _ConcreteBackend(InferenceBackend):
            async def generate(self, request):
                return GenerateResponse(
                    content="full response text",
                    model_family="qwen",
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
                    model_family="qwen",
                    role="coder",
                    tokens_generated=3,
                    session_id=None,
                )

        backend = _ConcreteBackend()
        chunks = [t async for t in backend.generate_stream(_make_request())]
        assert len(chunks) == 1
