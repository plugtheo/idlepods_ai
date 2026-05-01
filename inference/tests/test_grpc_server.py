"""
Tests for the gRPC server (inference/app/grpc/server.py).

All tests mock out grpcio + proto stubs so they run without any
compiled protobuf extension modules.

Covers:
- _build_pydantic_request: correct conversion of proto fields
- _build_pydantic_request: HasField() used for optional scalar defaults
- _InferenceServicer.Generate: happy path returns GenerateResponse proto
- _InferenceServicer.Generate: exception calls context.abort() then returns
- _InferenceServicer.GenerateStream: yields token chunks then is_final sentinel
- _InferenceServicer.GenerateStream: exception calls context.abort() then returns
- serve(): graceful shutdown via shutdown() clears _server reference
- serve(): logs warning when _GRPC_AVAILABLE=False and returns without error
"""

import asyncio
import sys
import types
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from shared.contracts.inference import GenerateRequest, GenerateResponse, Message


# ──────────────────────────────────────────────────────────────────────────────
# Stub the grpcio modules so we can test without them installed
# ──────────────────────────────────────────────────────────────────────────────

def _install_grpc_stubs():
    """
    Register minimal fake modules for grpc, grpc.aio, and the generated
    proto stubs into sys.modules so the server module can be imported and
    exercised without a real grpcio installation.
    """
    # grpc top-level module
    grpc_mod = types.ModuleType("grpc")
    grpc_mod.Compression = MagicMock()
    grpc_mod.Compression.Gzip = 2
    grpc_mod.StatusCode = MagicMock()
    grpc_mod.StatusCode.INTERNAL = "INTERNAL"

    # grpc.aio sub-module
    grpc_aio_mod = types.ModuleType("grpc.aio")
    grpc_aio_mod.server = MagicMock()
    grpc_aio_mod.ServicerContext = object  # base class placeholder

    grpc_mod.aio = grpc_aio_mod
    sys.modules.setdefault("grpc", grpc_mod)
    sys.modules.setdefault("grpc.aio", grpc_aio_mod)

    # shared.grpc_stubs package
    stub_pkg = types.ModuleType("shared.grpc_stubs")
    sys.modules.setdefault("shared.grpc_stubs", stub_pkg)

    # inference_pb2
    pb2 = types.ModuleType("shared.grpc_stubs.inference_pb2")

    class _FakeGenerateRequest:
        def __init__(self, **kw):
            self.__dict__.update(kw)
            self._has = set()

        def HasField(self, name):
            return name in self._has

    class _FakeGenerateResponse:
        def __init__(self, content="", tokens_generated=0):
            self.content = content
            self.tokens_generated = tokens_generated

    class _FakeGenerateStreamChunk:
        def __init__(self, token="", is_final=False, tokens_generated=0):
            self.token = token
            self.is_final = is_final
            self.tokens_generated = tokens_generated

    class _FakeMessageProto:
        def __init__(self, role=0, content=""):
            self.role = role
            self.content = content

    pb2.GenerateRequest = _FakeGenerateRequest
    pb2.GenerateResponse = _FakeGenerateResponse
    pb2.GenerateStreamChunk = _FakeGenerateStreamChunk
    pb2.MessageProto = _FakeMessageProto

    sys.modules.setdefault("shared.grpc_stubs.inference_pb2", pb2)
    stub_pkg.inference_pb2 = pb2

    # inference_pb2_grpc
    pb2_grpc = types.ModuleType("shared.grpc_stubs.inference_pb2_grpc")

    class _FakeServiceServicer:
        pass

    pb2_grpc.InferenceServiceServicer = _FakeServiceServicer
    pb2_grpc.add_InferenceServiceServicer_to_server = MagicMock()
    pb2_grpc.InferenceServiceStub = MagicMock()

    sys.modules.setdefault("shared.grpc_stubs.inference_pb2_grpc", pb2_grpc)
    stub_pkg.inference_pb2_grpc = pb2_grpc

    return grpc_mod, grpc_aio_mod, pb2, pb2_grpc


_grpc_mod, _grpc_aio_mod, _pb2, _pb2_grpc = _install_grpc_stubs()


def _make_proto_request(
    model_family="qwen",
    role="coder",
    messages=None,
    session_id="sess-grpc",
    optional_fields=None,
):
    """Build a fake proto GenerateRequest with optional field tracking."""
    req = _pb2.GenerateRequest(
        model_family=model_family,
        role=role,
        messages=messages or [_pb2.MessageProto(role=1, content="write code")],
        session_id=session_id,
        adapter_name=None,
        max_tokens=0,
        temperature=0.0,
        top_p=0.0,
    )
    if optional_fields:
        for f in optional_fields:
            req._has.add(f)
    return req


# ──────────────────────────────────────────────────────────────────────────────
# _build_pydantic_request
# ──────────────────────────────────────────────────────────────────────────────

class TestBuildPydanticRequest:
    def setup_method(self):
        import services.inference.app.grpc.server as srv
        self.srv = srv

    def test_basic_conversion(self):
        proto_req = _make_proto_request()
        pydantic_req = self.srv._build_pydantic_request(proto_req)
        assert pydantic_req.model_family == "qwen"
        assert pydantic_req.role == "coder"
        assert len(pydantic_req.messages) == 1
        assert pydantic_req.messages[0].role == "user"
        assert pydantic_req.messages[0].content == "write code"

    def test_defaults_applied_when_optional_fields_absent(self):
        proto_req = _make_proto_request()
        pydantic_req = self.srv._build_pydantic_request(proto_req)
        assert pydantic_req.max_tokens == 1024
        assert pydantic_req.temperature == 0.2
        assert pydantic_req.top_p == 0.95
        assert pydantic_req.adapter_name is None

    def test_optional_fields_used_when_present(self):
        proto_req = _make_proto_request(
            optional_fields={"max_tokens", "temperature", "top_p", "adapter_name"}
        )
        proto_req.max_tokens = 512
        proto_req.temperature = 0.7
        proto_req.top_p = 0.8
        proto_req.adapter_name = "coding_lora"

        pydantic_req = self.srv._build_pydantic_request(proto_req)
        assert pydantic_req.max_tokens == 512
        assert pydantic_req.temperature == 0.7
        assert pydantic_req.top_p == 0.8
        assert pydantic_req.adapter_name == "coding_lora"

    def test_role_enum_mapped_correctly(self):
        """ROLE_SYSTEM=0, ROLE_USER=1, ROLE_ASSISTANT=2 → string roles."""
        msgs = [
            _pb2.MessageProto(role=0, content="system"),
            _pb2.MessageProto(role=1, content="user"),
            _pb2.MessageProto(role=2, content="assistant"),
        ]
        proto_req = _make_proto_request(messages=msgs)
        pydantic_req = self.srv._build_pydantic_request(proto_req)
        roles = [m.role for m in pydantic_req.messages]
        assert roles == ["system", "user", "assistant"]

    def test_unknown_role_enum_defaults_to_user(self):
        msgs = [_pb2.MessageProto(role=99, content="?")]
        proto_req = _make_proto_request(messages=msgs)
        pydantic_req = self.srv._build_pydantic_request(proto_req)
        assert pydantic_req.messages[0].role == "user"


# ──────────────────────────────────────────────────────────────────────────────
# _InferenceServicer.Generate (unary)
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestInferenceServicerGenerate:
    def _make_servicer(self, backend):
        import services.inference.app.grpc.server as srv
        # Patch get_backend to return our mock
        with patch("services.inference.app.grpc.server.get_backend", return_value=backend):
            # _InferenceServicer is only defined when _GRPC_AVAILABLE; we
            # directly instantiate it from the server module.
            servicer = srv._InferenceServicer()
        return servicer

    async def test_generate_happy_path(self):
        backend = MagicMock()
        backend.generate = AsyncMock(
            return_value=GenerateResponse(
                content="result text",
                model_family="qwen",
                role="coder",
                tokens_generated=10,
                session_id=None,
            )
        )
        import services.inference.app.grpc.server as srv

        context = AsyncMock()
        proto_req = _make_proto_request()

        with patch("services.inference.app.grpc.server.get_backend", return_value=backend):
            servicer = srv._InferenceServicer()
            resp = await servicer.Generate(proto_req, context)

        assert resp.content == "result text"
        assert resp.tokens_generated == 10
        context.abort.assert_not_called()

    async def test_generate_exception_calls_abort_and_returns(self):
        """Exception must call context.abort(INTERNAL, ...) and return — not re-raise."""
        backend = MagicMock()
        backend.generate = AsyncMock(side_effect=RuntimeError("backend crash"))
        import services.inference.app.grpc.server as srv

        context = AsyncMock()
        proto_req = _make_proto_request()

        with patch("services.inference.app.grpc.server.get_backend", return_value=backend):
            servicer = srv._InferenceServicer()
            result = await servicer.Generate(proto_req, context)

        context.abort.assert_awaited_once()
        # StatusCode.INTERNAL is a stub string "INTERNAL" — confirm abort was called with it
        import grpc
        assert context.abort.call_args[0][0] == grpc.StatusCode.INTERNAL
        # Must return (not raise) — so result is None
        assert result is None


# ──────────────────────────────────────────────────────────────────────────────
# _InferenceServicer.GenerateStream (server-side streaming)
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestInferenceServicerGenerateStream:
    async def _collect_chunks(self, servicer, proto_req, context):
        chunks = []
        async for chunk in servicer.GenerateStream(proto_req, context):
            chunks.append(chunk)
        return chunks

    async def test_yields_token_chunks_then_final_sentinel(self):
        async def _gen(req):
            yield "Hello"
            yield " world"

        backend = MagicMock()
        backend.generate_stream = MagicMock(side_effect=_gen)
        import services.inference.app.grpc.server as srv

        context = AsyncMock()
        proto_req = _make_proto_request()

        with patch("services.inference.app.grpc.server.get_backend", return_value=backend):
            servicer = srv._InferenceServicer()
            chunks = await self._collect_chunks(servicer, proto_req, context)

        # Two token chunks + one is_final sentinel
        assert len(chunks) == 3
        assert chunks[0].token == "Hello"
        assert chunks[0].is_final is False
        assert chunks[1].token == " world"
        assert chunks[1].is_final is False
        assert chunks[2].token == ""
        assert chunks[2].is_final is True
        assert chunks[2].tokens_generated == 2

    async def test_final_sentinel_tokens_generated_count(self):
        """tokens_generated on the sentinel equals number of token chunks."""
        async def _gen(req):
            for t in ["a", "b", "c", "d", "e"]:
                yield t

        backend = MagicMock()
        backend.generate_stream = MagicMock(side_effect=_gen)
        import services.inference.app.grpc.server as srv

        context = AsyncMock()
        proto_req = _make_proto_request()

        with patch("services.inference.app.grpc.server.get_backend", return_value=backend):
            servicer = srv._InferenceServicer()
            chunks = await self._collect_chunks(servicer, proto_req, context)

        sentinel = chunks[-1]
        assert sentinel.is_final is True
        assert sentinel.tokens_generated == 5

    async def test_stream_exception_calls_abort_and_returns(self):
        async def _bad_gen(req):
            yield "partial"
            raise RuntimeError("stream blew up")

        backend = MagicMock()
        backend.generate_stream = MagicMock(side_effect=_bad_gen)
        import services.inference.app.grpc.server as srv

        context = AsyncMock()
        proto_req = _make_proto_request()

        with patch("services.inference.app.grpc.server.get_backend", return_value=backend):
            servicer = srv._InferenceServicer()
            chunks = []
            # Should not raise — GenerateStream handles internally
            async for chunk in servicer.GenerateStream(proto_req, context):
                chunks.append(chunk)

        context.abort.assert_awaited_once()
        import grpc
        assert context.abort.call_args[0][0] == grpc.StatusCode.INTERNAL
        # Only the partial chunk was yielded before the error
        assert len(chunks) == 1
        assert chunks[0].token == "partial"


# ──────────────────────────────────────────────────────────────────────────────
# serve() / shutdown()
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestServeAndShutdown:
    async def test_serve_logs_warning_when_grpc_unavailable(self, caplog):
        import services.inference.app.grpc.server as srv
        original = srv._GRPC_AVAILABLE
        srv._GRPC_AVAILABLE = False
        try:
            import logging
            with caplog.at_level(logging.WARNING, logger="services.inference.app.grpc.server"):
                await srv.serve(port=50099)
            assert any("not started" in r.message for r in caplog.records)
        finally:
            srv._GRPC_AVAILABLE = original

    async def test_shutdown_clears_server_reference(self):
        import services.inference.app.grpc.server as srv

        mock_server = AsyncMock()
        srv._server = mock_server

        await srv.shutdown(grace=0.0)

        mock_server.stop.assert_awaited_once_with(0.0)
        assert srv._server is None

    async def test_shutdown_is_noop_when_no_server(self):
        """shutdown() must not raise when called before serve()."""
        import services.inference.app.grpc.server as srv
        srv._server = None
        # Should not raise
        await srv.shutdown(grace=1.0)
