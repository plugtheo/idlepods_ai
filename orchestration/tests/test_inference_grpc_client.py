"""
Tests for the orchestration-side gRPC inference client
(orchestration/app/clients/inference_grpc.py).

We stub grpcio and the proto stubs into sys.modules before importing the client
so these tests run without real grpcio installed.

Covers
------
- GrpcInferenceClient._build_proto_request: default sampling fields omitted
- GrpcInferenceClient._build_proto_request: non-default fields included
- GrpcInferenceClient._build_proto_request: role enum mapping
- GrpcInferenceClient._build_proto_request: unknown role defaults to ROLE_USER
- GrpcInferenceClient.generate_stream: yields tokens, stops at is_final sentinel
- GrpcInferenceClient.generate_stream: empty-token chunks are skipped
- GrpcInferenceClient.generate: returns GenerateResponse with echoed fields
- GrpcInferenceClient.health: returns False on TRANSIENT_FAILURE state
- GrpcInferenceClient.health: returns True on READY state
"""

import sys
import types
import pytest
from unittest.mock import AsyncMock, MagicMock

from shared.contracts.inference import GenerateRequest, Message


# ─── Stub grpc / proto modules into sys.modules ───────────────────────────────

def _ensure_grpc_stubs():
    """Install minimal fakes for grpc and the proto stubs if not already done."""
    # grpc top-level
    grpc_mod = sys.modules.setdefault("grpc", types.ModuleType("grpc"))
    # Only overwrite if not already set by another test module (e.g. test_grpc_server.py)
    if not hasattr(grpc_mod, "Compression") or not hasattr(grpc_mod.Compression, "Gzip"):
        grpc_mod.Compression = MagicMock()
        grpc_mod.Compression.Gzip = 2
    if not hasattr(grpc_mod, "StatusCode") or not hasattr(grpc_mod.StatusCode, "INTERNAL"):
        grpc_mod.StatusCode = MagicMock()
        grpc_mod.StatusCode.INTERNAL = "INTERNAL"  # used by server abort() calls

    # grpc.aio
    grpc_aio_mod = sys.modules.setdefault("grpc.aio", types.ModuleType("grpc.aio"))
    if not hasattr(grpc_aio_mod, "insecure_channel"):
        grpc_aio_mod.insecure_channel = MagicMock()
    grpc_mod.aio = grpc_aio_mod

    # shared.grpc_stubs package
    stub_pkg = sys.modules.setdefault("shared.grpc_stubs", types.ModuleType("shared.grpc_stubs"))

    # inference_pb2
    pb2 = sys.modules.setdefault(
        "shared.grpc_stubs.inference_pb2", types.ModuleType("shared.grpc_stubs.inference_pb2")
    )

    class _ProtoRequest:
        def __init__(self, *, backend="", role="", messages=None, session_id="",
                     adapter_name=None, max_tokens=None, temperature=None, top_p=None):
            self.backend = backend
            self.role = role
            self.messages = messages or []
            self.session_id = session_id
            self.adapter_name = adapter_name
            self.max_tokens = max_tokens
            self.temperature = temperature
            self.top_p = top_p

        def HasField(self, name: str) -> bool:
            """Proto3 optional field presence check — field is present when not None."""
            return getattr(self, name, None) is not None

    class _ProtoMessage:
        def __init__(self, role=1, content=""):
            self.role = role
            self.content = content

    class _FakeChunk:
        def __init__(self, token="", is_final=False, tokens_generated=0):
            self.token = token
            self.is_final = is_final
            self.tokens_generated = tokens_generated

    class _FakeResponse:
        def __init__(self, content="", tokens_generated=0):
            self.content = content
            self.tokens_generated = tokens_generated

    # Only overwrite class attributes if the module was freshly created (no GenerateRequest yet)
    # This prevents clobbering stubs already installed by test_grpc_server.py which include HasField.
    if not hasattr(pb2, "GenerateRequest"):
        pb2.GenerateRequest = _ProtoRequest
    if not hasattr(pb2, "MessageProto"):
        pb2.MessageProto = _ProtoMessage
    if not hasattr(pb2, "GenerateStreamChunk"):
        pb2.GenerateStreamChunk = _FakeChunk
    if not hasattr(pb2, "GenerateResponse"):
        pb2.GenerateResponse = _FakeResponse
    stub_pkg.inference_pb2 = pb2

    # inference_pb2_grpc
    pb2_grpc = sys.modules.setdefault(
        "shared.grpc_stubs.inference_pb2_grpc",
        types.ModuleType("shared.grpc_stubs.inference_pb2_grpc"),
    )
    if not hasattr(pb2_grpc, "InferenceServiceStub"):
        pb2_grpc.InferenceServiceStub = MagicMock(return_value=MagicMock())
    if not hasattr(pb2_grpc, "add_InferenceServiceServicer_to_server"):
        pb2_grpc.add_InferenceServiceServicer_to_server = MagicMock()
    stub_pkg.inference_pb2_grpc = pb2_grpc

    return grpc_mod, grpc_aio_mod, pb2, pb2_grpc


_grpc_mod, _grpc_aio_mod, _pb2, _pb2_grpc = _ensure_grpc_stubs()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_client() -> "GrpcInferenceClient":
    """Return a GrpcInferenceClient with a mocked channel and stub."""
    from services.orchestration.app.clients.inference_grpc import GrpcInferenceClient  # type: ignore

    client = object.__new__(GrpcInferenceClient)  # skip __init__ safety checks
    client._channel = MagicMock()
    client._stub = MagicMock()
    return client


def _make_request(
    backend="primary",
    role="coder",
    messages=None,
    max_tokens=1024,
    temperature=0.2,
    top_p=0.95,
    adapter_name=None,
    session_id="sess-1",
) -> GenerateRequest:
    if messages is None:
        messages = [Message(role="user", content="write a function")]
    return GenerateRequest(
        backend=backend,
        role=role,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        adapter_name=adapter_name,
        session_id=session_id,
    )


# ─── _build_proto_request ──────────────────────────────────────────────────

class TestBuildProtoRequest:
    def test_default_fields_omitted_from_wire(self):
        """
        When sampling params equal server defaults, the proto fields should be
        absent (not set) on the proto object — zero bytes on the wire.
        """
        client = _make_client()
        req = _make_request()  # defaults: max_tokens=1024, temp=0.2, top_p=0.95
        proto = client._build_proto_request(req)
        # Field not set at all (absent on wire) OR set to None — both acceptable
        assert getattr(proto, "max_tokens", None) is None
        assert getattr(proto, "temperature", None) is None
        assert getattr(proto, "top_p", None) is None
        assert getattr(proto, "adapter_name", None) is None

    def test_non_default_fields_included(self):
        client = _make_client()
        req = _make_request(max_tokens=512, temperature=0.7, top_p=0.8)
        proto = client._build_proto_request(req)
        assert proto.max_tokens == 512
        assert proto.temperature == 0.7
        assert proto.top_p == 0.8

    def test_adapter_name_included_when_set(self):
        client = _make_client()
        req = _make_request(adapter_name="coding_lora")
        proto = client._build_proto_request(req)
        assert proto.adapter_name == "coding_lora"

    def test_role_enum_system_maps_to_0(self):
        client = _make_client()
        req = _make_request(messages=[Message(role="system", content="sys")])
        proto = client._build_proto_request(req)
        assert proto.messages[0].role == 0

    def test_role_enum_user_maps_to_1(self):
        client = _make_client()
        req = _make_request(messages=[Message(role="user", content="u")])
        proto = client._build_proto_request(req)
        assert proto.messages[0].role == 1

    def test_role_enum_assistant_maps_to_2(self):
        client = _make_client()
        req = _make_request(messages=[Message(role="assistant", content="a")])
        proto = client._build_proto_request(req)
        assert proto.messages[0].role == 2

    def test_unknown_role_defaults_to_user_enum(self):
        client = _make_client()
        req = _make_request(messages=[Message(role="unknown_role", content="?")])
        proto = client._build_proto_request(req)
        assert proto.messages[0].role == 1  # ROLE_USER

    def test_backend_and_role_strings_passed_through(self):
        client = _make_client()
        req = _make_request(backend="primary", role="planner")
        proto = client._build_proto_request(req)
        assert proto.backend == "primary"
        assert proto.role == "planner"

    def test_session_id_empty_string_when_none(self):
        client = _make_client()
        req = _make_request(session_id=None)
        proto = client._build_proto_request(req)
        assert proto.session_id == ""


# ─── generate_stream ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestGrpcClientGenerateStream:
    async def _collect_tokens(self, client, request):
        tokens = []
        async for token in client.generate_stream(request):
            tokens.append(token)
        return tokens

    def _make_chunk_stream(self, *chunks):
        """Return an object that is async-iterable over fake chunks."""
        class _AsyncIter:
            def __aiter__(self_inner):
                return self_inner
            async def __anext__(self_inner):
                if not self_inner._chunks:
                    raise StopAsyncIteration
                return self_inner._chunks.pop(0)

        it = _AsyncIter()
        it._chunks = list(chunks)
        return it

    async def test_yields_tokens_stops_at_is_final(self):
        client = _make_client()
        chunks = self._make_chunk_stream(
            _pb2.GenerateStreamChunk(token="Hello", is_final=False),
            _pb2.GenerateStreamChunk(token=" world", is_final=False),
            _pb2.GenerateStreamChunk(token="", is_final=True, tokens_generated=2),
        )
        client._stub.GenerateStream = MagicMock(return_value=chunks)

        tokens = await self._collect_tokens(client, _make_request())
        assert tokens == ["Hello", " world"]

    async def test_empty_token_chunks_are_skipped(self):
        """Empty-string tokens before the sentinel are not yielded."""
        client = _make_client()
        chunks = self._make_chunk_stream(
            _pb2.GenerateStreamChunk(token="A", is_final=False),
            _pb2.GenerateStreamChunk(token="", is_final=False),  # empty mid-stream
            _pb2.GenerateStreamChunk(token="B", is_final=False),
            _pb2.GenerateStreamChunk(token="", is_final=True),
        )
        client._stub.GenerateStream = MagicMock(return_value=chunks)

        tokens = await self._collect_tokens(client, _make_request())
        assert tokens == ["A", "B"]

    async def test_only_sentinel_yields_nothing(self):
        """A stream that immediately sends is_final yields no tokens."""
        client = _make_client()
        chunks = self._make_chunk_stream(
            _pb2.GenerateStreamChunk(token="", is_final=True),
        )
        client._stub.GenerateStream = MagicMock(return_value=chunks)

        tokens = await self._collect_tokens(client, _make_request())
        assert tokens == []

    async def test_build_proto_request_called_with_correct_fields(self):
        """generate_stream must pass the built proto request to the stub."""
        client = _make_client()
        captured = {}

        def _stub_stream(proto_req):
            captured["req"] = proto_req
            return self._make_chunk_stream(
                _pb2.GenerateStreamChunk(token="t", is_final=False),
                _pb2.GenerateStreamChunk(token="", is_final=True),
            )

        client._stub.GenerateStream = _stub_stream
        pydantic_req = _make_request(backend="primary", role="planner")

        await self._collect_tokens(client, pydantic_req)

        assert captured["req"].backend == "primary"
        assert captured["req"].role == "planner"


# ─── generate (unary) ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestGrpcClientGenerate:
    async def test_returns_generate_response_with_echoed_fields(self):
        client = _make_client()
        proto_resp = _pb2.GenerateResponse(content="result", tokens_generated=5)
        client._stub.Generate = AsyncMock(return_value=proto_resp)

        req = _make_request(backend="primary", role="coder", session_id="sess-x")
        client._version_checked = False
        response = await client.generate(req)

        assert response.content == "result"
        assert response.tokens_generated == 5
        assert response.backend == "primary"   # echoed from request
        assert response.role == "coder"          # echoed from request
        assert response.session_id == "sess-x"  # echoed from request


# ─── health ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestGrpcClientHealth:
    async def test_health_returns_false_on_transient_failure(self):
        client = _make_client()
        state_mock = MagicMock()
        state_mock.name = "TRANSIENT_FAILURE"
        client._channel.get_state = MagicMock(return_value=state_mock)

        from services.orchestration.app.clients.inference_grpc import GrpcInferenceClient
        result = await GrpcInferenceClient.health(client)
        assert result is False

    async def test_health_returns_false_on_shutdown(self):
        client = _make_client()
        state_mock = MagicMock()
        state_mock.name = "SHUTDOWN"
        client._channel.get_state = MagicMock(return_value=state_mock)

        from services.orchestration.app.clients.inference_grpc import GrpcInferenceClient
        result = await GrpcInferenceClient.health(client)
        assert result is False

    async def test_health_returns_true_on_ready(self):
        client = _make_client()
        state_mock = MagicMock()
        state_mock.name = "READY"
        client._channel.get_state = MagicMock(return_value=state_mock)

        from services.orchestration.app.clients.inference_grpc import GrpcInferenceClient
        result = await GrpcInferenceClient.health(client)
        assert result is True

    async def test_health_returns_false_on_exception(self):
        client = _make_client()
        client._channel.get_state = MagicMock(side_effect=RuntimeError("network error"))

        from services.orchestration.app.clients.inference_grpc import GrpcInferenceClient
        result = await GrpcInferenceClient.health(client)
        assert result is False
