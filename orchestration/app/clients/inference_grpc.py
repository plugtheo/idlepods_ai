"""
Inference Service gRPC client
==============================
Used by the Orchestration Service when ``ORCHESTRATION__INFERENCE_USE_GRPC=true``.
Exposes the same ``async generate(request) → response`` interface as the HTTP
``InferenceClient`` so ``nodes.py`` is transport-agnostic.

Both a blocking ``generate()`` and a streaming ``generate_stream()`` are
provided — the streaming variant maps to the ``GenerateStream`` server-side
streaming RPC and yields token fragments as they arrive.

Requires generated proto stubs at ``shared/grpc_stubs/``.  If stubs or
``grpcio`` are absent, instantiation raises ``RuntimeError`` with an
actionable message.

Payload optimizations applied
------------------------------
- MessageRole enum: 1-byte varint per turn instead of 4–9-byte string.
- Optional sampling params: fields not set are absent on the wire.  We only
  send temperature/top_p/max_tokens/adapter_name when they differ from the
  agreed server defaults (0.2 / 0.95 / 1024 / None).
- No echoed response fields: GenerateResponse carries only content +
  tokens_generated; backend/role/session_id are elided.
- gzip channel compression: applied at channel level for large message bodies
  (system prompts, history context).
"""

from __future__ import annotations

import json
import logging
from typing import AsyncGenerator

from shared.contracts.inference import GenerateRequest, GenerateResponse

logger = logging.getLogger(__name__)

try:
    import grpc
    from grpc import aio as grpc_aio
    from shared.grpc_stubs import inference_pb2, inference_pb2_grpc  # type: ignore[import]
    _GRPC_AVAILABLE = True
except ImportError as _import_err:
    _GRPC_AVAILABLE = False
    _import_err_msg = str(_import_err)


try:
    from shared.grpc_stubs._version import PROTO_SCHEMA_HASH as _PROTO_SCHEMA_HASH
except ImportError:
    _PROTO_SCHEMA_HASH = None

# Map the three legal role strings to the MessageRole enum values.
# Avoids a per-call getattr lookup into the generated module.
_ROLE_TO_ENUM: dict[str, int] = {
    "system":    0,   # ROLE_SYSTEM
    "user":      1,   # ROLE_USER
    "assistant": 2,   # ROLE_ASSISTANT
    "tool":      3,   # ROLE_TOOL
}

# Server-side defaults for optional sampling fields.
# We skip sending a field when the value equals its default: absent on wire.
# IMPORTANT: these must mirror inference/app/config/settings.py grpc_default_* byte-for-byte.
# Cross-process drift (e.g. env override applied to one side only) silently corrupts the
# wire-elision optimisation: the server applies a different default than the client assumed.
_DEFAULT_MAX_TOKENS  = 1024
_DEFAULT_TEMPERATURE = 0.2
_DEFAULT_TOP_P       = 0.95


class GrpcInferenceClient:
    """
    Async gRPC client for the Inference Service.

    One persistent channel is shared across all ``generate()`` and
    ``generate_stream()`` calls — gRPC multiplexes concurrent RPCs over
    HTTP/2 so no connection pool is needed.  gzip compression is enabled at
    channel level to shrink large message bodies (system prompts, history).
    """

    def __init__(self, host: str, port: int) -> None:
        if not _GRPC_AVAILABLE:
            raise RuntimeError(
                f"gRPC inference client unavailable: {_import_err_msg}.  "
                "Run `python scripts/generate_protos.py` and install grpcio."
            )
        options = [
            ("grpc.default_compression_algorithm", grpc.Compression.Gzip),
        ]
        self._channel = grpc_aio.insecure_channel(f"{host}:{port}", options=options)
        self._stub = inference_pb2_grpc.InferenceServiceStub(self._channel)
        self._version_checked = False
        logger.info("gRPC inference client connected to %s:%d (gzip)", host, port)

    def _build_proto_request(
        self, request: GenerateRequest
    ) -> "inference_pb2.GenerateRequest":
        """Convert the shared Pydantic contract to a proto request message."""
        def _build_msg_proto(m):
            kwargs = {"role": _ROLE_TO_ENUM.get(m.role, 1), "content": m.content or ""}
            if m.tool_calls:
                kwargs["tool_calls_json"] = json.dumps(m.tool_calls)
            if m.tool_call_id:
                kwargs["tool_call_id"] = m.tool_call_id
            if m.name:
                kwargs["name"] = m.name
            return inference_pb2.MessageProto(**kwargs)

        proto_req = inference_pb2.GenerateRequest(
            backend=request.backend,
            role=request.role,
            messages=[_build_msg_proto(m) for m in request.messages],
            session_id=request.session_id or "",
        )

        # Only serialize optional fields when they differ from the server default.
        # Proto3 optional = field presence: absent field = zero bytes on wire.
        if request.adapter_name is not None:
            proto_req.adapter_name = request.adapter_name
        if request.max_tokens != _DEFAULT_MAX_TOKENS:
            proto_req.max_tokens = request.max_tokens
        if request.temperature != _DEFAULT_TEMPERATURE:
            proto_req.temperature = request.temperature
        if request.top_p != _DEFAULT_TOP_P:
            proto_req.top_p = request.top_p
        if request.tools:
            proto_req.tools_json = json.dumps(
                [t.model_dump(exclude_none=True) for t in request.tools]
            )

        return proto_req

    async def _verify_proto_version(self) -> None:
        """Assert the server's proto schema hash matches the compiled stubs."""
        if _PROTO_SCHEMA_HASH is None:
            logger.debug("Proto _version.py not found — skipping schema compatibility check.")
            return
        try:
            resp = await self._stub.GetProtoVersion(inference_pb2.ProtoVersionRequest())
            if resp.schema_hash != _PROTO_SCHEMA_HASH:
                raise RuntimeError(
                    f"Proto schema mismatch: server={resp.schema_hash!r} "
                    f"client={_PROTO_SCHEMA_HASH!r}. "
                    "Regenerate stubs with `python scripts/generate_protos.py` and redeploy."
                )
            logger.info("Proto schema verified: %s…", _PROTO_SCHEMA_HASH[:12])
        except RuntimeError:
            raise
        except Exception as exc:
            logger.warning("Proto version check failed: %s — continuing without verification.", exc)

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        """
        Send a GenerateRequest over gRPC and return a GenerateResponse.

        Only non-default sampling params are placed on the wire — absent
        optional fields are zero bytes.  The server applies its own defaults
        when a field is absent.
        """
        if not self._version_checked:
            await self._verify_proto_version()
            self._version_checked = True
        proto_req = self._build_proto_request(request)
        proto_resp = await self._stub.Generate(proto_req)
        tool_calls = json.loads(proto_resp.tool_calls_json) if proto_resp.tool_calls_json else None
        return GenerateResponse(
            content=proto_resp.content,
            backend=request.backend,             # echo from request, not response
            role=request.role,                   # echo from request, not response
            tokens_generated=proto_resp.tokens_generated,
            session_id=request.session_id,
            tool_calls=tool_calls,
        )

    async def generate_stream(
        self, request: GenerateRequest
    ) -> AsyncGenerator[str, None]:
        """
        Call the ``GenerateStream`` server-side streaming RPC and yield each
        token fragment as it arrives.

        The final chunk (``is_final=True``) carries only metadata and its
        ``token`` field is empty — it is silently consumed here.
        """
        proto_req = self._build_proto_request(request)
        async for chunk in self._stub.GenerateStream(proto_req):
            if chunk.is_final:
                break
            if chunk.token:
                yield chunk.token

    async def health(self) -> bool:
        """Returns True if the gRPC channel is open (does not send a real RPC)."""
        try:
            state = self._channel.get_state(try_to_connect=True)
            return state.name not in ("TRANSIENT_FAILURE", "SHUTDOWN")
        except Exception:
            return False

    async def close(self) -> None:
        """Close the underlying gRPC channel."""
        await self._channel.close()

