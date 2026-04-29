"""
Inference Service — gRPC server
================================
Runs an async gRPC server alongside the FastAPI/uvicorn HTTP server.
Both transports share the same backend logic via
``services.inference.app.backends.factory.get_backend(model_family)``.

Port: configurable via ``INFERENCE__GRPC_PORT`` (default 50051).

The server requires generated proto stubs at ``shared/grpc_stubs/``.
Generate them with::

    python scripts/generate_protos.py

If stubs are not available, ``serve()`` logs a warning and returns without
starting the server — the HTTP endpoint continues to operate normally.
"""

from __future__ import annotations

import logging

from ..config.settings import settings

logger = logging.getLogger(__name__)

try:
    import grpc
    from grpc import aio as grpc_aio
    from shared.grpc_stubs import inference_pb2, inference_pb2_grpc  # type: ignore[import]
    _GRPC_AVAILABLE = True
except ImportError:
    _GRPC_AVAILABLE = False

try:
    from grpc_health.v1 import health as _health_module
    from grpc_health.v1 import health_pb2, health_pb2_grpc  # type: ignore[import]
    _HEALTH_AVAILABLE = True
except ImportError:
    _HEALTH_AVAILABLE = False


# Map MessageRole enum integers to the role strings the pydantic layer expects.
_ENUM_TO_ROLE: dict[int, str] = {
    0: "system",
    1: "user",
    2: "assistant",
}

# Server-side defaults for optional sampling fields — read from InferenceSettings
# so a single env-var change (INFERENCE__GRPC_DEFAULT_MAX_TOKENS etc.) updates
# behaviour without touching source. Applied via HasField() presence check.
def _default_max_tokens()  -> int:   return settings.grpc_default_max_tokens
def _default_temperature() -> float: return settings.grpc_default_temperature
def _default_top_p()       -> float: return settings.grpc_default_top_p

# Module-level server reference — set by serve(), cleared by shutdown().
_server: "grpc_aio.Server | None" = None


def _build_pydantic_request(request: "inference_pb2.GenerateRequest"):  # type: ignore[name-defined]
    """Convert a proto GenerateRequest to the shared Pydantic contract."""
    from shared.contracts.inference import GenerateRequest, Message

    messages = [
        Message(
            role=_ENUM_TO_ROLE.get(m.role, "user"),
            content=m.content,
        )
        for m in request.messages
    ]

    max_tokens   = request.max_tokens  if request.HasField("max_tokens")  else _default_max_tokens()
    temperature  = request.temperature if request.HasField("temperature") else _default_temperature()
    top_p        = request.top_p       if request.HasField("top_p")       else _default_top_p()
    adapter_name = request.adapter_name if request.HasField("adapter_name") else None

    return GenerateRequest(
        model_family=request.model_family,
        role=request.role,
        messages=messages,
        adapter_name=adapter_name,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        session_id=request.session_id or None,
    )


if _GRPC_AVAILABLE:
    from ..backends.factory import get_backend

    class _InferenceServicer(inference_pb2_grpc.InferenceServiceServicer):  # type: ignore[misc]
        """Maps gRPC requests → InferenceBackend → gRPC responses."""

        async def Generate(
            self,
            request: "inference_pb2.GenerateRequest",
            context: "grpc_aio.ServicerContext",
        ) -> "inference_pb2.GenerateResponse":
            try:
                pydantic_req = _build_pydantic_request(request)
                backend = get_backend(pydantic_req.model_family)
                response = await backend.generate(pydantic_req)
                return inference_pb2.GenerateResponse(
                    content=response.content,
                    tokens_generated=response.tokens_generated,
                )
            except Exception as exc:
                logger.error("gRPC Generate error: %s", exc, exc_info=True)
                await context.abort(grpc.StatusCode.INTERNAL, str(exc))
                return  # abort() sets status but does not raise; explicit return needed

        async def GenerateStream(
            self,
            request: "inference_pb2.GenerateRequest",
            context: "grpc_aio.ServicerContext",
        ):
            """
            Server-side streaming RPC — yields token fragments as the backend
            produces them.  The final chunk has ``is_final=True`` and carries
            the total ``tokens_generated`` count; its ``token`` field is empty.
            """
            try:
                pydantic_req = _build_pydantic_request(request)
                backend = get_backend(pydantic_req.model_family)
                token_count = 0
                async for token in backend.generate_stream(pydantic_req):
                    token_count += 1
                    yield inference_pb2.GenerateStreamChunk(
                        token=token,
                        is_final=False,
                        tokens_generated=0,
                    )
                # Final sentinel chunk
                yield inference_pb2.GenerateStreamChunk(
                    token="",
                    is_final=True,
                    tokens_generated=token_count,
                )
            except Exception as exc:
                logger.error("gRPC GenerateStream error: %s", exc, exc_info=True)
                await context.abort(grpc.StatusCode.INTERNAL, str(exc))
                return


async def serve(port: int = 50051) -> None:
    """
    Start the gRPC server and block until it terminates.

    Designed to run as a background asyncio task alongside uvicorn::

        task = asyncio.create_task(serve(port=settings.grpc_port))
        task.add_done_callback(...)   # log unexpected exits
    """
    global _server

    if not _GRPC_AVAILABLE:
        logger.warning(
            "grpcio or proto stubs not available — gRPC server not started.  "
            "Run `python scripts/generate_protos.py` and install grpcio."
        )
        return

    _server = grpc_aio.server(compression=grpc.Compression.Gzip)
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(
        _InferenceServicer(), _server
    )

    # Register the standard gRPC health service so infrastructure health
    # checks (Kubernetes, grpc_health_probe, etc.) can probe port 50051
    # independently of the HTTP /health endpoint.
    if _HEALTH_AVAILABLE:
        health_servicer = _health_module.HealthServicer()
        health_pb2_grpc.add_HealthServicer_to_server(health_servicer, _server)
        health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
        health_servicer.set(
            "inference.v1.InferenceService",
            health_pb2.HealthCheckResponse.SERVING,
        )
    else:
        logger.info(
            "grpcio-health-checking not installed — gRPC health service unavailable.  "
            "Install with: pip install grpcio-health-checking"
        )

    _server.add_insecure_port(f"[::]:{port}")
    await _server.start()
    logger.info("gRPC Inference server listening on port %d (gzip)", port)
    await _server.wait_for_termination()


async def shutdown(grace: float = 5.0) -> None:
    """
    Gracefully stop the gRPC server, waiting up to *grace* seconds for
    in-flight RPCs to complete before hard-killing them.

    Call from the FastAPI shutdown hook so long-running ``Generate`` /
    ``GenerateStream`` calls finish cleanly rather than being hard-killed
    when the container stops.
    """
    global _server
    if _server is not None:
        logger.info("Stopping gRPC server (grace=%.1fs)…", grace)
        await _server.stop(grace)
        _server = None
        logger.info("gRPC server stopped.")
