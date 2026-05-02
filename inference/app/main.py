"""
Inference Service — entry point
================================
Starts a FastAPI application that exposes:
  POST  /v1/generate         — primary inference endpoint (HTTP, blocking)
  POST  /v1/generate/stream  — token-streaming SSE endpoint (HTTP)
  GET   /health              — health check

An async gRPC server is also started as a background task on
``INFERENCE__GRPC_PORT`` (default 50051), sharing the same backend.
The gRPC service exposes both ``Generate`` (unary) and ``GenerateStream``
(server-side streaming) RPCs.

Run directly:
    uvicorn services.inference.app.main:app --host 0.0.0.0 --port 8010

Or via Docker (see Dockerfile in this directory).
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI

from .config.settings import settings
from .routes.generate import router as generate_router
from .routes.adapters import router as adapters_router

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s | %(name)s | %(message)s",
)

_log = logging.getLogger(__name__)

# Holds the gRPC background task so the shutdown hook can cancel it cleanly.
_grpc_task: Optional[asyncio.Task] = None


def _on_grpc_task_done(task: asyncio.Task) -> None:
    """Callback attached to the gRPC background task."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        _log.error(
            "gRPC server task exited with an exception: %s", exc, exc_info=exc
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _grpc_task

    # Eagerly initialise the default backend so the first real request is not slow.
    from .backends.factory import get_backend
    from shared.contracts.models import load_registry
    get_backend(load_registry().default_backend)

    # Start gRPC server as a background task alongside uvicorn.
    from .grpc.server import serve as grpc_serve
    _grpc_task = asyncio.create_task(grpc_serve())
    _grpc_task.add_done_callback(_on_grpc_task_done)

    _log.info(
        "Inference Service started on HTTP port %d, gRPC port %d",
        settings.port,
        settings.grpc_port,
    )

    yield

    # Gracefully stop the gRPC server before uvicorn exits.
    # stop(grace=5) allows in-flight RPCs up to 5 seconds to finish before
    # the server is forcibly terminated.
    from .grpc.server import shutdown as grpc_shutdown
    await grpc_shutdown()

    if _grpc_task is not None and not _grpc_task.done():
        _grpc_task.cancel()
        try:
            await _grpc_task
        except asyncio.CancelledError:
            pass
        _grpc_task = None


app = FastAPI(
    title="Inference Service",
    description=(
        "Provides LLM generation for the IdleDev multi-agent pipeline. "
        "Calls local vLLM servers for all inference requests."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(generate_router)
app.include_router(adapters_router)
