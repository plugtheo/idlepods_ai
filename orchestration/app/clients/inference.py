"""
Inference Service HTTP client
==============================
Used by graph node functions to call the Inference Service.

The client is a module-level singleton created on first access.
All calls are async — use `await get_inference_client().generate(request)`.
For streaming use `async for token in get_inference_client().generate_stream(request)`.

Connection pooling
------------------
`InferenceClient` holds a persistent `httpx.AsyncClient` for the lifetime
of the process.  This eliminates per-call TCP setup/teardown overhead that
accumulated to 2–4 extra round-trips per user request when a new client
was created inside `generate()`.

gRPC routing
------------
When `ORCHESTRATION__INFERENCE_USE_GRPC=true`, `get_inference_client()`
returns a `GrpcInferenceClient` instead.  Both expose the same
`async generate(request) → response` and
`async generate_stream(request) → AsyncIterator[str]` interfaces so node
code is unchanged.
"""

from __future__ import annotations

import json
import logging
from typing import AsyncGenerator

import httpx

from shared.contracts.inference import GenerateRequest, GenerateResponse
from ..config.settings import settings

logger = logging.getLogger(__name__)


class InferenceClient:
    """Async HTTP client for the Inference Service — persistent connection pool."""

    def __init__(self, base_url: str, timeout: float) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(
                max_connections=settings.http_max_connections,
                max_keepalive_connections=settings.http_max_keepalive_connections,
            ),
        )

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Send a GenerateRequest and return the GenerateResponse."""
        url = f"{self._base_url}/v1/generate"
        resp = await self._client.post(url, json=request.model_dump(mode="json"))
        resp.raise_for_status()
        return GenerateResponse.model_validate(resp.json())

    async def generate_stream(
        self, request: GenerateRequest
    ) -> AsyncGenerator[str, None]:
        """
        Call ``POST /v1/generate/stream`` and yield token fragments via SSE.

        Each SSE line is ``data: {"token": "...", "is_final": false}``.
        The final ``{"is_final": true}`` sentinel ends the stream.
        Errors are reported as ``{"error": "..."}`` — raised as RuntimeError.
        """
        url = f"{self._base_url}/v1/generate/stream"
        async with self._client.stream(
            "POST", url, json=request.model_dump(mode="json")
        ) as resp:
            resp.raise_for_status()
            async for raw_line in resp.aiter_lines():
                if not raw_line.startswith("data: "):
                    continue
                try:
                    payload = json.loads(raw_line[6:])
                except json.JSONDecodeError:
                    continue
                if "error" in payload:
                    raise RuntimeError(
                        f"Inference stream error: {payload['error']}"
                    )
                if payload.get("is_final"):
                    return
                token: str = payload.get("token", "")
                if token:
                    yield token

    async def get_model_info(self) -> dict:
        """GET /v1/model-info → {"deepseek": <max_model_len>, "mistral": <max_model_len>}."""
        resp = await self._client.get(f"{self._base_url}/v1/model-info")
        resp.raise_for_status()
        return resp.json()

    async def tokenize(self, backend: str, text: str) -> int:
        """POST /v1/tokenize → token count for text using the specified backend."""
        resp = await self._client.post(
            f"{self._base_url}/v1/tokenize",
            json={"backend": backend, "text": text},
        )
        resp.raise_for_status()
        return int(resp.json()["token_count"])

    async def health(self) -> bool:
        """Returns True if the Inference Service is reachable."""
        try:
            resp = await self._client.get(f"{self._base_url}/health")
            return resp.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        """Close the underlying connection pool.  Call from app shutdown hook."""
        await self._client.aclose()


_http_client: InferenceClient | None = None
_grpc_client = None  # GrpcInferenceClient | None — imported lazily to avoid hard dep


def get_inference_client() -> InferenceClient:
    """
    Return the singleton inference client (HTTP or gRPC based on settings).

    The first call creates the client; subsequent calls reuse it.
    """
    global _http_client, _grpc_client

    if settings.inference_use_grpc:
        if _grpc_client is None:
            from .inference_grpc import GrpcInferenceClient
            _grpc_client = GrpcInferenceClient(
                host=settings.inference_grpc_host,
                port=settings.inference_grpc_port,
            )
        return _grpc_client  # type: ignore[return-value]

    if _http_client is None:
        _http_client = InferenceClient(
            base_url=settings.inference_url,
            timeout=settings.request_timeout,
        )
    return _http_client


async def close_inference_client() -> None:
    """Close and reset all client singletons.  Call from app shutdown hook."""
    global _http_client, _grpc_client
    if _http_client is not None:
        await _http_client.close()
        _http_client = None
    if _grpc_client is not None:
        await _grpc_client.close()
        _grpc_client = None
