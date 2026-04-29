"""
RemoteVLLMBackend
=================
Connects to any vLLM-compatible server (OpenAI-compatible API).
Suitable for user-hosted GPU servers, shared inference endpoints, or
cloud-deployed vLLM instances.

Structurally identical to LocalVLLMBackend.  The only differences are:
- The base URL is externally configured rather than derived from container names.
- An optional Bearer token auth header is added when a token is provided.
- SSL verification can be toggled.

The same _fix_bpe_artifacts() post-processing is applied to all output and
the same adapter availability cache (2-minute TTL) is used.
"""

from __future__ import annotations

import json
import logging
import time
from typing import AsyncGenerator, Optional

import httpx

from shared.contracts.inference import GenerateRequest, GenerateResponse
from ..config.settings import settings
from .base import InferenceBackend, InferenceError
from .local_vllm import (
    ADAPTER_CACHE_TTL_SECONDS,
    _AdapterRegistry,
    _ADAPTER_STOP_TOKENS,
    _build_adapter_prompt,
    _fix_bpe_artifacts,
    _resolve_model,
)

logger = logging.getLogger(__name__)


class RemoteVLLMBackend(InferenceBackend):
    """
    Connects to any vLLM-compatible server (OpenAI-compatible API).
    Suitable for user-hosted GPU servers, shared inference endpoints,
    or cloud-deployed vLLM instances.
    """

    def __init__(
        self,
        model_family: str,
        base_url: str,
        model_id: str,
        auth_token: str = "",
        ssl_verify: bool = True,
    ) -> None:
        self._model_family = model_family
        self._base_url = base_url.rstrip("/")
        self._model_id = model_id
        self._auth_token = auth_token
        self._timeout = settings.request_timeout_seconds

        headers = {}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        self._client = httpx.AsyncClient(
            timeout=self._timeout,
            verify=ssl_verify,
            headers=headers,
            limits=httpx.Limits(
                max_connections=settings.http_max_connections,
                max_keepalive_connections=settings.http_max_keepalive_connections,
            ),
        )
        self._registry = _AdapterRegistry(base_url, model_id)

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        effective_model = await _resolve_model(
            self._model_id, request.adapter_name, self._registry, request.role
        )

        messages = [m.model_dump() for m in request.messages]
        using_adapter = effective_model != self._model_id

        logger.debug(
            "RemoteVLLM → %s  model=%s  role=%s  msgs=%d  adapter=%s",
            self._base_url, effective_model, request.role, len(messages), using_adapter,
        )

        try:
            if using_adapter:
                payload = {
                    "model":       effective_model,
                    "prompt":      _build_adapter_prompt(messages),
                    "max_tokens":  request.max_tokens,
                    "temperature": request.temperature,
                    "top_p":       request.top_p,
                    "stop":        _ADAPTER_STOP_TOKENS,
                }
                resp = await self._client.post(
                    f"{self._base_url}/v1/completions",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                resp.raise_for_status()
                data = resp.json()
                content = _fix_bpe_artifacts(data["choices"][0]["text"])
            else:
                payload = {
                    "model":       effective_model,
                    "messages":    messages,
                    "max_tokens":  request.max_tokens,
                    "temperature": request.temperature,
                    "top_p":       request.top_p,
                }
                resp = await self._client.post(
                    f"{self._base_url}/v1/chat/completions",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                resp.raise_for_status()
                data = resp.json()
                content = _fix_bpe_artifacts(data["choices"][0]["message"]["content"])
        except Exception as exc:
            raise InferenceError(str(exc)) from exc

        tokens_out = data.get("usage", {}).get("completion_tokens", 0)

        logger.info(
            "RemoteVLLM ← role=%s  family=%s  model=%s  tokens=%d",
            request.role, self._model_family, effective_model, tokens_out,
        )

        return GenerateResponse(
            content=content,
            model_family=self._model_family,
            role=request.role,
            tokens_generated=tokens_out,
            session_id=request.session_id,
        )

    async def generate_stream(
        self, request: GenerateRequest
    ) -> AsyncGenerator[str, None]:
        """Stream tokens via SSE from the remote vLLM server."""
        effective_model = await _resolve_model(
            self._model_id, request.adapter_name, self._registry, request.role
        )

        messages = [m.model_dump() for m in request.messages]
        using_adapter = effective_model != self._model_id

        if using_adapter:
            stream_payload: dict = {
                "model":       effective_model,
                "prompt":      _build_adapter_prompt(messages),
                "max_tokens":  request.max_tokens,
                "temperature": request.temperature,
                "top_p":       request.top_p,
                "stop":        _ADAPTER_STOP_TOKENS,
                "stream":      True,
            }
            stream_endpoint = f"{self._base_url}/v1/completions"
        else:
            stream_payload = {
                "model":       effective_model,
                "messages":    messages,
                "max_tokens":  request.max_tokens,
                "temperature": request.temperature,
                "top_p":       request.top_p,
                "stream":      True,
            }
            stream_endpoint = f"{self._base_url}/v1/chat/completions"

        async with self._client.stream(
            "POST",
            stream_endpoint,
            json=stream_payload,
            headers={"Content-Type": "application/json"},
        ) as resp:
            resp.raise_for_status()
            async for raw_line in resp.aiter_lines():
                if not raw_line.startswith("data: "):
                    continue
                data = raw_line[6:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if using_adapter:
                    token = chunk.get("choices", [{}])[0].get("text", "")
                else:
                    token = (
                        chunk.get("choices", [{}])[0]
                        .get("delta", {})
                        .get("content", "")
                    )
                if token:
                    yield _fix_bpe_artifacts(token)

    async def list_adapters(self) -> list[str]:
        """Return names of currently loaded LoRA adapters for this backend."""
        return await self._registry.list_known()

    async def health(self) -> dict:
        """Check the remote vLLM server's health endpoint."""
        try:
            resp = await self._client.get(f"{self._base_url}/health", timeout=5.0)
            resp.raise_for_status()
            status = "ok"
        except Exception as exc:
            logger.warning("Health check failed for %s: %s", self._base_url, exc)
            status = "unavailable"
        return {
            "status": status,
            "backend": "remote_vllm",
            "url": self._base_url,
        }
