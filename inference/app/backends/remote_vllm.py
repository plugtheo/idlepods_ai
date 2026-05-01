"""
RemoteVLLMBackend
=================
Connects to any vLLM-compatible server (OpenAI-compatible API).
All requests go through /v1/chat/completions — adapters are activated via
the model name field, not the legacy /v1/completions endpoint.

Native tool calling is supported: pass tools in the request and tool_calls
are extracted from the response.  Thinking mode is always disabled via
chat_template_kwargs so Qwen3 runs in non-thinking mode.
"""

from __future__ import annotations

import json
import logging
from typing import AsyncGenerator, Optional

import httpx

from shared.contracts.inference import GenerateRequest, GenerateResponse
from ..config.settings import settings
from .base import InferenceBackend, InferenceError
from .local_vllm import (
    ADAPTER_CACHE_TTL_SECONDS,
    _AdapterRegistry,
    _fix_bpe_artifacts,
    _resolve_model,
)

logger = logging.getLogger(__name__)


class RemoteVLLMBackend(InferenceBackend):
    """
    Connects to any vLLM-compatible server (OpenAI-compatible API).
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

        messages = [m.model_dump(exclude_none=True) for m in request.messages]

        logger.debug(
            "RemoteVLLM → %s  model=%s  role=%s  msgs=%d  tools=%s",
            self._base_url, effective_model, request.role, len(messages),
            bool(request.tools),
        )

        payload: dict = {
            "model":       effective_model,
            "messages":    messages,
            "max_tokens":  request.max_tokens,
            "temperature": request.temperature,
            "top_p":       request.top_p,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if request.tools:
            payload["tools"] = [t.model_dump() for t in request.tools]

        try:
            resp = await self._client.post(
                f"{self._base_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            raise InferenceError(str(exc)) from exc

        choice  = data["choices"][0]
        message = choice["message"]
        content = _fix_bpe_artifacts(message.get("content") or "")
        tool_calls = message.get("tool_calls")

        tokens_out = data.get("usage", {}).get("completion_tokens", 0)

        logger.info(
            "RemoteVLLM ← role=%s  family=%s  model=%s  tokens=%d  tool_calls=%s",
            request.role, self._model_family, effective_model, tokens_out,
            bool(tool_calls),
        )

        return GenerateResponse(
            content=content,
            model_family=self._model_family,
            role=request.role,
            tokens_generated=tokens_out,
            session_id=request.session_id,
            tool_calls=tool_calls,
        )

    async def generate_stream(
        self, request: GenerateRequest
    ) -> AsyncGenerator[str, None]:
        """Stream tokens via SSE from the remote vLLM server."""
        effective_model = await _resolve_model(
            self._model_id, request.adapter_name, self._registry, request.role
        )

        messages = [m.model_dump(exclude_none=True) for m in request.messages]

        stream_payload: dict = {
            "model":       effective_model,
            "messages":    messages,
            "max_tokens":  request.max_tokens,
            "temperature": request.temperature,
            "top_p":       request.top_p,
            "stream":      True,
            "chat_template_kwargs": {"enable_thinking": False},
        }

        async with self._client.stream(
            "POST",
            f"{self._base_url}/v1/chat/completions",
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
                token = (
                    chunk.get("choices", [{}])[0]
                    .get("delta", {})
                    .get("content", "")
                )
                if token:
                    yield _fix_bpe_artifacts(token)

    async def load_adapter(self, adapter_name: str, lora_path: str) -> bool:
        """Load a LoRA adapter at runtime via vLLM's load_lora_adapter API."""
        return await self._registry.load(adapter_name, lora_path)

    async def unload_adapter(self, adapter_name: str) -> bool:
        """Unload a LoRA adapter at runtime via vLLM's unload_lora_adapter API."""
        return await self._registry.unload(adapter_name)

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
