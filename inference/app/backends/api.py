"""
APIBackend (LiteLLM)
====================
Routes generation requests to an external LLM provider via LiteLLM.
LiteLLM presents a single OpenAI-compatible interface to 100+ providers
including Anthropic, OpenAI, Together.ai, Groq, Cohere, and more.

Default: Anthropic claude-3-5-haiku-20241022 — cost-efficient and reliable.
Swap provider via INFERENCE__API_PROVIDER and INFERENCE__API_MODEL env vars.

Requirements
------------
    pip install litellm

Role-to-model mapping
---------------------
In local mode each role maps to a specific model family (deepseek or
mistral).  In API mode a single model handles all roles — the role
distinction is entirely expressed through the system prompt included in
`request.messages[0]`.
"""

from __future__ import annotations

import logging
import os
from typing import AsyncGenerator

from shared.contracts.inference import GenerateRequest, GenerateResponse
from ..config.settings import settings
from .base import InferenceBackend

logger = logging.getLogger(__name__)


class APIBackend(InferenceBackend):
    """Calls any LiteLLM-supported provider for inference."""

    def __init__(self) -> None:
        try:
            import litellm as _litellm  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "litellm is required for API mode. "
                "Install it with: pip install litellm"
            ) from exc

        # Inject the API key into the environment so LiteLLM picks it up
        # via its standard env-var convention (e.g. ANTHROPIC_API_KEY).
        if settings.api_key:
            provider_upper = settings.api_provider.upper().replace("-", "_")
            env_var = f"{provider_upper}_API_KEY"
            os.environ.setdefault(env_var, settings.api_key)

        self._default_model = settings.api_model
        self._role_overrides = settings.role_model_overrides
        if self._role_overrides:
            logger.info(
                "APIBackend role overrides: %s",
                ", ".join(f"{r}={m}" for r, m in self._role_overrides.items()),
            )
        logger.info(
            "APIBackend ready: provider=%s  default_model=%s",
            settings.api_provider,
            self._default_model,
        )

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        import litellm

        # Per-role override: points to a fine-tuned endpoint in dev/stage/prod.
        # Falls back to the default api_model if no override is configured for this role.
        model = self._role_overrides.get(request.role, self._default_model)
        if model != self._default_model:
            logger.debug(
                "APIBackend role override: role=%s  model=%s",
                request.role, model,
            )

        messages = [m.model_dump() for m in request.messages]

        logger.debug(
            "APIBackend → model=%s  role=%s  msgs=%d",
            model, request.role, len(messages),
        )

        # litellm.acompletion is the async variant
        response = await litellm.acompletion(
            model=model,
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )

        content = response.choices[0].message.content or ""
        tokens_out = getattr(response.usage, "completion_tokens", 0)

        logger.info(
            "APIBackend ← role=%s  model=%s  tokens=%d",
            request.role, model, tokens_out,
        )

        return GenerateResponse(
            content=content,
            model_family=settings.api_provider,
            role=request.role,
            tokens_generated=tokens_out,
            session_id=request.session_id,
        )

    async def generate_stream(
        self, request: GenerateRequest
    ) -> AsyncGenerator[str, None]:
        """
        Stream tokens via LiteLLM's ``stream=True`` mode.

        Works for every LiteLLM-supported provider that exposes a streaming
        endpoint (Anthropic, OpenAI, Together.ai, Groq, …).  Each chunk
        carries ``choices[0].delta.content``; None/empty chunks are skipped.
        """
        import litellm

        model = self._role_overrides.get(request.role, self._default_model)
        messages = [m.model_dump() for m in request.messages]

        response = await litellm.acompletion(
            model=model,
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stream=True,
        )

        async for chunk in response:
            token: str = chunk.choices[0].delta.content or ""
            if token:
                yield token
