"""
Backend factory
================
Reads INFERENCE__MODE from the environment and returns the appropriate
InferenceBackend instance.  Import `get_backend()` everywhere you need
an inference backend — never instantiate backends directly.
"""

from __future__ import annotations

import logging

from .base import InferenceBackend
from ..config.settings import settings

logger = logging.getLogger(__name__)

_backend_instance: InferenceBackend | None = None


def get_backend() -> InferenceBackend:
    """
    Return the singleton inference backend.

    The instance is created once on first call and reused for the lifetime
    of the process.  Thread-safe for read access after the first call.
    """
    global _backend_instance

    if _backend_instance is not None:
        return _backend_instance

    mode = settings.mode.lower()

    if mode == "local":
        from .local_vllm import LocalVLLMBackend
        logger.info("InferenceBackend: LocalVLLMBackend (deepseek@%s, mistral@%s)",
                    settings.deepseek_url, settings.mistral_url)
        _backend_instance = LocalVLLMBackend()

    elif mode == "api":
        from .api import APIBackend
        logger.info("InferenceBackend: APIBackend (provider=%s, model=%s)",
                    settings.api_provider, settings.api_model)
        _backend_instance = APIBackend()

    else:
        raise ValueError(
            f"Unknown INFERENCE__MODE '{mode}'. Use 'local' or 'api'."
        )

    return _backend_instance
