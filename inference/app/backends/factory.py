"""
Backend factory
================
Returns the local vLLM inference backend singleton.
Import `get_backend()` everywhere you need an inference backend — never
instantiate backends directly.
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

    from .local_vllm import LocalVLLMBackend
    logger.info("InferenceBackend: LocalVLLMBackend (deepseek@%s, mistral@%s)",
                settings.deepseek_url, settings.mistral_url)
    _backend_instance = LocalVLLMBackend()

    return _backend_instance
