"""
Backend factory
================
Returns the configured inference backend for the Qwen model family.
Backends are singletons — one instance per model family, created on first
call and reused for the lifetime of the process.

Config drives the selection:
  INFERENCE__QWEN_BACKEND=local_vllm  (default) → LocalVLLMBackend
  INFERENCE__QWEN_BACKEND=remote_vllm            → RemoteVLLMBackend
"""

from __future__ import annotations

import logging

from .base import InferenceBackend
from ..config.settings import settings

logger = logging.getLogger(__name__)

_backends: dict[str, InferenceBackend] = {}


def get_backend(model_family: str) -> InferenceBackend:
    """
    Return the singleton inference backend for *model_family*.

    Parameters
    ----------
    model_family:
        "qwen" (case-insensitive).
    """
    family = model_family.lower()

    if family in _backends:
        return _backends[family]

    if family == "qwen":
        backend_type = settings.qwen_backend
        url          = settings.qwen_url
        model_id     = settings.qwen_model_id
        auth_token   = settings.qwen_auth_token
        ssl_verify   = settings.qwen_ssl_verify
    else:
        raise ValueError(
            f"Unknown model_family '{model_family}'. Supported: qwen"
        )

    if backend_type == "remote_vllm":
        from .remote_vllm import RemoteVLLMBackend
        backend = RemoteVLLMBackend(family, url, model_id, auth_token, ssl_verify)
        logger.info(
            "InferenceBackend[%s]: RemoteVLLMBackend → %s", family, url
        )
    else:
        from .local_vllm import LocalVLLMBackend
        backend = LocalVLLMBackend(family, url, model_id)
        logger.info(
            "InferenceBackend[%s]: LocalVLLMBackend → %s", family, url
        )

    _backends[family] = backend
    return backend
