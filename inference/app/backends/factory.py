"""
Backend factory
================
Returns the configured inference backend for a given model family.
Backends are singletons — one instance per model family, created on first
call and reused for the lifetime of the process.

Config drives the selection per family:
  INFERENCE__DEEPSEEK_BACKEND=local_vllm  (default) → LocalVLLMBackend
  INFERENCE__DEEPSEEK_BACKEND=remote_vllm            → RemoteVLLMBackend

Existing env vars (INFERENCE__DEEPSEEK_URL, INFERENCE__MISTRAL_URL) keep
their names and defaults so existing deployments work without config changes.
"""

from __future__ import annotations

import logging

from .base import InferenceBackend
from ..config.settings import settings

logger = logging.getLogger(__name__)

# One singleton per model family, keyed by lowercased family name.
_backends: dict[str, InferenceBackend] = {}


def get_backend(model_family: str) -> InferenceBackend:
    """
    Return the singleton inference backend for *model_family*.

    Parameters
    ----------
    model_family:
        "deepseek" or "mistral" (case-insensitive).
    """
    family = model_family.lower()

    if family in _backends:
        return _backends[family]

    if family == "deepseek":
        backend_type = settings.deepseek_backend
        url          = settings.deepseek_url
        model_id     = settings.deepseek_model_id
        auth_token   = settings.deepseek_auth_token
        ssl_verify   = settings.deepseek_ssl_verify
    elif family == "mistral":
        backend_type = settings.mistral_backend
        url          = settings.mistral_url
        model_id     = settings.mistral_model_id
        auth_token   = settings.mistral_auth_token
        ssl_verify   = settings.mistral_ssl_verify
    else:
        raise ValueError(
            f"Unknown model_family '{model_family}'. Supported: deepseek, mistral"
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
