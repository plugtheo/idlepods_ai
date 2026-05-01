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

import json
import logging
from pathlib import Path

from .base import InferenceBackend
from ..config.settings import settings

logger = logging.getLogger(__name__)

# One singleton per model family, keyed by lowercased family name.
_backends: dict[str, InferenceBackend] = {}

# Maps capability label (bootstrap form) → model family (vLLM server).
# Must stay in sync with docker/compose.yml vLLM service split.
CAPABILITY_TO_FAMILY: dict[str, str] = {
    "coding":    "deepseek",
    "debugging": "deepseek",
    "review":    "deepseek",
    "planning":  "mistral",
    "research":  "mistral",
    "criticism": "mistral",
}


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


def get_backend_for_capability(capability: str) -> InferenceBackend:
    """Return the backend that serves *capability* (e.g. 'coding' → deepseek backend)."""
    family = CAPABILITY_TO_FAMILY.get(capability.lower())
    if family is None:
        raise ValueError(
            f"Unknown capability '{capability}'. Known: {list(CAPABILITY_TO_FAMILY)}"
        )
    return get_backend(family)


async def bootstrap_adapters(manifest_path: str) -> None:
    """
    On inference startup: read manifest and load active (and previous) adapters
    onto the appropriate vLLM server.  Replaces static --lora-modules.
    """
    p = Path(manifest_path)
    if not p.exists():
        logger.info("bootstrap_adapters: no manifest at %s — nothing to load", manifest_path)
        return
    try:
        manifest = json.loads(p.read_text())
    except Exception as exc:
        logger.warning("bootstrap_adapters: could not read manifest: %s — skipping", exc)
        return

    for adapter_name, entry in manifest.get("adapters", {}).items():
        capability = entry.get("capability", "")
        # New schema: active_path; old schema: fall through gracefully.
        active_path = entry.get("active_path")
        previous_path = entry.get("previous_path")
        if not active_path:
            logger.debug("bootstrap_adapters: %s has no active_path — skipping", adapter_name)
            continue
        try:
            backend = get_backend_for_capability(capability)
        except ValueError as exc:
            logger.warning("bootstrap_adapters: %s", exc)
            continue
        ok = await backend.load_adapter(adapter_name, active_path)
        if ok:
            logger.info("bootstrap_adapters: loaded %s", adapter_name)
        if previous_path:
            prev_name = f"{adapter_name}__prev"
            ok2 = await backend.load_adapter(prev_name, previous_path)
            if ok2:
                logger.info("bootstrap_adapters: loaded prev-warm %s", prev_name)
