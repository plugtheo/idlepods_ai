"""
Backend factory
================
Returns the configured inference backend for a given backend name.
Backends are singletons keyed by backend name, created on first call
and reused for the lifetime of the process.

Backend identity and config are read from models.yaml via the shared registry.
"""

from __future__ import annotations

import logging
from pathlib import Path

from shared.contracts.models import load_registry, get_backend_entry
from shared.manifest import read_manifest, LegacyManifestError

from .base import InferenceBackend

logger = logging.getLogger(__name__)

_backends: dict[str, InferenceBackend] = {}


def get_backend(backend_name: str) -> InferenceBackend:
    """Return the singleton inference backend for *backend_name*."""
    if backend_name in _backends:
        return _backends[backend_name]

    entry = get_backend_entry(backend_name)

    if entry.backend_type == "remote_vllm":
        from .remote_vllm import RemoteVLLMBackend
        backend: InferenceBackend = RemoteVLLMBackend(backend_name, entry)
        logger.info("InferenceBackend[%s]: RemoteVLLMBackend → %s", backend_name, entry.served_url)
    else:
        from .local_vllm import LocalVLLMBackend
        backend = LocalVLLMBackend(backend_name, entry)
        logger.info("InferenceBackend[%s]: LocalVLLMBackend → %s", backend_name, entry.served_url)

    _backends[backend_name] = backend
    return backend


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
        manifest = read_manifest(p)
    except LegacyManifestError as exc:
        logger.fatal(
            "bootstrap_adapters: %s — skipping adapter load. "
            "Run: python scripts/migrate_manifest.py",
            exc,
        )
        return
    except Exception as exc:
        logger.warning("bootstrap_adapters: could not read manifest: %s — skipping", exc)
        return

    for adapter_name, entry in manifest.adapters.items():
        backend_key = entry.backend
        active_path = entry.active_path
        previous_path = entry.previous_path
        if not active_path:
            logger.debug("bootstrap_adapters: %s has no active_path — skipping", adapter_name)
            continue
        try:
            backend = get_backend(backend_key)
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
