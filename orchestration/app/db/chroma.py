"""
ChromaDB client factory
========================
Returns a module-level singleton ChromaDB client.

- CHROMADB_HOST set and non-empty → chromadb.HttpClient (remote server)
- CHROMADB_HOST empty (default)   → chromadb.PersistentClient (local file store)

Both the experience upsert path and the few-shot retrieval path call this
factory and receive the same client instance, ensuring they read/write the
same collection.
"""

from __future__ import annotations

import json
import logging
import threading

logger = logging.getLogger(__name__)

_TASK_FP_COLLECTION = "task_fingerprints"

_client = None
_client_lock = threading.Lock()


def get_chroma_client():
    """Return the singleton ChromaDB client, initialising it on first call."""
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is not None:
            return _client
        from ..config.settings import settings
        import chromadb

        if settings.chromadb_host:
            _client = chromadb.HttpClient(
                host=settings.chromadb_host,
                port=settings.chromadb_port,
                ssl=settings.chromadb_ssl,
            )
            logger.info(
                "ChromaDB HttpClient → %s:%d (ssl=%s)",
                settings.chromadb_host,
                settings.chromadb_port,
                settings.chromadb_ssl,
            )
        else:
            _client = chromadb.PersistentClient(path=settings.chromadb_path)
            logger.info("ChromaDB PersistentClient → %s", settings.chromadb_path)

    return _client


def load_task_fingerprints(task_id: str) -> dict[str, str]:
    """Return the persisted fingerprint dict for ``task_id``, or {} if absent."""
    try:
        col = get_chroma_client().get_or_create_collection(_TASK_FP_COLLECTION)
        result = col.get(ids=[task_id], include=["documents"])
        docs = result.get("documents") or []
        if docs and docs[0]:
            return json.loads(docs[0])
    except Exception as exc:
        logger.warning("load_task_fingerprints(%s) failed: %s", task_id, exc)
    return {}


def save_task_fingerprints(task_id: str, fingerprints: dict[str, str]) -> None:
    """Upsert ``fingerprints`` for ``task_id`` into ChromaDB."""
    try:
        col = get_chroma_client().get_or_create_collection(_TASK_FP_COLLECTION)
        col.upsert(ids=[task_id], documents=[json.dumps(fingerprints)])
    except Exception as exc:
        logger.warning("save_task_fingerprints(%s) failed: %s", task_id, exc)
