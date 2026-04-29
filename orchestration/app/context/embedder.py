"""
Embedding model singleton
==========================
Thin wrapper around sentence-transformers.  The model is loaded once on first
use (lazy singleton).  Thread-safe via double-checked locking with a
threading.Lock — the model is initialised inside run_in_executor thread-pool
workers, so an asyncio.Lock is not sufficient here.
"""

from __future__ import annotations

import asyncio
import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)

_model = None
_model_lock = threading.Lock()


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer
                from ..config.settings import settings
                logger.info("Loading embedding model: %s", settings.embedding_model)
                _model = SentenceTransformer(settings.embedding_model)
                logger.info("Embedding model loaded.")
    return _model


def embed(text: str) -> np.ndarray:
    """Convert *text* to a normalised dense vector (shape (384,), float32)."""
    model = _get_model()
    vector = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return vector.astype(np.float32)


async def embed_async(text: str) -> np.ndarray:
    """Async wrapper — runs CPU-bound encode in the default thread-pool executor."""
    return await asyncio.to_thread(embed, text)
