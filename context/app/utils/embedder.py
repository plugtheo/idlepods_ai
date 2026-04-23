"""
Embedder
=========
Thin wrapper around sentence-transformers for converting text to
384-dimensional dense vectors.  The model is loaded once on first use
(lazy singleton) to avoid paying the startup cost when the service starts.

Usage:
    from services.context.app.utils.embedder import embed
    vector = embed("Write a rate limiter in Python")  # returns np.ndarray shape (384,)
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np

logger = logging.getLogger(__name__)

_model = None  # sentence_transformers.SentenceTransformer singleton


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        from ..config.settings import settings
        logger.info("Loading embedding model: %s", settings.embedding_model)
        _model = SentenceTransformer(settings.embedding_model)
        logger.info("Embedding model loaded.")
    return _model


def embed(text: str) -> np.ndarray:
    """
    Convert *text* to a normalised dense vector.

    Returns
    -------
    np.ndarray of shape (384,), dtype float32.
    """
    model = _get_model()
    vector = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return vector.astype(np.float32)


async def embed_async(text: str) -> np.ndarray:
    """
    Async wrapper around `embed` — runs the CPU-bound model inference in the
    default thread-pool executor so the event loop is not blocked.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, embed, text)
