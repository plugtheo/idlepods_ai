"""
Few-shot retrieval (RAG)
=========================
Searches the ChromaDB vector store for past experiences that are
semantically similar to the current user prompt.

The store is populated by the Experience Service.  If the store is
empty or unavailable, the function returns an empty list gracefully
so the Orchestration Service can continue without enrichment.
"""

from __future__ import annotations

import asyncio
import logging
from typing import List

from shared.contracts.context import FewShotExample
from ..config.settings import settings
from ..utils.embedder import embed_async

logger = logging.getLogger(__name__)

_chroma_collection = None  # lazy-initialised
_chroma_lock = asyncio.Lock()  # prevents concurrent double-init on startup


def _init_chroma_sync():
    """Synchronous ChromaDB initialisation — runs in a thread-pool executor."""
    import chromadb

    # Self-hosted ChromaDB HTTP server
    client = chromadb.HttpClient(
        host=settings.chroma_host,
        port=settings.chroma_port,
    )
    logger.info(
        "ChromaDB HttpClient connecting to %s:%d",
        settings.chroma_host,
        settings.chroma_port,
    )

    collection = client.get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info(
        "ChromaDB collection '%s' ready (count=%d)",
        settings.chroma_collection,
        collection.count(),
    )
    return collection


async def _get_collection():
    global _chroma_collection
    if _chroma_collection is not None:
        return _chroma_collection
    async with _chroma_lock:
        # Double-checked locking: re-test after acquiring the lock.
        if _chroma_collection is not None:
            return _chroma_collection
        try:
            loop = asyncio.get_running_loop()
            _chroma_collection = await loop.run_in_executor(None, _init_chroma_sync)
        except Exception as exc:
            logger.warning("ChromaDB unavailable: %s — few-shot retrieval disabled.", exc)
    return _chroma_collection


async def retrieve_few_shots(prompt: str) -> List[FewShotExample]:
    """
    Return up to `settings.max_few_shots` past experiences that are
    semantically similar to *prompt*.

    Parameters
    ----------
    prompt:
        Raw user prompt from the current request.

    Returns
    -------
    List of FewShotExample sorted by similarity (most similar first).
    Empty list when the store is empty or unavailable.
    """
    collection = await _get_collection()
    if collection is None:
        return []

    try:
        count = collection.count()
        if count == 0:
            return []

        query_vector = (await embed_async(prompt)).tolist()
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=min(settings.max_few_shots, count),
            include=["documents", "metadatas", "distances"],
        )

        examples: List[FewShotExample] = []
        docs      = results.get("documents", [[]])[0]
        metas     = results.get("metadatas",  [[]])[0]
        distances = results.get("distances",   [[]])[0]

        for doc, meta, distance in zip(docs, metas, distances):
            # ChromaDB 'cosine' space returns distance, not similarity.
            # similarity = 1 - distance
            similarity = 1.0 - float(distance)
            if similarity < settings.similarity_threshold:
                continue
            examples.append(
                FewShotExample(
                    problem=meta.get("problem", doc[:500]),
                    solution=meta.get("solution", ""),
                    score=float(meta.get("score", 0.0)),
                    category=meta.get("category", "general"),
                )
            )

        logger.info(
            "Few-shot retrieval: prompt_len=%d  candidates=%d  accepted=%d",
            len(prompt), len(docs), len(examples),
        )
        return examples

    except Exception as exc:
        logger.error("Few-shot retrieval failed: %s", exc, exc_info=True)
        return []
