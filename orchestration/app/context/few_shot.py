"""
Few-shot retrieval (RAG)
=========================
Searches the ChromaDB vector store for past experiences semantically similar
to the current user prompt.  Returns an empty list gracefully when the store
is empty or unavailable so the pipeline continues without enrichment.

All ChromaDB queries and embedding calls run in asyncio.to_thread() to avoid
blocking the event loop under concurrent requests.
"""

from __future__ import annotations

import asyncio
import logging
from typing import List

from shared.contracts.context import FewShotExample
from ..config.settings import settings
from ..db.chroma import get_chroma_client
from .embedder import embed_async

logger = logging.getLogger(__name__)

_collection = None
_collection_lock = asyncio.Lock()


def _init_collection_sync():
    client = get_chroma_client()
    collection = client.get_or_create_collection(
        name=settings.chromadb_collection,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info(
        "ChromaDB collection '%s' ready (count=%d)",
        settings.chromadb_collection,
        collection.count(),
    )
    return collection


async def _get_collection():
    global _collection
    if _collection is not None:
        return _collection
    async with _collection_lock:
        if _collection is not None:
            return _collection
        try:
            _collection = await asyncio.to_thread(_init_collection_sync)
        except Exception as exc:
            logger.warning("ChromaDB unavailable: %s — few-shot retrieval disabled.", exc)
    return _collection


async def search(
    prompt: str,
    *,
    exclude_task_id: str | None = None,
) -> List[FewShotExample]:
    """
    Return up to settings.max_few_shots past experiences similar to *prompt*.

    When ``exclude_task_id`` is set, results from that task are filtered out
    via a ChromaDB ``where`` clause so the current run cannot retrieve its own
    in-flight experiences (or any prior ones from the same multi-turn task).

    Returns an empty list when the store is empty or unavailable.
    """
    collection = await _get_collection()
    if collection is None:
        return []

    try:
        count = await asyncio.to_thread(lambda: collection.count())
        if count == 0:
            return []

        query_vector = (await embed_async(prompt)).tolist()

        def _query():
            kwargs = {
                "query_embeddings": [query_vector],
                "n_results":        min(settings.max_few_shots, count),
                "include":          ["documents", "metadatas", "distances"],
            }
            if exclude_task_id:
                # Existing rows upserted before task_id metadata existed have an
                # empty-string task_id; the $ne filter still excludes the matching
                # current task_id without dropping the legacy rows.
                kwargs["where"] = {"task_id": {"$ne": exclude_task_id}}
            return collection.query(**kwargs)

        results = await asyncio.to_thread(_query)

        examples: List[FewShotExample] = []
        docs      = results.get("documents", [[]])[0]
        metas     = results.get("metadatas",  [[]])[0]
        distances = results.get("distances",   [[]])[0]

        for doc, meta, distance in zip(docs, metas, distances):
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
            "few_shot_retrieval",
            extra={
                "event": "few_shot_retrieval",
                "candidates": len(docs),
                "accepted": len(examples),
                "stored_categories": [m.get("category") for m in metas],
                "min_similarity": min((1.0 - float(d) for d in distances), default=None),
            },
        )
        return examples

    except Exception as exc:
        logger.error("Few-shot retrieval failed: %s", exc, exc_info=True)
        return []
