"""
ChromaDB upsert — experience vectors
=======================================
Embeds the experience prompt and upserts it into the shared ChromaDB
collection so the few-shot retrieval path can find semantically similar
past solutions.

ChromaDB failure is non-fatal: the JSONL file is the authoritative record.
All exceptions are caught and logged as warnings.
"""

from __future__ import annotations

import asyncio
import logging

from shared.contracts.experience import ExperienceEvent
from ..config.settings import settings
from ..db.chroma import get_chroma_client

logger = logging.getLogger(__name__)

_collection = None
_collection_lock = asyncio.Lock()


def _get_collection_sync():
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=settings.chromadb_collection,
        metadata={"hnsw:space": "cosine"},
    )


async def _get_collection():
    global _collection
    if _collection is not None:
        return _collection
    async with _collection_lock:
        if _collection is not None:
            return _collection
        try:
            _collection = await asyncio.to_thread(_get_collection_sync)
        except Exception as exc:
            logger.warning("ChromaDB collection unavailable: %s — upsert disabled.", exc)
    return _collection


async def upsert(event: ExperienceEvent) -> None:
    """Embed the prompt and upsert into ChromaDB.  Non-fatal on any error."""
    try:
        collection = await _get_collection()
        if collection is None:
            logger.warning("[%s] ChromaDB unavailable — skipping upsert", event.session_id[:8])
            return

        from ..context.embedder import embed_async
        embedding = (await embed_async(event.prompt)).tolist()

        def _upsert_sync():
            collection.upsert(
                ids=[event.session_id],
                embeddings=[embedding],
                documents=[event.prompt],
                metadatas=[{
                    "solution": event.final_output,
                    "score": str(event.final_score),
                    "agent_chain": ",".join(event.agent_chain),
                    "converged": str(event.converged),
                }],
            )

        await asyncio.to_thread(_upsert_sync)
        logger.debug("[%s] Experience upserted to ChromaDB", event.session_id[:8])
    except Exception as exc:
        logger.warning("[%s] ChromaDB upsert failed: %s", event.session_id[:8], exc)
