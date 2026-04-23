"""
ChromaDB vector store for experiences
========================================
Embeds the experience prompt and stores it in a dedicated ChromaDB
collection so that the Context Service can later retrieve few-shot
examples by semantic similarity.

The collection is shared with, or separate from, the one used by the
Context Service — controlled by ``EXPERIENCE__CHROMA_COLLECTION``.
"""

from __future__ import annotations

import asyncio
import logging
import threading

from shared.contracts.experience import ExperienceEvent
from ..config.settings import settings

logger = logging.getLogger(__name__)

_chroma_client = None
_collection = None
_chroma_lock = asyncio.Lock()  # prevents concurrent double-init on startup

# Module-level embedding model singleton — loaded once, reused on every call.
# threading.Lock (not asyncio.Lock) because _get_embed_model() runs inside
# run_in_executor thread-pool workers, not on the event-loop thread.
_embed_model = None
_embed_model_lock = threading.Lock()


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        with _embed_model_lock:
            if _embed_model is None:  # double-checked locking
                from sentence_transformers import SentenceTransformer
                logger.info("Loading embedding model: %s", settings.embedding_model)
                _embed_model = SentenceTransformer(settings.embedding_model)
                logger.info("Embedding model loaded.")
    return _embed_model


def _get_collection_sync():
    global _chroma_client, _collection

    import chromadb

    if settings.chroma_api_key:
        # ChromaDB Cloud — api.trychroma.com
        _chroma_client = chromadb.CloudClient(
            tenant=settings.chroma_tenant,
            database=settings.chroma_database,
            api_key=settings.chroma_api_key,
        )
        logger.info(
            "ChromaDB CloudClient: tenant=%s  database=%s",
            settings.chroma_tenant,
            settings.chroma_database,
        )
    else:
        # Self-hosted ChromaDB HTTP server
        _chroma_client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
        )
        logger.info(
            "ChromaDB HttpClient connecting to %s:%d",
            settings.chroma_host,
            settings.chroma_port,
        )

    _collection = _chroma_client.get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


async def _get_collection():
    global _collection
    if _collection is not None:
        return _collection
    async with _chroma_lock:
        if _collection is not None:
            return _collection
        try:
            loop = asyncio.get_running_loop()
            _collection = await loop.run_in_executor(None, _get_collection_sync)
        except Exception as exc:
            logger.warning("ChromaDB unavailable: %s — vector store disabled.", exc)
    return _collection


def _embed_sync(text: str):
    """Embed *text* using the cached model singleton (not async-safe alone)."""
    model = _get_embed_model()
    return model.encode(text, show_progress_bar=False).tolist()


async def _embed(text: str):
    """Async wrapper — offloads CPU-bound encode to the default thread pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _embed_sync, text)


async def upsert_experience(event: ExperienceEvent) -> None:
    """Embed the prompt and upsert into ChromaDB."""
    try:
        collection = await _get_collection()
        if collection is None:
            logger.warning("[%s] ChromaDB unavailable — skipping upsert", event.session_id[:8])
            return
        embedding = await _embed(event.prompt)
        # Store under the same metadata keys the Context Service reads back:
        #   "solution" → final_output   (was "final_output", causing empty RAG solutions)
        #   "score"    → final_score    (was "final_score",   causing 0.0 RAG scores)
        collection.upsert(
            ids=[event.session_id],
            embeddings=[embedding],
            documents=[event.prompt],
            metadatas=[
                {
                    "solution": event.final_output,
                    "score": str(event.final_score),
                    "agent_chain": ",".join(event.agent_chain),
                    "converged": str(event.converged),
                }
            ],
        )
        logger.debug("[%s] Experience upserted to ChromaDB", event.session_id[:8])
    except Exception as exc:
        # Non-fatal — JSONL is the source of truth
        logger.warning("[%s] ChromaDB upsert failed: %s", event.session_id[:8], exc)
