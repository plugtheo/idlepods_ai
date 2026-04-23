"""
Context Service — entry point
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .config.settings import settings
from .routes.build import router as build_router

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s | %(name)s | %(message)s",
)

_log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm the embedding model — avoids cold-start latency on the first request.
    from .utils.embedder import embed_async
    try:
        await embed_async("warmup")
        _log.info("Embedding model pre-loaded.")
    except Exception as exc:
        _log.warning("Embedding model warmup failed: %s", exc)

    # Pre-connect to ChromaDB — surfaces misconfigurations at startup rather than
    # silently on the first user request.
    from .retrieval.few_shot import _get_collection
    try:
        await _get_collection()
    except Exception as exc:
        _log.warning("ChromaDB startup probe failed: %s", exc)

    _log.info("Context Service started on port %d", settings.port)
    yield


app = FastAPI(
    title="Context Service",
    description=(
        "Enriches user prompts for the IdleDev pipeline. "
        "Runs few-shot RAG retrieval and repo snippet scanning concurrently."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(build_router)
