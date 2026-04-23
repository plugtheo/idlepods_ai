"""
Experience Service — entry point
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .config.settings import settings
from .routes.record import router as record_router

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s | %(name)s | %(message)s",
)

_log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _log.info("Experience Service started on port %d", settings.port)
    yield


app = FastAPI(
    title="Experience Service",
    description=(
        "Stores agent interaction records (prompt → output, quality score, "
        "agent chain) to a local JSONL file and a ChromaDB vector store. "
        "Notifies the Training Service after every write so it can evaluate "
        "whether a LoRA training run should begin."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(record_router)
