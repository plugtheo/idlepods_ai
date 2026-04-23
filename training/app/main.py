"""
Training Service — entry point
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .config.settings import settings
from .routes.trigger import router as trigger_router

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s | %(name)s | %(message)s",
)

_log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _log.info("Training Service started on port %d", settings.port)
    yield


app = FastAPI(
    title="Training Service",
    description=(
        "Evaluates experience batch diversity thresholds and launches LoRA "
        "adapter training in an isolated subprocess when the batch is large "
        "and varied enough. Exposes a trigger endpoint for the Experience "
        "Service to call after every new record."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(trigger_router)
