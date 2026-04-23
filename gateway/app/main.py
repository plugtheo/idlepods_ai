"""
Gateway Service — entry point
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .config.settings import settings
from .clients.orchestration import close as close_orchestration_client
from .middleware.auth import APIKeyMiddleware
from .routes.chat import router as chat_router

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(levelname)-8s | %(name)s | %(message)s",
)

_log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _log.info("Gateway Service started on port %d", settings.port)
    yield
    await close_orchestration_client()


app = FastAPI(
    title="IdleDev Gateway",
    description=(
        "Single entry-point for all external client requests. "
        "Classifies prompts, enforces authentication, and proxies to the "
        "Orchestration Service."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(APIKeyMiddleware)
app.include_router(chat_router)
