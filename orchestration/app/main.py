"""
Orchestration Service — entry point
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .config.settings import settings
from .routes.run import router as run_router

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s | %(name)s | %(message)s",
)

_log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _log.info("Orchestration Service started on port %d", settings.port)
    yield
    from .clients.inference import close_inference_client
    await close_inference_client()


app = FastAPI(
    title="Orchestration Service",
    description=(
        "Drives the IdleDev multi-agent LangGraph pipeline. "
        "Receives a prompt, routes it, enriches context, runs agents iteratively "
        "until converged, stores the experience, and returns the final response."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(run_router)
