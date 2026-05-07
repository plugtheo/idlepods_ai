"""
Training Service — entry point

Exposes:
  POST /v1/training/trigger  — evaluate thresholds and launch training if met
  GET  /health               — health check for Docker probes
"""

from fastapi import FastAPI

from .routes.trigger import router as trigger_router

app = FastAPI(
    title="Training Service",
    description="Evaluates training thresholds and launches LoRA adapter fine-tuning.",
    version="1.0.0",
)

app.include_router(trigger_router)


@app.get("/health", summary="Health check")
async def health() -> dict:
    return {"status": "ok", "service": "training"}
