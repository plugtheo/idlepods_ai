"""
Training trigger route
========================
POST /v1/training/trigger   — evaluate thresholds and optionally start LoRA
GET  /health                — liveness probe
GET  /v1/training/status    — returns whether a training job is currently running
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter

from shared.contracts.training import TrainingTriggerRequest, TrainingTriggerResponse
from ..config.settings import settings
from ..utils.experience_reader import check_diversity, load_experiences, to_training_records
from ..utils.trainer_launcher import launch_training, is_training_running


logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/v1/training/trigger",
    response_model=TrainingTriggerResponse,
    summary="Evaluate diversity thresholds and conditionally start LoRA training",
)
async def trigger_training(request: TrainingTriggerRequest) -> TrainingTriggerResponse:
    """
    Called by the Experience Service after every new experience is stored.

    1. Loads all experiences from the shared JSONL file.
    2. Runs the diversity check (batch size + score spread + fingerprint ratio).
    3. If all checks pass and no training is running, launches LoRA in background.
    4. Returns a response indicating whether training was triggered and why.
    """
    # LoRA training requires locally downloaded base model weights (loaded into GPU
    # RAM by Unsloth before rank-decomposed update matrices can be built).
    if is_training_running():
        return TrainingTriggerResponse(
            capability=request.capability,
            triggered=False,
            reason="training already in progress",
        )

    records = load_experiences()
    passed, reason = check_diversity(records)

    if not passed:
        logger.debug("Training not triggered: %s", reason)
        return TrainingTriggerResponse(
            capability=request.capability,
            triggered=False,
            reason=reason,
        )

    training_records = to_training_records(records)
    if not training_records:
        return TrainingTriggerResponse(
            capability=request.capability,
            triggered=False,
            reason="no records meet minimum quality score after filtering",
        )

    asyncio.create_task(
        launch_training(request.capability, training_records)
    )

    logger.info(
        "Training triggered: capability=%s, records=%d (%s)",
        request.capability,
        len(training_records),
        reason,
    )
    return TrainingTriggerResponse(
        capability=request.capability,
        triggered=True,
        reason=reason,
    )


@router.get("/v1/training/status", summary="Is a training job running?")
async def training_status() -> dict:
    return {"running": is_training_running()}


@router.get("/health", summary="Health check")
async def health() -> dict:
    return {"status": "ok", "service": "training"}
