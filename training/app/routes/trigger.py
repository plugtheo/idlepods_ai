"""
Training trigger route — POST /v1/training/trigger

Accepts a TrainingTriggerRequest, evaluates diversity thresholds against
the current JSONL experience store, and launches trainer_entry if criteria
are met.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from shared.contracts.training import TrainingTriggerRequest, TrainingTriggerResponse
from ..utils.experience_reader import check_diversity, load_experiences
from ..utils.trainer_launcher import is_training_running, launch_training

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/v1/training/status", summary="Return current training state")
async def status() -> dict:
    return {"running": is_training_running()}


def to_training_records(experiences: list) -> list:
    """Pass experience records through; trainer_entry handles the format."""
    return list(experiences)


@router.post(
    "/v1/training/trigger",
    response_model=TrainingTriggerResponse,
    summary="Evaluate threshold and trigger training if criteria are met",
)
async def trigger(request: TrainingTriggerRequest) -> TrainingTriggerResponse:
    capability = request.capability

    if is_training_running():
        return TrainingTriggerResponse(
            capability=capability,
            triggered=False,
            reason="training job in progress",
        )

    experiences = load_experiences()
    passed, reason = check_diversity(experiences)
    if not passed:
        return TrainingTriggerResponse(
            capability=capability,
            triggered=False,
            reason=reason,
        )

    records = to_training_records(experiences)
    asyncio.create_task(launch_training(capability, records))
    logger.info("Training triggered for capability=%s records=%d", capability, len(records))

    return TrainingTriggerResponse(
        capability=capability,
        triggered=True,
        reason=reason,
    )
