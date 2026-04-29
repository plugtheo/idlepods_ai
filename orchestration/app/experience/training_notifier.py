"""
Training Service notifier
===========================
POST /v1/training/trigger on the Training Service after a new experience is
stored, so the Training Service can decide whether to start a LoRA run.

The call is fire-and-forget from the recorder — failures are logged but never
propagated to the caller.
"""

from __future__ import annotations

import logging

import httpx

from shared.contracts.training import TrainingTriggerRequest
from ..config.settings import settings

logger = logging.getLogger(__name__)


async def notify(capability: str, session_id: str) -> None:
    """POST a training trigger to the Training Service.  Non-fatal on any error."""
    if not settings.training_url:
        return

    payload = TrainingTriggerRequest(
        capability=capability,
        new_experience_count=1,
        session_id=session_id,
    )
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{settings.training_url}/v1/training/trigger",
                json=payload.model_dump(),
            )
            resp.raise_for_status()
            logger.info("Training trigger response: %s", resp.json())
    except Exception as exc:
        logger.warning("Training trigger failed (non-fatal): %s", exc)
