"""
Training trigger client
========================
Calls ``POST /v1/training/trigger`` on the Training Service after a
new experience is stored, so the Training Service can decide whether
to start a LoRA run.

The call is fire-and-forget — failures are logged but not propagated
to the caller.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

from shared.contracts.training import TrainingTriggerRequest
from ..config.settings import settings
from ..storage.jsonl_store import count_experiences

logger = logging.getLogger(__name__)

# Keeps strong references to fire-and-forget tasks so the GC cannot collect
# them before they complete.  Each task removes itself via done callback.
_background_tasks: set = set()


async def _notify_training(capability: str, new_count: int, session_id: str) -> None:
    # In API mode the operator leaves EXPERIENCE__TRAINING_URL unset (or empty)
    # to skip starting the Training Service entirely.  Nothing to notify.
    if not settings.training_url:
        return

    total = await count_experiences()
    if total < settings.min_batch_size:
        logger.debug(
            "Training skip — only %d experiences stored (threshold %d)",
            total,
            settings.min_batch_size,
        )
        return

    payload = TrainingTriggerRequest(
        capability=capability,
        new_experience_count=new_count,
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


def fire_notify_training(capability: str, new_count: int, session_id: str) -> None:
    """Schedule the training notification as a background asyncio task."""
    task = asyncio.create_task(
        _notify_training(capability, new_count, session_id)
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
