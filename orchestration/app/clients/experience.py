"""
Experience Service async publisher
====================================
After a quality-converged pipeline run, Orchestration fires and forgets
an ExperienceEvent to the Experience Service.

The call is non-blocking — it is dispatched as a background asyncio Task.
If the Experience Service is unavailable the error is logged but the user
response is unaffected.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

from shared.contracts.experience import ExperienceEvent
from ..config.settings import settings

logger = logging.getLogger(__name__)


async def publish_experience(event: ExperienceEvent) -> None:
    """
    POST *event* to the Experience Service.

    This function is normally scheduled as an asyncio background task.
    It never raises — all exceptions are caught and logged.
    """
    url = f"{settings.experience_url.rstrip('/')}/v1/experience"

    try:
        payload = event.model_dump(mode="json")
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
        logger.info(
            "Experience published: session=%s  score=%.2f",
            event.session_id, event.final_score,
        )
    except Exception as exc:
        logger.warning(
            "Could not publish experience (session=%s): %s",
            event.session_id, exc,
        )


def fire_publish_experience(event: ExperienceEvent) -> None:
    """
    Schedule `publish_experience` as a background asyncio Task.

    Safe to call from synchronous code within an async context.
    The caller does not await this and the user response is not delayed.
    """
    asyncio.create_task(publish_experience(event))
