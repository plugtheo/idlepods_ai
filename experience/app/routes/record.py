"""
Experience record route
=========================
POST /v1/experience   — record one ExperienceEvent
GET  /health          — liveness probe
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from shared.contracts.experience import ExperienceEvent
from ..clients.training import fire_notify_training
from ..storage.jsonl_store import append_experience
from ..storage.vector_store import upsert_experience

logger = logging.getLogger(__name__)

router = APIRouter()


class RecordResponse(BaseModel):
    session_id: str
    stored: bool


@router.post(
    "/v1/experience",
    response_model=RecordResponse,
    summary="Record an experience event",
)
async def record_experience(event: ExperienceEvent) -> RecordResponse:
    """
    1. Append the event to the JSONL file (source of truth).
    2. Upsert the embedded prompt into ChromaDB (for future RAG retrieval).
    3. Fire-and-forget a training trigger notification.
    """
    try:
        await append_experience(event)
    except Exception as exc:
        logger.error("[%s] Failed to store experience: %s", event.session_id[:8], exc)
        raise HTTPException(status_code=500, detail=f"Storage error: {exc}") from exc

    # Non-critical — don't fail the request if ChromaDB is unavailable
    await upsert_experience(event)

    # Derive capability from agent_chain (e.g. "coder" → "coding")
    capability = _infer_capability(event.agent_chain)
    fire_notify_training(capability, new_count=1, session_id=event.session_id)

    return RecordResponse(session_id=event.session_id, stored=True)


@router.get("/health", summary="Health check")
async def health() -> dict:
    return {"status": "ok", "service": "experience"}


def _infer_capability(agent_chain: list[str]) -> str:
    """Map the dominant agent in the chain to a capability label string."""
    # Priority order: first matching agent wins.
    # Returns capability labels (e.g. "coding") not role names (e.g. "coder")
    # so the Training Service receives the expected format without substring hacks.
    priority = [
        ("coder",      "coding"),
        ("debugger",   "debugging"),
        ("researcher", "research"),
        ("planner",    "planning"),
        ("reviewer",   "review"),
        ("critic",     "criticism"),
    ]
    for role, capability in priority:
        if role in agent_chain:
            return capability
    return "general"
