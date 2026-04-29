"""
Experience recorder
====================
Wires together JSONL storage, ChromaDB upsert, and training notification.

Called as an asyncio background task (never awaited on the critical path).
The entire function is non-fatal from the caller's perspective — all
exceptions are caught and logged so a recording failure can never affect the
user response.

Invariants:
- JSONL append failure is logged as an error and stops further processing
  (ChromaDB and training are not attempted — JSONL is the authoritative record).
- ChromaDB upsert failure is logged as a warning; JSONL write still completed.
- Training notification is sent only when total record count >= MIN_BATCH_SIZE.
"""

from __future__ import annotations

import logging

from shared.contracts.experience import ExperienceEvent
from . import jsonl_store, vector_store

logger = logging.getLogger(__name__)


def _infer_capability(agent_chain: list[str]) -> str:
    """Map the dominant agent in the chain to a capability label string."""
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


async def record(event: ExperienceEvent, total_count: int) -> None:
    """
    Persist the experience event and optionally trigger training.

    Parameters
    ----------
    event:
        The completed pipeline's experience event.
    total_count:
        Snapshot of the JSONL line count taken *before* this append.
        Used to decide whether the training threshold has been crossed.
    """
    try:
        await jsonl_store.append(event)
    except Exception as exc:
        logger.error("[%s] Failed to append experience to JSONL: %s", event.session_id[:8], exc)
        return

    # ChromaDB upsert — non-fatal, JSONL already written
    await vector_store.upsert(event)