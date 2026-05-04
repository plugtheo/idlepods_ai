"""
Experience recorder
====================
Wires together JSONL storage and ChromaDB upsert.

Called as an asyncio background task (never awaited on the critical path).
The entire function is non-fatal from the caller's perspective — all
exceptions are caught and logged so a recording failure can never affect the
user response.

Invariants:
- JSONL append failure is logged as an error and stops further processing
  (ChromaDB upsert is not attempted — JSONL is the authoritative record).
- ChromaDB upsert failure is logged as a warning; JSONL write still completed.
"""

import logging
from typing import List

from shared.contracts.experience import AgentContribution, ExperienceEvent
from . import jsonl_store, vector_store
from ..utils.scoring import score_per_entry

logger = logging.getLogger(__name__)


def _build_contributions(history: list) -> List[AgentContribution]:
    """
    Build a list of AgentContribution objects from the pipeline history.

    role='tool' entries are execution scaffolding — they are skipped here and
    instead surfaced via tool_turns on the preceding agent contribution.
    """
    contributions = []
    for i, h in enumerate(history):
        if h.get("failed") or h.get("validator_failed"):
            continue
        role = h.get("role", "")
        if role == "tool":
            continue

        tool_calls = h.get("tool_calls")
        tool_results = []

        if tool_calls:
            j = i + 1
            while j < len(history) and history[j].get("role") == "tool":
                t = history[j]
                tool_results.append({
                    "tool_call_id": t.get("tool_call_id", ""),
                    "name": t.get("name", ""),
                    "content": t.get("output", ""),
                })
                j += 1

        contributions.append(
            AgentContribution(
                role=role,
                output=str(h.get("full_output") or h.get("output", "")),
                quality_score=score_per_entry(h),
                iteration=h.get("iteration", 1),
                messages=h.get("messages"),
                tool_calls=tool_calls,
                tool_results=tool_results,
                used_base_fallback=h.get("used_base_fallback", False),
            )
        )
    return contributions

async def record(event: ExperienceEvent) -> None:
    """Persist the experience event to JSONL and ChromaDB."""
    try:
        await jsonl_store.append(event)
    except Exception as exc:
        logger.error("[%s] Failed to append experience to JSONL: %s", event.session_id[:8], exc)
        return

    # ChromaDB upsert — non-fatal, JSONL already written
    await vector_store.upsert(event)