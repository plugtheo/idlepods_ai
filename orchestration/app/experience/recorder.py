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

from __future__ import annotations

import logging

from typing import Any, Dict, List

from shared.contracts.experience import AgentContribution, ExperienceEvent
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


def _build_contributions(
    iteration_history: List[Dict[str, Any]],
    role: str,
    quality_score: float,
    iteration: int,
    messages: List[Dict[str, Any]],
) -> List[AgentContribution]:
    """
    Walk iteration_history and group consecutive (assistant-with-tool_calls,
    tool-result*) rows into AgentContribution objects.

    The final assistant turn (no tool_calls) becomes the `output` field.
    Pure role='tool' rows that are not paired are skipped defensively.
    role='tool_result' rows are skipped for backward-compat with Plan 1 history.
    """
    tool_calls_buf: List[Dict[str, Any]] = []
    tool_results_buf: List[Dict[str, Any]] = []
    contributions: List[AgentContribution] = []

    output_text = ""
    final_tool_calls: List[Dict[str, Any]] = []
    final_tool_results: List[Dict[str, Any]] = []

    for entry in iteration_history:
        entry_role = entry.get("role", "")
        if entry_role == "tool_result":
            continue
        if entry_role == "assistant":
            tc = entry.get("tool_calls")
            if tc:
                if tool_calls_buf:
                    final_tool_calls.extend(tool_calls_buf)
                    final_tool_results.extend(tool_results_buf)
                tool_calls_buf = list(tc)
                tool_results_buf = []
            else:
                final_tool_calls.extend(tool_calls_buf)
                final_tool_results.extend(tool_results_buf)
                tool_calls_buf = []
                tool_results_buf = []
                output_text = entry.get("content") or ""
        elif entry_role == "tool":
            if tool_calls_buf:
                tool_results_buf.append({
                    "tool_call_id": entry.get("tool_call_id", ""),
                    "content": entry.get("content", ""),
                })

    contributions.append(AgentContribution(
        role=role,
        output=output_text,
        quality_score=quality_score,
        iteration=iteration,
        messages=messages or None,
        tool_calls=final_tool_calls or None,
        tool_results=final_tool_results or None,
    ))
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