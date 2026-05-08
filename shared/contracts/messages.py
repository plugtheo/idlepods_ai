"""
Shared OpenAI message construction helpers.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def build_openai_messages(
    role: str,
    system_prompt: str,
    user_prompt: str,
    tool_rounds: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Return an OpenAI-format messages list including any assistant{tool_calls}
    + tool{tool_call_id, content} rounds.

    tool_rounds: list of dicts with keys:
        - tool_calls: list of OpenAI tool_call objects (assistant turn)
        - tool_results: list of {tool_call_id, content} (tool turns)
    """
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    for round_ in (tool_rounds or []):
        tool_calls = round_.get("tool_calls")
        if tool_calls:
            messages.append({"role": "assistant", "tool_calls": tool_calls, "content": None})
        for result in round_.get("tool_results") or []:
            messages.append({
                "role": "tool",
                "tool_call_id": result.get("tool_call_id", ""),
                "content": result.get("content", ""),
            })
    return messages
