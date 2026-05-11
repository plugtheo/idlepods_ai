"""
SFT pair builder.

Converts an AgentContribution + AdapterRecipe into one or more training
records. For sft_format="openai_messages", each record is shaped as
{"messages": [...]} suitable for TRL >= 0.24 SFTTrainer with apply_chat_template.

Return contract
---------------
`build_sft_pair` always returns ``List[Dict[str, Any]]``. The list contains:
  - one record when the contribution has no tool calls
  - two records when the contribution invoked tools (a "tool_target" record
    that ends at the assistant→tool_calls turn, plus a "full" record that
    includes the tool results and the final assistant answer). Both records
    share the same shape so callers can iterate uniformly.
"""
from __future__ import annotations

from typing import Any, Dict, List

from shared.contracts.experience import AgentContribution
from shared.contracts.training import AdapterRecipe


def build_sft_pair(
    contribution: AgentContribution,
    recipe: AdapterRecipe,
    role: str,
    system_prompt: str = "",
    user_prompt: str = "",
) -> List[Dict[str, Any]]:
    """Return one or more training records keyed by ``recipe.sft_format``."""
    if recipe.sft_format == "openai_messages":
        return _build_openai_messages_records(contribution, system_prompt, user_prompt)
    raise ValueError(f"Unsupported sft_format: {recipe.sft_format!r}")


def _append_final_assistant(messages: List[Dict[str, Any]], final_content: str) -> None:
    """Append the final assistant turn unless it's already the last message."""
    last = messages[-1] if messages else None
    if last and last.get("role") == "assistant" and (last.get("content") or "") == final_content:
        return
    messages.append({"role": "assistant", "content": final_content})


def _build_openai_messages_records(
    contribution: AgentContribution,
    system_prompt: str,
    user_prompt: str,
) -> List[Dict[str, Any]]:
    # Seed message list: prefer the recorded message history (which already
    # includes the system + user turns) and fall back to constructing one
    # from the supplied prompts.
    if contribution.messages:
        base_messages: List[Dict[str, Any]] = list(contribution.messages)
    else:
        base_messages = []
        if system_prompt:
            base_messages.append({"role": "system", "content": system_prompt})
        if user_prompt:
            base_messages.append({"role": "user", "content": user_prompt})

    tool_calls = contribution.tool_calls or []
    tool_results = contribution.tool_results or []
    final_content = contribution.output or ""

    if tool_calls:
        assistant_tool_turn = {"role": "assistant", "tool_calls": list(tool_calls), "content": None}

        # Pair 1 — train the model to emit the tool_calls turn given the prompt.
        tool_target_messages = list(base_messages) + [assistant_tool_turn]

        # Pair 2 — train the model to produce the final answer given the
        # prompt + tool_calls + tool_results.
        full_messages = list(base_messages) + [assistant_tool_turn]
        for result in tool_results:
            full_messages.append({
                "role": "tool",
                "tool_call_id": result.get("tool_call_id", ""),
                "content": result.get("content", ""),
            })
        _append_final_assistant(full_messages, final_content)

        return [
            {"messages": tool_target_messages},
            {"messages": full_messages},
        ]

    # No tool calls — single record with any tool_results inline (rare) and
    # the final assistant turn.
    messages = list(base_messages)
    for result in tool_results:
        messages.append({
            "role": "tool",
            "tool_call_id": result.get("tool_call_id", ""),
            "content": result.get("content", ""),
        })
    _append_final_assistant(messages, final_content)
    return [{"messages": messages}]
