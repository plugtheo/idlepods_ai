"""
SFT pair builder.

Converts an AgentContribution + AdapterRecipe into a training record.
For sft_format="openai_messages", produces {"messages": [...]} suitable
for TRL >= 0.24 SFTTrainer with apply_chat_template.
For sft_format="legacy_response_marker", produces {"prompt": ..., "completion": ...}.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from shared.contracts.experience import AgentContribution
from shared.contracts.training import AdapterRecipe


def build_sft_pair(
    contribution: AgentContribution,
    recipe: AdapterRecipe,
    role: str,
    system_prompt: str = "",
    user_prompt: str = "",
) -> Dict[str, Any]:
    """
    Return a training record dict keyed by the recipe's sft_format.

    For openai_messages:
        {"messages": [system, user, assistant{tool_calls}, tool{...}, assistant{final}]}

    For legacy_response_marker:
        {"prompt": "[SYSTEM]\\n...\\n[USER]\\n...\\n[RESPONSE]\\n", "completion": output}
    """
    if recipe.sft_format == "openai_messages":
        return _build_openai_messages_record(contribution, system_prompt, user_prompt)
    return _build_legacy_record(contribution, system_prompt, user_prompt)


def _build_openai_messages_record(
    contribution: AgentContribution,
    system_prompt: str,
    user_prompt: str,
) -> Dict[str, Any]:
    messages: List[Dict[str, Any]] = []

    if contribution.messages:
        messages = list(contribution.messages)
    else:
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if user_prompt:
            messages.append({"role": "user", "content": user_prompt})

    tool_calls = contribution.tool_calls or []
    tool_results = contribution.tool_results or []

    if tool_calls:
        messages.append({"role": "assistant", "tool_calls": tool_calls, "content": None})
        for result in tool_results:
            messages.append({
                "role": "tool",
                "tool_call_id": result.get("tool_call_id", ""),
                "content": result.get("content", ""),
            })

    messages.append({"role": "assistant", "content": contribution.output})
    return {"messages": messages}


def _build_legacy_record(
    contribution: AgentContribution,
    system_prompt: str,
    user_prompt: str,
) -> Dict[str, Any]:
    prompt = (
        f"[SYSTEM]\n{system_prompt}\n\n"
        f"[USER]\n{user_prompt}\n\n"
        f"[RESPONSE]\n"
    )
    return {"prompt": prompt, "completion": contribution.output}
