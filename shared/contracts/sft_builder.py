"""
SFT pair builder.

Converts an AgentContribution + AdapterRecipe into a training record.
For sft_format="openai_messages", produces {"messages": [...]} suitable
for TRL >= 0.24 SFTTrainer with apply_chat_template.
"""
from __future__ import annotations

from typing import Any, Dict, List, Union

from shared.contracts.experience import AgentContribution
from shared.contracts.training import AdapterRecipe


def build_sft_pair(
    contribution: AgentContribution,
    recipe: AdapterRecipe,
    role: str,
    system_prompt: str = "",
    user_prompt: str = "",
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Return a training record dict keyed by the recipe's sft_format.

    For openai_messages:
        {"messages": [system, user, assistant{tool_calls}, tool{...}, assistant{final}]}

    """
    if recipe.sft_format == "openai_messages":
        return _build_openai_messages_record(contribution, system_prompt, user_prompt)
    raise ValueError("Unsupported sft_format")

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
        assistant_tool_turn = {"role": "assistant", "tool_calls": list(tool_calls), "content": None}
        
        tool_target_msgs = list(messages) + [assistant_tool_turn]
        tool_target_record = {
            "messages": tool_target_msgs,
        }   
        
        full_msgs = list(messages) + [assistant_tool_turn]
        for result in tool_results:
            full_msgs.append({
                "role": "tool",
                "tool_call_id": result.get("tool_call_id", ""),
                "content": result.get("content", ""),
            })
        
        _last = full_msgs[-1] if full_msgs else None
        final_content = contribution.output or ""
        if not (_last and _last.get("role") == "assistant" and (_last.get("content") or "") == final_content):
            full_msgs.append({"role": "assistant", "content": final_content})

        full_record = {
            "messages": full_msgs,
        }

        return [tool_target_record, full_record]

    for result in tool_results:
        messages.append({
            "role": "tool",
            "tool_call_id": result.get("tool_call_id", ""),
            "content": result.get("content", ""),
        })

    # Append final assistant (guard against duplicate)
    _last = messages[-1] if messages else None
    final_content = contribution.output or ""
    if not (_last and _last.get("role") == "assistant" and (_last.get("content") or "") == final_content):
        messages.append({"role": "assistant", "content": final_content})


    return {"messages": messages}