"""
SFT pair builder.

Converts an AgentContribution + AdapterRecipe into a training record.
For sft_format="openai_messages", produces {"messages": [...]} suitable
for TRL >= 0.24 SFTTrainer with apply_chat_template.
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
) -> Dict[str, Any]:
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


    return {"messages": messages }


    """
    Hard validation: Qwen3 chat template must wrap assistant tool-call turns
    inside {% generation %} ... {% endgeneration %}.

    If missing, training with assistant_only_loss=True will silently corrupt
    tool-call learning. This is a blocker for Phase 3.
    """

    sample_messages = [
        {"role": "user", "content": "test"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "t1",
                    "type": "python",
                    "function": {"name": "f", "arguments": "{}"},
                }
            ],
            "content": None,
        },
    ]

    rendered = tokenizer.apply_chat_template(
        sample_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    has_start = "{% generation %}" in rendered
    has_end = "{% endgeneration %}" in rendered

    if not (has_start and has_end):
        raise RuntimeError(
            "Qwen3 chat template is missing {% generation %} wrapping for "
            "assistant tool-call turns. Tool-call masking will be incorrect. "
            "Patch the chat template or install a custom assistant-mask collator."
        )