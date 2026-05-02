"""Tests for sft_builder.build_sft_pair."""
import pytest

from shared.contracts.experience import AgentContribution
from shared.contracts.training import AdapterRecipe
from orchestration.app.experience.sft_builder import build_sft_pair


def _make_contribution(**kwargs) -> AgentContribution:
    defaults = dict(
        role="coder",
        output="def add(a, b): return a + b",
        quality_score=0.9,
        iteration=1,
    )
    defaults.update(kwargs)
    return AgentContribution(**defaults)


def _openai_recipe() -> AdapterRecipe:
    return AdapterRecipe(
        target_modules=["q_proj"],
        sft_format="openai_messages",
        tool_call_style="openai_native",
    )


def _legacy_recipe() -> AdapterRecipe:
    return AdapterRecipe(
        target_modules=["q_proj"],
        sft_format="legacy_response_marker",
    )


def test_openai_messages_no_tool_calls():
    contrib = _make_contribution()
    result = build_sft_pair(contrib, _openai_recipe(), "coder",
                            system_prompt="You are a coder.", user_prompt="Write add().")
    assert "messages" in result
    msgs = result["messages"]
    roles = [m["role"] for m in msgs]
    assert roles[0] == "system"
    assert roles[1] == "user"
    assert roles[-1] == "assistant"
    assert msgs[-1]["content"] == "def add(a, b): return a + b"


def test_openai_messages_with_tool_calls():
    tool_calls = [{"id": "call_1", "type": "function",
                   "function": {"name": "read_file", "arguments": '{"path":"/tmp/f.py"}'}}]
    tool_results = [{"tool_call_id": "call_1", "content": "# contents"}]
    contrib = _make_contribution(tool_calls=tool_calls, tool_results=tool_results)
    result = build_sft_pair(contrib, _openai_recipe(), "coder",
                            system_prompt="sys", user_prompt="user")
    msgs = result["messages"]
    roles = [m["role"] for m in msgs]
    # system, user, assistant(tool_calls), tool, assistant(final)
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    tc_msg = msgs[2]
    assert tc_msg["tool_calls"] == tool_calls
    assert tc_msg["content"] is None
    tool_msg = msgs[3]
    assert tool_msg["tool_call_id"] == "call_1"
    assert tool_msg["content"] == "# contents"


def test_three_turn_tool_history_shape():
    """Canonical 3-turn (coder→tool_calls, tool, coder→final) history."""
    tool_calls = [{"id": "c1", "type": "function",
                   "function": {"name": "run_command", "arguments": '{"cmd":"ls"}'}}]
    tool_results = [{"tool_call_id": "c1", "content": "main.py\n"}]
    messages = [
        {"role": "system", "content": "You are a coder."},
        {"role": "user", "content": "List files."},
    ]
    contrib = _make_contribution(
        messages=messages,
        tool_calls=tool_calls,
        tool_results=tool_results,
        output="Here are the files: main.py",
    )
    result = build_sft_pair(contrib, _openai_recipe(), "coder")
    msgs = result["messages"]
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert msgs[2]["role"] == "assistant"
    assert msgs[2]["tool_calls"] == tool_calls
    assert msgs[3]["role"] == "tool"
    assert msgs[4]["role"] == "assistant"
    assert msgs[4]["content"] == "Here are the files: main.py"


def test_legacy_marker_format():
    contrib = _make_contribution()
    result = build_sft_pair(contrib, _legacy_recipe(), "coder",
                            system_prompt="sys", user_prompt="user")
    assert "prompt" in result and "completion" in result
    assert result["prompt"].startswith("[SYSTEM]\nsys")
    assert "[RESPONSE]" in result["prompt"]
    assert result["completion"] == "def add(a, b): return a + b"
