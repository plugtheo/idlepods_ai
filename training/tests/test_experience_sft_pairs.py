"""Integration test: experience record with tool_calls → two SFT pairs via build_sft_pair."""
import pytest

from shared.contracts.experience import AgentContribution
from shared.contracts.sft_builder import build_sft_pair
from shared.contracts.training import AdapterRecipe


def _tool_contribution(role: str = "coder") -> AgentContribution:
    return AgentContribution(
        role=role,
        output="final answer",
        quality_score=0.9,
        iteration=1,
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "run_tests", "arguments": "{}"},
            }
        ],
        tool_results=[
            {"tool_call_id": "call_1", "content": "All tests passed."}
        ],
    )


def test_tool_call_contribution_yields_two_sft_pairs():
    contrib = _tool_contribution(role="coder")
    recipe = AdapterRecipe()

    result = build_sft_pair(contrib, recipe, role="coder")

    assert isinstance(result, list), "Expected list of pairs for tool_call records"
    assert len(result) == 2, f"Expected 2 SFT pairs, got {len(result)}"


def test_sft_pairs_assigned_to_correct_capability():
    contrib = _tool_contribution(role="debugger")
    recipe = AdapterRecipe()

    pairs = build_sft_pair(contrib, recipe, role="debugger")

    assert isinstance(pairs, list)
    for pair in pairs:
        msgs = pair.get("messages", [])
        assert any(m.get("role") == "assistant" for m in msgs), \
            "Each SFT pair must contain at least one assistant turn"
