"""
Tests for LangGraph pipeline nodes.

The Inference Service client is mocked — no real HTTP calls.
We verify that:
- Each node calls the inference client with correct role/model_family
- The returned state delta contains the new history entry
- System prompt is included in messages
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from shared.contracts.inference import GenerateResponse


def _base_state(**overrides):
    state = {
        "session_id": "test-sess",
        "user_prompt": "write a binary search function",
        "agent_chain": ["coder", "reviewer"],
        "agent_chain_index": 1,
        "few_shots": [],
        "repo_snippets": [],
        "system_hints": "use type hints",
        "current_iteration": 1,
        "max_iterations": 5,
        "convergence_threshold": 0.85,
        "iteration_history": [],
        "conversation_history": [],
        "last_output": "",
        "iteration_scores": [],
        "best_score": 0.0,
        "best_output": "",
        "converged": False,
        "quality_converged": False,
        "final_output": "",
        "final_score": 0.0,
        "pending_tool_calls": [],
        "tool_steps_used": 0,
        "tool_originating_role": "",
    }
    state.update(overrides)
    return state


def _mock_inference_client(content="generated code"):
    client = MagicMock()
    client.generate = AsyncMock(
        return_value=GenerateResponse(
            content=content,
            model_family="qwen",
            role="coder",
            tokens_generated=50,
        )
    )
    return client


@pytest.mark.asyncio
class TestCoreAgentNode:
    async def test_coder_node_returns_state_delta(self):
        from services.orchestration.app.graph.nodes import coder_node

        mock_client = _mock_inference_client("def binary_search(): pass")
        with patch(
            "services.orchestration.app.graph.nodes.get_inference_client",
            return_value=mock_client,
        ):
            delta = await coder_node(_base_state())

        assert "iteration_history" in delta
        history = delta["iteration_history"]
        assert len(history) == 1
        entry = history[0]
        assert entry["role"] == "coder"
        assert entry["output"] == "[VALIDATOR_FAIL:short_text]"
        assert entry["iteration"] == 1

    async def test_planner_node_returns_state_delta(self):
        from services.orchestration.app.graph.nodes import planner_node

        mock_client = _mock_inference_client("Plan: 1. Define 2. Implement")
        with patch(
            "services.orchestration.app.graph.nodes.get_inference_client",
            return_value=mock_client,
        ):
            delta = await planner_node(_base_state())

        history = delta["iteration_history"]
        assert history[0]["role"] == "planner"

    async def test_reviewer_node_returns_state_delta(self):
        from services.orchestration.app.graph.nodes import reviewer_node

        mock_client = _mock_inference_client("SCORE: 0.85 looks good")
        with patch(
            "services.orchestration.app.graph.nodes.get_inference_client",
            return_value=mock_client,
        ):
            delta = await reviewer_node(_base_state())

        history = delta["iteration_history"]
        assert history[0]["role"] == "reviewer"

    async def test_inference_called_with_correct_role(self):
        from services.orchestration.app.graph.nodes import debugger_node

        mock_client = _mock_inference_client()
        with patch(
            "services.orchestration.app.graph.nodes.get_inference_client",
            return_value=mock_client,
        ):
            await debugger_node(_base_state())

        call_args = mock_client.generate.call_args[0][0]
        assert call_args.role == "debugger"

    async def test_system_hints_included_in_messages(self):
        from services.orchestration.app.graph.nodes import coder_node

        mock_client = _mock_inference_client()
        with patch(
            "services.orchestration.app.graph.nodes.get_inference_client",
            return_value=mock_client,
        ):
            await coder_node(_base_state(system_hints="use type hints"))

        req = mock_client.generate.call_args[0][0]
        system_content = req.messages[0].content
        assert "type hints" in system_content

    async def test_prior_history_included_for_later_nodes(self):
        from services.orchestration.app.graph.nodes import reviewer_node

        prior_history = [
            {"iteration": 1, "role": "coder", "output": "def f(): pass", "timestamp": "now"}
        ]
        mock_client = _mock_inference_client("SCORE: 0.70")
        with patch(
            "services.orchestration.app.graph.nodes.get_inference_client",
            return_value=mock_client,
        ):
            delta = await reviewer_node(_base_state(iteration_history=prior_history))

        # The new history should contain both the prior entry and the new one
        history = delta["iteration_history"]
        assert len(history) == 2
        assert history[0]["role"] == "coder"
        assert history[1]["role"] == "reviewer"
