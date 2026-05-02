"""Tests for planner_node plan creation/progression and plan-step injection."""
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestration.app.graph.nodes import planner_node, coder_node, _try_parse_plan_output
from orchestration.app.graph.state import AgentState
from orchestration.app.plans.schema import Plan, PlanStep
from shared.contracts.agent_prompts import PLAN_STEP_SYSTEM_TEMPLATE


def _make_state(**overrides) -> AgentState:
    base: AgentState = {
        "session_id": "test-session",
        "task_id": "test-task",
        "user_prompt": "Fix the auth bug",
        "agent_chain": ["planner", "coder"],
        "agent_chain_index": 0,
        "allowed_files": [],
        "file_fingerprints": {},
        "few_shots": [],
        "repo_snippets": [],
        "system_hints": "",
        "current_iteration": 1,
        "max_iterations": 2,
        "convergence_threshold": 0.85,
        "conversation_history": [],
        "iteration_history": [],
        "last_output": "",
        "iteration_scores": [],
        "best_score": 0.0,
        "best_output": "",
        "converged": False,
        "quality_converged": False,
        "final_output": "",
        "final_score": 0.0,
        "plan": None,
        "plan_changed": False,
        "current_step_id": None,
    }
    base.update(overrides)
    return base


_PLAN_JSON = json.dumps({
    "goal": "Fix the auth bug",
    "steps": [
        {"id": "step-1", "description": "Investigate logs", "owner_role": "coder"},
        {"id": "step-2", "description": "Apply fix", "owner_role": "coder"},
    ],
})


@pytest.mark.asyncio
async def test_planner_creates_plan_from_llm_output():
    """When state.plan is None, planner runs LLM and parses JSON into plan."""
    plan_output = f"Here is the plan:\n```json\n{_PLAN_JSON}\n```\nLet me explain each step."
    mock_resp = MagicMock()
    mock_resp.content = plan_output
    mock_resp.tokens_generated = 50
    mock_resp.tool_calls = None

    with patch("orchestration.app.graph.nodes.get_inference_client") as mock_client_fn:
        mock_client = MagicMock()
        mock_client.generate = AsyncMock(return_value=mock_resp)
        mock_client_fn.return_value = mock_client

        state = _make_state()
        delta = await planner_node(state)

    assert delta.get("plan") is not None, "plan must be set after first planner call"
    assert delta.get("plan_changed") is True
    steps = delta["plan"]["steps"]
    assert len(steps) == 2
    assert steps[0]["id"] == "step-1"


@pytest.mark.asyncio
async def test_planner_does_not_regenerate_existing_plan():
    """When state.plan already exists, planner marks next step in_progress without LLM call."""
    now = datetime.now(timezone.utc).isoformat()
    existing_plan = {
        "goal": "Fix the auth bug",
        "steps": [
            {"id": "step-1", "description": "Investigate logs", "status": "pending",
             "owner_role": "coder", "evidence": "", "files_touched": [], "tools_used": [], "depends_on": []},
            {"id": "step-2", "description": "Apply fix", "status": "pending",
             "owner_role": "coder", "evidence": "", "files_touched": [], "tools_used": [], "depends_on": []},
        ],
        "created_at": now,
        "updated_at": now,
    }

    with patch("orchestration.app.graph.nodes.get_inference_client") as mock_client_fn:
        mock_client = MagicMock()
        mock_client.generate = AsyncMock(side_effect=AssertionError("LLM must not be called"))
        mock_client_fn.return_value = mock_client

        state = _make_state(plan=existing_plan)
        delta = await planner_node(state)

    assert delta.get("current_step_id") == "step-1"
    assert delta.get("plan_changed") is True
    assert delta["plan"]["steps"][0]["status"] == "in_progress"


@pytest.mark.asyncio
async def test_plan_step_injection_is_extra_system_message():
    """Plan-step context is injected as an extra system message, not concatenated."""
    from orchestration.app.graph.nodes import _build_messages
    from shared.contracts.agent_prompts import AGENT_PROMPTS

    now = datetime.now(timezone.utc).isoformat()
    plan = {
        "goal": "Fix auth bug",
        "steps": [
            {"id": "step-1", "description": "Write the fix", "status": "in_progress",
             "owner_role": "coder", "evidence": "", "files_touched": [], "tools_used": [], "depends_on": []},
        ],
        "created_at": now,
        "updated_at": now,
    }
    state = _make_state(plan=plan, current_step_id="step-1")
    messages = _build_messages("coder", state)

    system_messages = [m for m in messages if m.role == "system"]
    assert len(system_messages) >= 2, "Must have at least 2 system messages: base + plan-step"

    base_prompt = AGENT_PROMPTS["coder"]
    assert system_messages[0].content == base_prompt or base_prompt in (system_messages[0].content or ""), \
        "First system message must be the unmodified base coder prompt"

    plan_step_msg = system_messages[1]
    assert "step-1" in (plan_step_msg.content or "")
    assert "Write the fix" in (plan_step_msg.content or "")

    # Base prompt must be byte-for-byte unchanged
    assert system_messages[0].content != plan_step_msg.content


def test_try_parse_plan_output_json_fence():
    output = f'Here is the plan:\n```json\n{_PLAN_JSON}\n```\nMore text.'
    result = _try_parse_plan_output(output)
    assert result is not None
    assert result["goal"] == "Fix the auth bug"
    assert len(result["steps"]) == 2


def test_try_parse_plan_output_returns_none_on_garbage():
    result = _try_parse_plan_output("No JSON here at all.")
    assert result is None
