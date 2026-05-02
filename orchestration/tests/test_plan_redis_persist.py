"""Tests for Redis plan persistence (set_plan/get_plan) with fakeredis."""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestration.app.plans.schema import Plan, PlanStep
from datetime import datetime, timezone


def _make_plan() -> Plan:
    now = datetime.now(timezone.utc)
    return Plan(
        goal="Test task",
        steps=[
            PlanStep(id="step-1", description="Do the thing"),
            PlanStep(id="step-2", description="Test the thing"),
        ],
        created_at=now,
        updated_at=now,
    )


@pytest.fixture
def fake_redis_client():
    """In-memory fake Redis client for unit tests."""
    store: dict = {}
    ttls: dict = {}

    client = MagicMock()

    async def setex(key, ttl, value):
        store[key] = value
        ttls[key] = ttl

    async def get(key):
        return store.get(key)

    async def delete(key):
        store.pop(key, None)
        ttls.pop(key, None)

    client.setex = setex
    client.get = get
    client.delete = delete
    client._store = store
    client._ttls = ttls
    return client


@pytest.mark.asyncio
async def test_set_and_get_plan_round_trip(fake_redis_client):
    from orchestration.app.db import redis as _store
    _store._client = fake_redis_client
    _store._redis_ok = True

    plan = _make_plan()
    await _store.set_plan("task-abc", plan, ttl_s=3600)
    raw = await _store.get_plan("task-abc")

    assert raw is not None
    assert raw["goal"] == "Test task"
    assert len(raw["steps"]) == 2
    assert raw["steps"][0]["id"] == "step-1"


@pytest.mark.asyncio
async def test_set_plan_uses_correct_redis_key(fake_redis_client):
    from orchestration.app.db import redis as _store
    _store._client = fake_redis_client
    _store._redis_ok = True

    plan = _make_plan()
    await _store.set_plan("task-xyz", plan)

    assert "task_state:task-xyz" in fake_redis_client._store


@pytest.mark.asyncio
async def test_set_plan_applies_ttl(fake_redis_client):
    from orchestration.app.db import redis as _store
    _store._client = fake_redis_client
    _store._redis_ok = True

    plan = _make_plan()
    await _store.set_plan("task-ttl", plan, ttl_s=1234)

    assert fake_redis_client._ttls.get("task_state:task-ttl") == 1234


@pytest.mark.asyncio
async def test_get_plan_miss_returns_none(fake_redis_client):
    from orchestration.app.db import redis as _store
    _store._client = fake_redis_client
    _store._redis_ok = True

    result = await _store.get_plan("task-nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_ephemeral_path_does_not_call_set_plan():
    """When task_id == session_id (ephemeral), set_plan must NOT be called."""
    set_plan_mock = AsyncMock()
    with patch("orchestration.app.routes.run.session_store") as mock_store:
        mock_store.set_plan = set_plan_mock
        mock_store.get_plan = AsyncMock(return_value=None)
        mock_store.get_session = AsyncMock(return_value=[])
        mock_store.is_healthy = MagicMock(return_value=True)

        # Simulate ephemeral: task_id == session_id
        from orchestration.app.routes.run import _maybe_writeback_plan
        from orchestration.app.graph.state import AgentState

        now = datetime.now(timezone.utc).isoformat()
        plan_dict = {
            "goal": "Test",
            "steps": [{"id": "step-1", "description": "do", "status": "done",
                        "owner_role": None, "evidence": "", "files_touched": [],
                        "tools_used": [], "depends_on": []}],
            "created_at": now,
            "updated_at": now,
        }
        final_state: AgentState = {"plan": plan_dict, "plan_changed": True}
        initial_state: AgentState = {"task_id": "session-abc", "plan": None}

        _maybe_writeback_plan(
            final_state, initial_state,
            plan_ephemeral=True,
            plan_path_str="plans/current-task.md",
            session_id="session-abc",
            converged=True,
        )

    set_plan_mock.assert_not_called()
