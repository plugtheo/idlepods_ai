"""
Step 11 — test_jsonl_rotation.py

Monkeypatch _utcnow to advance one day mid-write; assert two shards exist
with correct contents.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from shared.contracts.experience import ExperienceEvent


def _make_event(session_id: str, prompt: str = "test") -> ExperienceEvent:
    return ExperienceEvent(
        session_id=session_id,
        prompt=prompt,
        final_output="output",
        agent_chain=["coder"],
        contributions=[],
        final_score=0.8,
        iterations=1,
        converged=True,
        iteration_scores=[0.8],
        intent="code",
        complexity="medium",
        timestamp=datetime.now(tz=timezone.utc),
        scorer_rule_version="v1",
    )


@pytest.fixture()
def patched_jsonl_dir(tmp_path):
    """Patch orchestration settings to use tmp_path as jsonl_dir."""
    with patch("orchestration.app.experience.jsonl_store.settings") as mock_settings:
        mock_settings.jsonl_dir = str(tmp_path)
        mock_settings.jsonl_path = str(tmp_path / "experiences.jsonl")
        yield tmp_path


@pytest.mark.asyncio
async def test_daily_rotation_creates_two_shards(patched_jsonl_dir):
    """Two writes on different UTC dates create two separate shard files."""
    tmp_path = patched_jsonl_dir

    import orchestration.app.experience.jsonl_store as store

    day1 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    day2 = datetime(2026, 1, 2, 12, 0, 0, tzinfo=timezone.utc)

    with patch.object(store, "_utcnow", return_value=day1):
        await store.append(_make_event("session-aaa"))

    with patch.object(store, "_utcnow", return_value=day2):
        await store.append(_make_event("session-bbb"))

    shards = sorted(tmp_path.glob("experiences-*.jsonl"))
    assert len(shards) == 2, f"Expected 2 shards, found: {[s.name for s in shards]}"
    assert shards[0].name == "experiences-20260101.jsonl"
    assert shards[1].name == "experiences-20260102.jsonl"

    records_day1 = [json.loads(l) for l in shards[0].read_text().splitlines() if l.strip()]
    records_day2 = [json.loads(l) for l in shards[1].read_text().splitlines() if l.strip()]

    assert len(records_day1) == 1
    assert records_day1[0]["session_id"] == "session-aaa"

    assert len(records_day2) == 1
    assert records_day2[0]["session_id"] == "session-bbb"


@pytest.mark.asyncio
async def test_same_day_writes_go_to_same_shard(patched_jsonl_dir):
    """Multiple writes on the same day append to a single shard."""
    tmp_path = patched_jsonl_dir

    import orchestration.app.experience.jsonl_store as store

    day = datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

    with patch.object(store, "_utcnow", return_value=day):
        await store.append(_make_event("session-x"))
        await store.append(_make_event("session-y"))
        await store.append(_make_event("session-z"))

    shards = sorted(tmp_path.glob("experiences-*.jsonl"))
    assert len(shards) == 1
    records = [json.loads(l) for l in shards[0].read_text().splitlines() if l.strip()]
    assert len(records) == 3
