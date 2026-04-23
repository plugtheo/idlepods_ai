"""
Tests for the JSONL experience store.

Uses tmp_path to avoid touching real data files.
"""
import json
import pytest
from unittest.mock import patch

from shared.contracts.experience import ExperienceEvent, AgentContribution


def _make_event(session_id="s1", prompt="test prompt", score=0.8):
    return ExperienceEvent(
        session_id=session_id,
        prompt=prompt,
        final_output="output",
        agent_chain=["coder", "reviewer"],
        contributions=[
            AgentContribution(role="coder", output="code", quality_score=score, iteration=1),
        ],
        final_score=score,
        iterations=1,
        converged=True,
    )


@pytest.mark.asyncio
class TestJSONLStore:
    async def test_append_creates_file(self, tmp_path):
        jsonl_file = tmp_path / "experiences.jsonl"

        mock_settings = type("S", (), {"jsonl_path": str(jsonl_file)})()
        with patch("services.experience.app.storage.jsonl_store.settings", mock_settings):
            from services.experience.app.storage import jsonl_store
            jsonl_store._lock = __import__("asyncio").Lock()

            await jsonl_store.append_experience(_make_event())

        assert jsonl_file.exists()

    async def test_count_returns_number_of_lines(self, tmp_path):
        jsonl_file = tmp_path / "experiences.jsonl"

        mock_settings = type("S", (), {"jsonl_path": str(jsonl_file)})()
        with patch("services.experience.app.storage.jsonl_store.settings", mock_settings):
            import asyncio
            from services.experience.app.storage import jsonl_store
            jsonl_store._lock = asyncio.Lock()

            await jsonl_store.append_experience(_make_event("s1"))
            await jsonl_store.append_experience(_make_event("s2"))
            count = await jsonl_store.count_experiences()

        assert count == 2

    async def test_count_returns_zero_if_file_missing(self, tmp_path):
        jsonl_file = tmp_path / "no_such_file.jsonl"

        mock_settings = type("S", (), {"jsonl_path": str(jsonl_file)})()
        with patch("services.experience.app.storage.jsonl_store.settings", mock_settings):
            import asyncio
            from services.experience.app.storage import jsonl_store
            jsonl_store._lock = asyncio.Lock()

            count = await jsonl_store.count_experiences()

        assert count == 0

    async def test_stored_data_is_valid_json(self, tmp_path):
        jsonl_file = tmp_path / "experiences.jsonl"

        mock_settings = type("S", (), {"jsonl_path": str(jsonl_file)})()
        with patch("services.experience.app.storage.jsonl_store.settings", mock_settings):
            import asyncio
            from services.experience.app.storage import jsonl_store
            jsonl_store._lock = asyncio.Lock()

            await jsonl_store.append_experience(_make_event("s1", "do something"))

        with jsonl_file.open() as f:
            line = f.readline()
        data = json.loads(line)
        assert data["session_id"] == "s1"
        assert data["prompt"] == "do something"
