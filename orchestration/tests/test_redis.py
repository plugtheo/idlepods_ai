"""Unit tests for orchestration/app/db/redis.py — session store."""
import json
import pytest
from unittest.mock import AsyncMock, patch


def _patch_client(mock_client):
    return patch(
        "services.orchestration.app.db.redis._get_client",
        return_value=mock_client,
    )


class TestGetSession:
    @pytest.mark.asyncio
    async def test_returns_parsed_history(self):
        from services.orchestration.app.db.redis import get_session
        history = [{"role": "coder", "output": "x = 1"}]
        client = AsyncMock()
        client.get.return_value = json.dumps(history)

        with _patch_client(client):
            result = await get_session("task-abc")

        client.get.assert_called_once_with("session:task-abc")
        assert result == history

    @pytest.mark.asyncio
    async def test_returns_empty_on_miss(self):
        from services.orchestration.app.db.redis import get_session
        client = AsyncMock()
        client.get.return_value = None

        with _patch_client(client):
            result = await get_session("task-miss")

        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_on_error(self):
        from services.orchestration.app.db.redis import get_session
        client = AsyncMock()
        client.get.side_effect = ConnectionError("redis down")

        with _patch_client(client):
            result = await get_session("task-err")

        assert result == []


class TestSaveSession:
    @pytest.mark.asyncio
    async def test_calls_setex_with_correct_args(self):
        from services.orchestration.app.db.redis import save_session
        history = [{"role": "planner", "output": "plan"}]
        client = AsyncMock()

        with _patch_client(client):
            await save_session("task-xyz", history, ttl=600)

        client.setex.assert_called_once_with(
            "session:task-xyz", 600, json.dumps(history)
        )

    @pytest.mark.asyncio
    async def test_swallows_exceptions(self):
        from services.orchestration.app.db.redis import save_session
        client = AsyncMock()
        client.setex.side_effect = ConnectionError("redis down")

        with _patch_client(client):
            await save_session("task-err", [], ttl=3600)
        # no exception raised — graceful degradation


class TestGetFingerprints:
    @pytest.mark.asyncio
    async def test_returns_parsed_fps(self):
        from services.orchestration.app.db.redis import get_fingerprints
        fps = {"src/main.py": "abc123"}
        client = AsyncMock()
        client.get.return_value = json.dumps(fps)

        with _patch_client(client):
            result = await get_fingerprints("task-1")

        client.get.assert_called_once_with("fps:v2:task-1")
        assert result == fps

    @pytest.mark.asyncio
    async def test_returns_none_on_miss(self):
        from services.orchestration.app.db.redis import get_fingerprints
        client = AsyncMock()
        client.get.return_value = None

        with _patch_client(client):
            result = await get_fingerprints("task-miss")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_error(self):
        from services.orchestration.app.db.redis import get_fingerprints
        client = AsyncMock()
        client.get.side_effect = ConnectionError("redis down")

        with _patch_client(client):
            result = await get_fingerprints("task-err")

        assert result is None


class TestSaveFingerprints:
    @pytest.mark.asyncio
    async def test_calls_setex_with_correct_args(self):
        from services.orchestration.app.db.redis import save_fingerprints
        fps = {"src/main.py": "abc123"}
        client = AsyncMock()

        with _patch_client(client):
            await save_fingerprints("task-1", fps, ttl=3600)

        client.setex.assert_called_once_with("fps:v2:task-1", 3600, json.dumps(fps))

    @pytest.mark.asyncio
    async def test_swallows_exceptions(self):
        from services.orchestration.app.db.redis import save_fingerprints
        client = AsyncMock()
        client.setex.side_effect = ConnectionError("redis down")

        with _patch_client(client):
            await save_fingerprints("task-err", {}, ttl=3600)


class TestGetSnippets:
    @pytest.mark.asyncio
    async def test_returns_parsed_snippets(self):
        from services.orchestration.app.db.redis import get_snippets
        snippets = [{"file": "src/main.py", "snippet": "def foo(): pass"}]
        client = AsyncMock()
        client.get.return_value = json.dumps(snippets)

        with _patch_client(client):
            result = await get_snippets("task-1")

        client.get.assert_called_once_with("snippets:v2:task-1")
        assert result == snippets

    @pytest.mark.asyncio
    async def test_returns_empty_on_miss(self):
        from services.orchestration.app.db.redis import get_snippets
        client = AsyncMock()
        client.get.return_value = None

        with _patch_client(client):
            result = await get_snippets("task-miss")

        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_on_error(self):
        from services.orchestration.app.db.redis import get_snippets
        client = AsyncMock()
        client.get.side_effect = ConnectionError("redis down")

        with _patch_client(client):
            result = await get_snippets("task-err")

        assert result == []


class TestSaveSnippets:
    @pytest.mark.asyncio
    async def test_calls_setex_with_correct_args(self):
        from services.orchestration.app.db.redis import save_snippets
        snippets = [{"file": "src/main.py", "snippet": "def foo(): pass"}]
        client = AsyncMock()

        with _patch_client(client):
            await save_snippets("task-1", snippets, ttl=3600)

        client.setex.assert_called_once_with("snippets:v2:task-1", 3600, json.dumps(snippets))

    @pytest.mark.asyncio
    async def test_swallows_exceptions(self):
        from services.orchestration.app.db.redis import save_snippets
        client = AsyncMock()
        client.setex.side_effect = ConnectionError("redis down")

        with _patch_client(client):
            await save_snippets("task-err", [], ttl=3600)
