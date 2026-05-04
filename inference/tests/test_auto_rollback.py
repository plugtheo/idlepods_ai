"""
Step 11 — test_auto_rollback.py

Simulate 5 fallback events within 60 s; assert /adapters/rollback is fired once.
"""
from __future__ import annotations

import time
from collections import deque
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import inference.app.backends.local_vllm as _mod


@pytest.fixture(autouse=True)
def _reset_fallback_counts():
    """Ensure module-level state is clean before each test."""
    _mod._adapter_fallback_counts.clear()
    yield
    _mod._adapter_fallback_counts.clear()


@pytest.mark.asyncio
async def test_auto_rollback_fires_after_threshold():
    """5 fallbacks within window → POST /adapters/rollback called exactly once."""
    rollback_calls = []

    mock_response = AsyncMock()
    mock_response.raise_for_status = MagicMock()
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(
        side_effect=lambda url, **kw: rollback_calls.append(url) or mock_response
    )

    mock_registry = AsyncMock()
    mock_registry.adapter_available = AsyncMock(return_value=False)

    with (
        patch.object(_mod.settings, "adapter_fallback_rollback_threshold", 5),
        patch.object(_mod.settings, "adapter_fallback_window_seconds", 60),
        patch("inference.app.backends.local_vllm.httpx.AsyncClient", return_value=mock_client),
    ):
        for _ in range(5):
            model, fallback = await _mod._resolve_model(
                model_id="base-model",
                adapter_name="coding_lora",
                registry=mock_registry,
                role="coder",
                base_url="http://localhost:8010",
            )
            assert model == "base-model"
            assert fallback is True

    assert len(rollback_calls) == 1, f"Expected 1 rollback call, got {len(rollback_calls)}"
    assert "/adapters/rollback" in rollback_calls[0]


@pytest.mark.asyncio
async def test_auto_rollback_not_fired_below_threshold():
    """4 fallbacks within window → no rollback."""
    rollback_calls = []

    mock_response = AsyncMock()
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(
        side_effect=lambda url, **kw: rollback_calls.append(url) or mock_response
    )

    mock_registry = AsyncMock()
    mock_registry.adapter_available = AsyncMock(return_value=False)

    with (
        patch.object(_mod.settings, "adapter_fallback_rollback_threshold", 5),
        patch.object(_mod.settings, "adapter_fallback_window_seconds", 60),
        patch("inference.app.backends.local_vllm.httpx.AsyncClient", return_value=mock_client),
    ):
        for _ in range(4):
            await _mod._resolve_model(
                model_id="base-model",
                adapter_name="coding_lora",
                registry=mock_registry,
                role="coder",
                base_url="http://localhost:8010",
            )

    assert rollback_calls == [], "No rollback expected below threshold"


@pytest.mark.asyncio
async def test_fallback_deque_cleared_after_rollback():
    """After rollback fires, the deque is cleared so the next event restarts count."""
    rollback_calls = []

    mock_response = AsyncMock()
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(
        side_effect=lambda url, **kw: rollback_calls.append(url) or mock_response
    )

    mock_registry = AsyncMock()
    mock_registry.adapter_available = AsyncMock(return_value=False)

    with (
        patch.object(_mod.settings, "adapter_fallback_rollback_threshold", 5),
        patch.object(_mod.settings, "adapter_fallback_window_seconds", 60),
        patch("inference.app.backends.local_vllm.httpx.AsyncClient", return_value=mock_client),
    ):
        # Trigger rollback
        for _ in range(5):
            await _mod._resolve_model("base", "coding_lora", mock_registry, "coder", "http://localhost:8010")

        assert len(rollback_calls) == 1
        assert len(_mod._adapter_fallback_counts.get("coder", deque())) == 0

        # 4 more fallbacks should not fire another rollback
        for _ in range(4):
            await _mod._resolve_model("base", "coding_lora", mock_registry, "coder", "http://localhost:8010")

        assert len(rollback_calls) == 1
