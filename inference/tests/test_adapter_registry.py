"""
Tests for _AdapterRegistry — TTL caching, refresh logic, fallback on failure.
"""
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
class TestAdapterRegistry:
    async def test_adapter_available_when_registered(self):
        """If /v1/models lists the adapter, adapter_available() returns True."""
        from services.inference.app.backends.local_vllm import _AdapterRegistry

        registry = _AdapterRegistry(
            base_url="http://fake-vllm:8000",
            base_model_id="deepseek-ai/deepseek-coder-6.7b-instruct",
            ttl=120,
        )

        model_list_resp = {
            "data": [
                {"id": "deepseek-ai/deepseek-coder-6.7b-instruct"},
                {"id": "deepseek-ai/deepseek-coder-6.7b-instruct/coding_lora"},
            ]
        }

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = MagicMock(return_value=model_list_resp)

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)

        # Inject the mock client directly — the registry is already constructed.
        registry._client = mock_client
        result = await registry.adapter_available("coding_lora")

        assert result is True

    async def test_adapter_not_available_when_not_listed(self):
        from services.inference.app.backends.local_vllm import _AdapterRegistry

        registry = _AdapterRegistry(
            base_url="http://fake-vllm:8000",
            base_model_id="deepseek-ai/deepseek-coder-6.7b-instruct",
            ttl=120,
        )

        model_list_resp = {
            "data": [
                {"id": "deepseek-ai/deepseek-coder-6.7b-instruct"},
            ]
        }

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = MagicMock(return_value=model_list_resp)

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)

        registry._client = mock_client
        result = await registry.adapter_available("coding_lora")

        assert result is False

    async def test_uses_stale_cache_on_network_failure(self):
        """If /v1/models fails, the previous cache is kept (non-fatal)."""
        from services.inference.app.backends.local_vllm import _AdapterRegistry

        registry = _AdapterRegistry(
            base_url="http://fake-vllm:8000",
            base_model_id="deepseek-ai/deepseek-coder-6.7b-instruct",
            ttl=120,
        )
        # Pre-populate cache as if a prior fetch succeeded (qualified form).
        registry._known = {"deepseek-ai/deepseek-coder-6.7b-instruct/coding_lora"}
        registry._fetched_at = 0.0  # Force stale so it tries to refresh

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=Exception("connection refused"))

        # Inject mock — refresh will fail, stale cache must survive.
        registry._client = mock_client
        result = await registry.adapter_available("coding_lora")

        assert result is True

    async def test_cache_not_refreshed_within_ttl(self):
        """Within TTL, no new HTTP request is made."""
        from services.inference.app.backends.local_vllm import _AdapterRegistry

        registry = _AdapterRegistry(
            base_url="http://fake-vllm:8000",
            base_model_id="deepseek-ai/deepseek-coder-6.7b-instruct",
            ttl=120,
        )
        # Mark cache as recently fetched
        registry._known = set()
        registry._fetched_at = time.monotonic()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=AssertionError("should not be called"))

        registry._client = mock_client
        result = await registry.adapter_available("coding_lora")

        assert result is False  # not in empty cache, but no HTTP call made
