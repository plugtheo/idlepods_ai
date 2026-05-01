"""
Tests for the backend factory (get_backend(backend_name)).

Covers:
- primary with local_vllm backend_type → LocalVLLMBackend
- primary with remote_vllm backend_type → RemoteVLLMBackend
- Unknown backend name raises ValueError
- Calling get_backend() twice for same name returns same instance (singleton)
"""
import pytest
from unittest.mock import MagicMock, patch

from shared.contracts.models import BackendEntry, ModelsRegistry


def _make_registry(backend_type="local_vllm"):
    entry = BackendEntry(
        served_url="http://vllm-primary:8000",
        model_id="Qwen/Qwen3-14B",
        backend_type=backend_type,
    )
    return ModelsRegistry(default_backend="primary", backends={"primary": entry})


class TestBackendFactory:
    def setup_method(self):
        """Reset per-name singletons before each test."""
        import services.inference.app.backends.factory as factory_mod
        factory_mod._backends.clear()

    def test_primary_local_returns_local_vllm_backend(self):
        with patch(
            "services.inference.app.backends.factory.get_backend_entry",
            return_value=_make_registry().backends["primary"],
        ):
            from services.inference.app.backends.factory import get_backend
            from services.inference.app.backends.local_vllm import LocalVLLMBackend

            backend = get_backend("primary")
            assert isinstance(backend, LocalVLLMBackend)

    def test_primary_remote_returns_remote_vllm_backend(self):
        with patch(
            "services.inference.app.backends.factory.get_backend_entry",
            return_value=_make_registry(backend_type="remote_vllm").backends["primary"],
        ):
            from services.inference.app.backends.factory import get_backend
            from services.inference.app.backends.remote_vllm import RemoteVLLMBackend

            backend = get_backend("primary")
            assert isinstance(backend, RemoteVLLMBackend)

    def test_unknown_backend_raises(self):
        from services.inference.app.backends.factory import get_backend

        with pytest.raises(ValueError):
            get_backend("nonexistent_backend_xyz")

    def test_singleton_per_name(self):
        with patch(
            "services.inference.app.backends.factory.get_backend_entry",
            return_value=_make_registry().backends["primary"],
        ):
            from services.inference.app.backends.factory import get_backend

            b1 = get_backend("primary")
            b2 = get_backend("primary")
            assert b1 is b2
