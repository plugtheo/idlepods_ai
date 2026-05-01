"""
Tests for the backend factory (get_backend(model_family)).

Covers:
- qwen with local_vllm backend → LocalVLLMBackend
- qwen with remote_vllm backend → RemoteVLLMBackend
- Unknown model_family raises ValueError
- Calling get_backend() twice for same family returns same instance
- Family name is case-insensitive
"""
import pytest
from unittest.mock import MagicMock, patch


def _make_settings(qwen_backend="local_vllm"):
    s = MagicMock()
    s.qwen_backend = qwen_backend
    s.qwen_url = "http://vllm-qwen:8000"
    s.qwen_model_id = "Qwen/Qwen3-14B"
    s.qwen_auth_token = ""
    s.qwen_ssl_verify = True
    s.request_timeout_seconds = 120.0
    return s


class TestBackendFactory:
    def setup_method(self):
        """Reset per-family singletons before each test."""
        import services.inference.app.backends.factory as factory_mod
        factory_mod._backends.clear()

    def test_qwen_local_returns_local_vllm_backend(self):
        with patch("services.inference.app.backends.factory.settings", _make_settings()):
            from services.inference.app.backends.factory import get_backend
            from services.inference.app.backends.local_vllm import LocalVLLMBackend

            backend = get_backend("qwen")
            assert isinstance(backend, LocalVLLMBackend)

    def test_qwen_remote_returns_remote_vllm_backend(self):
        with patch(
            "services.inference.app.backends.factory.settings",
            _make_settings(qwen_backend="remote_vllm"),
        ):
            from services.inference.app.backends.factory import get_backend
            from services.inference.app.backends.remote_vllm import RemoteVLLMBackend

            backend = get_backend("qwen")
            assert isinstance(backend, RemoteVLLMBackend)

    def test_unknown_family_raises(self):
        with patch("services.inference.app.backends.factory.settings", _make_settings()):
            from services.inference.app.backends.factory import get_backend

            with pytest.raises(ValueError, match="Unknown model_family"):
                get_backend("llama")

    def test_family_name_case_insensitive(self):
        with patch("services.inference.app.backends.factory.settings", _make_settings()):
            from services.inference.app.backends.factory import get_backend
            from services.inference.app.backends.local_vllm import LocalVLLMBackend

            backend = get_backend("Qwen")
            assert isinstance(backend, LocalVLLMBackend)

    def test_singleton_per_family(self):
        with patch("services.inference.app.backends.factory.settings", _make_settings()):
            from services.inference.app.backends.factory import get_backend

            b1 = get_backend("qwen")
            b2 = get_backend("qwen")
            assert b1 is b2
