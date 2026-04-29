"""
Tests for the backend factory (get_backend(model_family)).

Covers:
- deepseek/mistral with local_vllm backend → LocalVLLMBackend
- deepseek/mistral with remote_vllm backend → RemoteVLLMBackend
- Unknown model_family raises ValueError
- Calling get_backend() twice for same family returns same instance
"""
import pytest
from unittest.mock import MagicMock, patch


def _make_settings(deepseek_backend="local_vllm", mistral_backend="local_vllm"):
    s = MagicMock()
    s.deepseek_backend = deepseek_backend
    s.deepseek_url = "http://deepseek:8000"
    s.deepseek_model_id = "deepseek-ai/deepseek-coder-6.7b-instruct"
    s.deepseek_auth_token = ""
    s.deepseek_ssl_verify = True
    s.mistral_backend = mistral_backend
    s.mistral_url = "http://mistral:8001"
    s.mistral_model_id = "mistralai/Mistral-7B-Instruct-v0.1"
    s.mistral_auth_token = ""
    s.mistral_ssl_verify = True
    s.request_timeout_seconds = 120.0
    return s


class TestBackendFactory:
    def setup_method(self):
        """Reset per-family singletons before each test."""
        import services.inference.app.backends.factory as factory_mod
        factory_mod._backends.clear()

    def test_deepseek_local_returns_local_vllm_backend(self):
        with patch("services.inference.app.backends.factory.settings", _make_settings()):
            from services.inference.app.backends.factory import get_backend
            from services.inference.app.backends.local_vllm import LocalVLLMBackend

            backend = get_backend("deepseek")
            assert isinstance(backend, LocalVLLMBackend)

    def test_mistral_local_returns_local_vllm_backend(self):
        with patch("services.inference.app.backends.factory.settings", _make_settings()):
            from services.inference.app.backends.factory import get_backend
            from services.inference.app.backends.local_vllm import LocalVLLMBackend

            backend = get_backend("mistral")
            assert isinstance(backend, LocalVLLMBackend)

    def test_remote_vllm_backend_selected(self):
        with patch(
            "services.inference.app.backends.factory.settings",
            _make_settings(mistral_backend="remote_vllm"),
        ):
            from services.inference.app.backends.factory import get_backend
            from services.inference.app.backends.remote_vllm import RemoteVLLMBackend

            backend = get_backend("mistral")
            assert isinstance(backend, RemoteVLLMBackend)

    def test_unknown_family_raises(self):
        with patch("services.inference.app.backends.factory.settings", _make_settings()):
            from services.inference.app.backends.factory import get_backend

            with pytest.raises(ValueError, match="Unknown model_family"):
                get_backend("gpt4")

    def test_family_name_case_insensitive(self):
        with patch("services.inference.app.backends.factory.settings", _make_settings()):
            from services.inference.app.backends.factory import get_backend
            from services.inference.app.backends.local_vllm import LocalVLLMBackend

            backend = get_backend("DeepSeek")
            assert isinstance(backend, LocalVLLMBackend)

    def test_singleton_per_family(self):
        with patch("services.inference.app.backends.factory.settings", _make_settings()):
            from services.inference.app.backends.factory import get_backend

            b1 = get_backend("deepseek")
            b2 = get_backend("deepseek")
            assert b1 is b2

    def test_separate_singletons_per_family(self):
        with patch("services.inference.app.backends.factory.settings", _make_settings()):
            from services.inference.app.backends.factory import get_backend

            deepseek = get_backend("deepseek")
            mistral = get_backend("mistral")
            assert deepseek is not mistral
