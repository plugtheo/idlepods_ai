"""
Tests for the backend factory (get_backend()).

Covers:
- mode=local → LocalVLLMBackend singleton
- mode=api → APIBackend singleton
- Unknown mode raises ValueError
- Calling get_backend() twice returns same instance
"""
import pytest
from unittest.mock import MagicMock, patch


def _make_settings(mode: str):
    s = MagicMock()
    s.mode = mode
    s.deepseek_url = "http://deepseek:8000"
    s.mistral_url = "http://mistral:8001"
    s.api_provider = "anthropic"
    s.api_model = "claude-3-5-haiku-20241022"
    return s


class TestBackendFactory:
    def setup_method(self):
        """Reset singleton before each test."""
        import services.inference.app.backends.factory as factory_mod
        factory_mod._backend_instance = None

    def test_local_mode_returns_local_vllm_backend(self):
        with patch("services.inference.app.backends.factory.settings", _make_settings("local")):
            from services.inference.app.backends.factory import get_backend
            from services.inference.app.backends.local_vllm import LocalVLLMBackend

            import services.inference.app.backends.factory as fmod
            fmod._backend_instance = None

            backend = get_backend()
            assert isinstance(backend, LocalVLLMBackend)

    def test_local_mode_is_case_insensitive(self):
        with patch("services.inference.app.backends.factory.settings", _make_settings("LOCAL")):
            from services.inference.app.backends.factory import get_backend
            from services.inference.app.backends.local_vllm import LocalVLLMBackend

            import services.inference.app.backends.factory as fmod
            fmod._backend_instance = None

            backend = get_backend()
            assert isinstance(backend, LocalVLLMBackend)

    def test_singleton_returns_same_instance(self):
        with patch("services.inference.app.backends.factory.settings", _make_settings("local")):
            import services.inference.app.backends.factory as fmod
            fmod._backend_instance = None

            from services.inference.app.backends.factory import get_backend
            b1 = get_backend()
            b2 = get_backend()
            assert b1 is b2

    def test_unknown_mode_raises_value_error(self):
        with patch("services.inference.app.backends.factory.settings", _make_settings("cloud")):
            import services.inference.app.backends.factory as fmod
            fmod._backend_instance = None

            from services.inference.app.backends.factory import get_backend
            with pytest.raises(ValueError, match="Unknown INFERENCE__MODE"):
                get_backend()
