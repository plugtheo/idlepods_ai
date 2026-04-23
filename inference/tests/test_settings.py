"""
Tests for InferenceSettings — env var parsing and defaults.
"""
import pytest
import os
from unittest.mock import patch


class TestInferenceSettings:
    def test_defaults(self):
        """Default mode is 'local' and URLs point to docker service names."""
        with patch.dict(os.environ, {}, clear=False):
            # Re-import to get fresh settings (env-prefixed with INFERENCE__)
            from services.inference.app.config.settings import InferenceSettings
            s = InferenceSettings()
            assert s.mode == "local"
            assert "deepseek" in s.deepseek_url
            assert "mistral" in s.mistral_url

    def test_mode_override(self):
        with patch.dict(os.environ, {"INFERENCE__MODE": "api"}):
            from services.inference.app.config.settings import InferenceSettings
            s = InferenceSettings()
            assert s.mode == "api"

    def test_role_model_overrides_empty_by_default(self):
        from services.inference.app.config.settings import InferenceSettings
        s = InferenceSettings()
        assert s.role_model_overrides == {}

    def test_role_model_overrides_parsed_from_json(self):
        json_val = '{"coder": "ft:gpt-4o:acme:coding:abc123"}'
        with patch.dict(os.environ, {"INFERENCE__ROLE_MODEL_OVERRIDES": json_val}):
            from services.inference.app.config.settings import InferenceSettings
            s = InferenceSettings()
            assert s.role_model_overrides.get("coder") == "ft:gpt-4o:acme:coding:abc123"

    def test_api_key_optional(self):
        from services.inference.app.config.settings import InferenceSettings
        s = InferenceSettings()
        # Should not raise; api_key defaults to None/empty
        assert s.api_key is None or isinstance(s.api_key, str)
