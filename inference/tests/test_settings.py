"""
Tests for InferenceSettings — env var parsing and defaults.
"""
import os
from unittest.mock import patch


class TestInferenceSettings:
    def test_defaults(self):
        with patch.dict(os.environ, {}, clear=False):
            from services.inference.app.config.settings import InferenceSettings
            s = InferenceSettings()
            assert s.models_yaml_path == "/config/models.yaml"
            assert s.accept_legacy_backend_names is False
