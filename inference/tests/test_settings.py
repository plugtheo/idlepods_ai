"""
Tests for InferenceSettings — env var parsing and defaults.
"""
import pytest
import os
from unittest.mock import patch


class TestInferenceSettings:
    def test_defaults(self):
        with patch.dict(os.environ, {}, clear=False):
            from services.inference.app.config.settings import InferenceSettings
            s = InferenceSettings()
            assert "deepseek" in s.deepseek_url
            assert "mistral" in s.mistral_url
