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
            assert "qwen" in s.qwen_url.lower()
            assert s.qwen_model_id == "Qwen/Qwen3-14B"
