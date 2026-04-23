"""
Tests for the Context Service embedder.

Covers:
- embed:       correct encode params, float32 cast, model is cached between calls
- embed_async: delegates to embed via run_in_executor, returns the same result
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture()
def reset_model():
    """Reset the singleton model to None before/after each test."""
    import services.context.app.utils.embedder as mod
    saved = mod._model
    mod._model = None
    yield
    mod._model = saved


class TestEmbed:

    def test_encode_called_with_correct_params(self, reset_model):
        import services.context.app.utils.embedder as mod
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros(384)
        with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
            mod.embed("hello world")
        mock_model.encode.assert_called_once_with(
            "hello world",
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def test_returns_float32_ndarray(self, reset_model):
        """Output must be float32 regardless of the model's native dtype."""
        import services.context.app.utils.embedder as mod
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
            result = mod.embed("test")
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    def test_model_is_cached_on_subsequent_calls(self, reset_model):
        """SentenceTransformer is instantiated only once, not on every embed() call."""
        import services.context.app.utils.embedder as mod
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros(384)
        with patch("sentence_transformers.SentenceTransformer", return_value=mock_model) as st_cls:
            mod.embed("first call")
            mod.embed("second call")
        st_cls.assert_called_once()


class TestEmbedAsync:

    @pytest.mark.asyncio
    async def test_delegates_to_embed_and_returns_result(self):
        """embed_async wraps embed via run_in_executor — same output, non-blocking."""
        import services.context.app.utils.embedder as mod
        expected = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        with patch.object(mod, "embed", return_value=expected) as mock_embed:
            result = await mod.embed_async("hello async")
        mock_embed.assert_called_once_with("hello async")
        np.testing.assert_array_equal(result, expected)
