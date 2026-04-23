"""
Tests for the Context Service few-shot retrieval (RAG).

Covers:
- _init_chroma_sync: cloud vs self-hosted client selection, collection args
- _get_collection:   init success/failure, caching (no reinitialisation)
- retrieve_few_shots: None collection, empty collection, similarity threshold
                      filtering, metadata fallback, query failure, defaults
"""

import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_mock_collection(count=1, docs=None, metas=None, distances=None):
    col = MagicMock()
    col.count.return_value = count
    if docs is not None:
        col.query.return_value = {
            "documents": [docs],
            "metadatas": [metas if metas is not None else [{}] * len(docs)],
            "distances": [distances if distances is not None else [0.0] * len(docs)],
        }
    return col


def _mock_settings(max_few_shots=4, similarity_threshold=0.68):
    s = MagicMock()
    s.max_few_shots = max_few_shots
    s.similarity_threshold = similarity_threshold
    return s


@pytest.fixture()
def reset_chroma_collection():
    """Reset the module-global _chroma_collection to None before/after each test."""
    import services.context.app.retrieval.few_shot as mod
    saved = mod._chroma_collection
    mod._chroma_collection = None
    yield
    mod._chroma_collection = saved


# ── _init_chroma_sync ─────────────────────────────────────────────────────────


class TestInitChromaSync:

    def _settings(self, *, api_key="", tenant="", database="",
                  host="chromadb", port=8000, collection="experiences"):
        s = MagicMock()
        s.chroma_api_key = api_key
        s.chroma_tenant = tenant
        s.chroma_database = database
        s.chroma_host = host
        s.chroma_port = port
        s.chroma_collection = collection
        return s

    def test_cloud_mode_creates_cloud_client(self):
        mock_col = _make_mock_collection(count=5)
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_col
        mock_settings = self._settings(api_key="sk-test", tenant="my-tenant", database="my-db")

        with (
            patch("services.context.app.retrieval.few_shot.settings", mock_settings),
            patch("chromadb.CloudClient", return_value=mock_client) as cloud_cls,
            patch("chromadb.HttpClient") as http_cls,
        ):
            from services.context.app.retrieval.few_shot import _init_chroma_sync
            result = _init_chroma_sync()

        cloud_cls.assert_called_once_with(
            tenant="my-tenant", database="my-db", api_key="sk-test"
        )
        http_cls.assert_not_called()
        assert result is mock_col

    def test_self_hosted_mode_creates_http_client(self):
        mock_col = _make_mock_collection(count=0)
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_col
        mock_settings = self._settings(api_key="", host="my-chroma", port=9000)

        with (
            patch("services.context.app.retrieval.few_shot.settings", mock_settings),
            patch("chromadb.CloudClient") as cloud_cls,
            patch("chromadb.HttpClient", return_value=mock_client) as http_cls,
        ):
            from services.context.app.retrieval.few_shot import _init_chroma_sync
            result = _init_chroma_sync()

        http_cls.assert_called_once_with(host="my-chroma", port=9000)
        cloud_cls.assert_not_called()
        assert result is mock_col

    def test_collection_created_with_cosine_space_and_correct_name(self):
        mock_col = _make_mock_collection()
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_col
        mock_settings = self._settings(api_key="", collection="my-experiences")

        with (
            patch("services.context.app.retrieval.few_shot.settings", mock_settings),
            patch("chromadb.HttpClient", return_value=mock_client),
        ):
            from services.context.app.retrieval.few_shot import _init_chroma_sync
            _init_chroma_sync()

        mock_client.get_or_create_collection.assert_called_once_with(
            name="my-experiences",
            metadata={"hnsw:space": "cosine"},
        )


# ── _get_collection ────────────────────────────────────────────────────────────


class TestGetCollection:

    @pytest.mark.asyncio
    async def test_successful_init_returns_collection(self, reset_chroma_collection):
        import services.context.app.retrieval.few_shot as mod
        mock_col = MagicMock()
        with patch.object(mod, "_init_chroma_sync", return_value=mock_col):
            result = await mod._get_collection()
        assert result is mock_col

    @pytest.mark.asyncio
    async def test_init_failure_returns_none(self, reset_chroma_collection):
        import services.context.app.retrieval.few_shot as mod
        with patch.object(mod, "_init_chroma_sync", side_effect=ConnectionError("unreachable")):
            result = await mod._get_collection()
        assert result is None

    @pytest.mark.asyncio
    async def test_cached_collection_not_reinitialised(self):
        """If _chroma_collection is already set, _init_chroma_sync must not be called."""
        import services.context.app.retrieval.few_shot as mod
        mock_col = MagicMock()
        saved = mod._chroma_collection
        mod._chroma_collection = mock_col
        try:
            with patch.object(mod, "_init_chroma_sync") as mock_init:
                result = await mod._get_collection()
            mock_init.assert_not_called()
            assert result is mock_col
        finally:
            mod._chroma_collection = saved


# ── retrieve_few_shots ────────────────────────────────────────────────────────


class TestRetrieveFewShots:

    @pytest.mark.asyncio
    async def test_returns_empty_when_collection_is_none(self):
        import services.context.app.retrieval.few_shot as mod
        with patch.object(mod, "_get_collection", new=AsyncMock(return_value=None)):
            result = await mod.retrieve_few_shots("any prompt")
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_when_collection_count_is_zero(self):
        import services.context.app.retrieval.few_shot as mod
        col = _make_mock_collection(count=0)
        with patch.object(mod, "_get_collection", new=AsyncMock(return_value=col)):
            result = await mod.retrieve_few_shots("any prompt")
        assert result == []
        col.query.assert_not_called()

    @pytest.mark.asyncio
    async def test_above_threshold_example_returned(self):
        import services.context.app.retrieval.few_shot as mod
        col = _make_mock_collection(
            count=1,
            docs=["Write a rate limiter"],
            metas=[{"problem": "rate limiter", "solution": "token bucket",
                    "score": "0.9", "category": "coding"}],
            distances=[0.1],   # similarity = 1 - 0.1 = 0.9 > 0.68
        )
        with (
            patch.object(mod, "_get_collection", new=AsyncMock(return_value=col)),
            patch("services.context.app.retrieval.few_shot.embed_async",
                  new=AsyncMock(return_value=np.zeros(384, dtype=np.float32))),
            patch("services.context.app.retrieval.few_shot.settings", _mock_settings()),
        ):
            result = await mod.retrieve_few_shots("rate limiter problem")

        assert len(result) == 1
        ex = result[0]
        assert ex.problem == "rate limiter"
        assert ex.solution == "token bucket"
        assert ex.score == 0.9
        assert ex.category == "coding"

    @pytest.mark.asyncio
    async def test_below_threshold_example_excluded(self):
        import services.context.app.retrieval.few_shot as mod
        col = _make_mock_collection(
            count=1,
            docs=["irrelevant"],
            metas=[{"problem": "irrelevant", "solution": "?",
                    "score": "0.1", "category": "general"}],
            distances=[0.5],   # similarity = 0.5 < 0.68
        )
        with (
            patch.object(mod, "_get_collection", new=AsyncMock(return_value=col)),
            patch("services.context.app.retrieval.few_shot.embed_async",
                  new=AsyncMock(return_value=np.zeros(384, dtype=np.float32))),
            patch("services.context.app.retrieval.few_shot.settings", _mock_settings()),
        ):
            result = await mod.retrieve_few_shots("something unrelated")
        assert result == []

    @pytest.mark.asyncio
    async def test_missing_problem_falls_back_to_doc(self):
        """When metadata has no 'problem' key, doc[:500] is used instead."""
        import services.context.app.retrieval.few_shot as mod
        doc_text = "document text used as fallback"
        col = _make_mock_collection(
            count=1,
            docs=[doc_text],
            metas=[{"solution": "ans", "score": "0.8", "category": "coding"}],
            distances=[0.05],  # similarity = 0.95
        )
        with (
            patch.object(mod, "_get_collection", new=AsyncMock(return_value=col)),
            patch("services.context.app.retrieval.few_shot.embed_async",
                  new=AsyncMock(return_value=np.zeros(384, dtype=np.float32))),
            patch("services.context.app.retrieval.few_shot.settings", _mock_settings()),
        ):
            result = await mod.retrieve_few_shots("coding question")
        assert len(result) == 1
        assert result[0].problem == doc_text

    @pytest.mark.asyncio
    async def test_missing_score_and_category_use_defaults(self):
        """score defaults to 0.0 and category defaults to 'general'."""
        import services.context.app.retrieval.few_shot as mod
        col = _make_mock_collection(
            count=1,
            docs=["doc"],
            metas=[{"problem": "p", "solution": "s"}],  # no score / category
            distances=[0.05],
        )
        with (
            patch.object(mod, "_get_collection", new=AsyncMock(return_value=col)),
            patch("services.context.app.retrieval.few_shot.embed_async",
                  new=AsyncMock(return_value=np.zeros(384, dtype=np.float32))),
            patch("services.context.app.retrieval.few_shot.settings", _mock_settings()),
        ):
            result = await mod.retrieve_few_shots("prompt")
        assert result[0].score == 0.0
        assert result[0].category == "general"

    @pytest.mark.asyncio
    async def test_query_exception_returns_empty_list(self):
        import services.context.app.retrieval.few_shot as mod
        col = MagicMock()
        col.count.return_value = 3
        col.query.side_effect = RuntimeError("DB failure")
        with (
            patch.object(mod, "_get_collection", new=AsyncMock(return_value=col)),
            patch("services.context.app.retrieval.few_shot.embed_async",
                  new=AsyncMock(return_value=np.zeros(384, dtype=np.float32))),
            patch("services.context.app.retrieval.few_shot.settings", _mock_settings()),
        ):
            result = await mod.retrieve_few_shots("any prompt")
        assert result == []

    @pytest.mark.asyncio
    async def test_n_results_capped_at_max_few_shots(self):
        """collection.query is called with n_results = min(max_few_shots, count)."""
        import services.context.app.retrieval.few_shot as mod
        col = _make_mock_collection(
            count=10,
            docs=["d"] * 2,
            metas=[{"problem": "p", "solution": "s", "score": "0.9", "category": "c"}] * 2,
            distances=[0.05, 0.05],
        )
        settings = _mock_settings(max_few_shots=2)
        with (
            patch.object(mod, "_get_collection", new=AsyncMock(return_value=col)),
            patch("services.context.app.retrieval.few_shot.embed_async",
                  new=AsyncMock(return_value=np.zeros(384, dtype=np.float32))),
            patch("services.context.app.retrieval.few_shot.settings", settings),
        ):
            await mod.retrieve_few_shots("prompt")

        call_kwargs = col.query.call_args
        assert call_kwargs.kwargs["n_results"] == 2
