"""
Tests for the experience vector store.

ChromaDB and SentenceTransformer are mocked — tests verify:
- upsert_experience() is non-fatal on ChromaDB failure
- upsert is called with correct data on success
"""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from shared.contracts.experience import ExperienceEvent, AgentContribution


def _make_event(session_id="sess-vec"):
    return ExperienceEvent(
        session_id=session_id,
        prompt="implement caching",
        final_output="class Cache: pass",
        agent_chain=["coder", "reviewer"],
        contributions=[
            AgentContribution(role="coder", output="code", quality_score=0.85, iteration=1),
        ],
        final_score=0.85,
        iterations=1,
        converged=True,
    )


@pytest.mark.asyncio
class TestVectorStore:
    async def test_upsert_non_fatal_on_chromadb_failure(self):
        """If ChromaDB raises, upsert_experience() should swallow the error."""
        from services.experience.app.storage import vector_store

        # Reset module-level singleton so our mock takes effect
        vector_store._chroma_client = None
        vector_store._collection = None

        with patch(
            "services.experience.app.storage.vector_store._get_collection",
            side_effect=Exception("ChromaDB unavailable"),
        ):
            # Should NOT raise
            await vector_store.upsert_experience(_make_event())

    async def test_upsert_calls_collection_upsert(self):
        """When ChromaDB is available, collection.upsert() is called."""
        from services.experience.app.storage import vector_store

        mock_collection = MagicMock()
        mock_collection.upsert = MagicMock()

        with (
            patch(
                "services.experience.app.storage.vector_store._get_collection",
                return_value=mock_collection,
            ),
            patch(
                "services.experience.app.storage.vector_store._embed",
                return_value=[0.1, 0.2, 0.3],
            ),
        ):
            await vector_store.upsert_experience(_make_event("my-session"))

        mock_collection.upsert.assert_called_once()
        call_kwargs = mock_collection.upsert.call_args[1]
        assert call_kwargs["ids"] == ["my-session"]
        assert call_kwargs["documents"] == ["implement caching"]

    async def test_upsert_embed_failure_is_non_fatal(self):
        """If embedding fails, the error is swallowed."""
        from services.experience.app.storage import vector_store

        with (
            patch(
                "services.experience.app.storage.vector_store._get_collection",
                return_value=MagicMock(),
            ),
            patch(
                "services.experience.app.storage.vector_store._embed",
                side_effect=RuntimeError("CUDA OOM"),
            ),
        ):
            await vector_store.upsert_experience(_make_event())
