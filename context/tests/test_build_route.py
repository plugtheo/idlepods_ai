"""
Tests for the Context Service /v1/context/build route.

Uses FastAPI TestClient with mocked retrievers.
"""
import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient



@pytest.fixture
def context_client():
    from services.context.app.main import app
    return TestClient(app)


class TestBuildRoute:
    @pytest.fixture(autouse=True)
    def mock_retrievers(self):
        from shared.contracts.context import FewShotExample, RepoSnippet
        few_shot = FewShotExample(problem="q", solution="a", score=0.9, category="coding")
        snippet = RepoSnippet(file="app/utils.py", snippet="def helper(): pass", relevance=0.7)

        with (
            patch(
                "services.context.app.routes.build.retrieve_few_shots",
                new_callable=AsyncMock,
                return_value=[few_shot],
            ),
            patch(
                "services.context.app.routes.build.retrieve_repo_snippets",
                new_callable=AsyncMock,
                return_value=[snippet],
            ),
        ):
            yield

    def test_health(self, context_client):
        resp = context_client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_health_returns_service_name(self, context_client):
        resp = context_client.get("/health")
        assert resp.json()["service"] == "context"

    def test_session_id_is_optional(self, context_client):
        resp = context_client.post(
            "/v1/context/build",
            json={
                "prompt": "implement a cache",
                "intent": "coding",
                "complexity": "moderate",
                "session_id": "abc-123",
            },
        )
        assert resp.status_code == 200

    def test_build_context_200(self, context_client):
        resp = context_client.post(
            "/v1/context/build",
            json={
                "prompt": "implement a sorting algorithm",
                "intent": "coding",
                "complexity": "moderate",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "few_shots" in body
        assert "repo_snippets" in body
        assert "system_hints" in body
        assert len(body["few_shots"]) == 1
        assert len(body["repo_snippets"]) == 1

    def test_build_context_missing_fields_returns_422(self, context_client):
        resp = context_client.post(
            "/v1/context/build",
            json={"prompt": "do something"},  # missing intent, complexity
        )
        assert resp.status_code == 422

    def test_build_context_degrades_on_retriever_error(self, context_client):
        """If retrievers raise, the route returns empty context (200) not 500."""
        with (
            patch(
                "services.context.app.routes.build.retrieve_few_shots",
                new_callable=AsyncMock,
                side_effect=Exception("ChromaDB down"),
            ),
            patch(
                "services.context.app.routes.build.retrieve_repo_snippets",
                new_callable=AsyncMock,
                side_effect=Exception("IO error"),
            ),
        ):
            resp = context_client.post(
                "/v1/context/build",
                json={
                    "prompt": "fix bug",
                    "intent": "debugging",
                    "complexity": "simple",
                },
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["few_shots"] == []
        assert body["repo_snippets"] == []

    def test_system_hints_coding_intent(self, context_client):
        resp = context_client.post(
            "/v1/context/build",
            json={"prompt": "write a parser", "intent": "coding", "complexity": "simple"},
        )
        assert resp.status_code == 200
        hints = resp.json()["system_hints"]
        assert "code" in hints.lower() or "production" in hints.lower()

    def test_system_hints_debugging_intent(self, context_client):
        resp = context_client.post(
            "/v1/context/build",
            json={"prompt": "fix crash", "intent": "debugging", "complexity": "simple"},
        )
        assert resp.status_code == 200
        hints = resp.json()["system_hints"]
        assert "root cause" in hints.lower() or "fix" in hints.lower()

    def test_system_hints_analysis_intent(self, context_client):
        resp = context_client.post(
            "/v1/context/build",
            json={"prompt": "analyse the data", "intent": "analysis", "complexity": "moderate"},
        )
        assert resp.status_code == 200
        hints = resp.json()["system_hints"]
        assert hints != ""


class TestBuildHints:
    """Unit tests for _build_hints — no HTTP overhead."""

    def test_coding_intent(self):
        from services.context.app.routes.build import _build_hints
        result = _build_hints("coding", "moderate")
        assert "code" in result.lower() or "production" in result.lower()

    def test_debugging_intent(self):
        from services.context.app.routes.build import _build_hints
        result = _build_hints("debugging", "moderate")
        assert "root cause" in result.lower() or "fix" in result.lower()

    def test_research_intent(self):
        from services.context.app.routes.build import _build_hints
        result = _build_hints("research", "moderate")
        assert "accurate" in result.lower() or "source" in result.lower() or "reasoning" in result.lower()

    def test_planning_intent(self):
        from services.context.app.routes.build import _build_hints
        result = _build_hints("planning", "moderate")
        assert "step" in result.lower() or "actionable" in result.lower()

    def test_analysis_intent(self):
        from services.context.app.routes.build import _build_hints
        result = _build_hints("analysis", "moderate")
        assert "systematic" in result.lower() or "evidence" in result.lower()

    def test_unknown_intent_returns_empty_string(self):
        from services.context.app.routes.build import _build_hints
        assert _build_hints("unknown_intent", "moderate") == ""

    def test_complex_complexity_appends_hint(self):
        from services.context.app.routes.build import _build_hints
        result = _build_hints("coding", "complex")
        assert "edge case" in result.lower() or "scalab" in result.lower()

    def test_simple_complexity_appends_hint(self):
        from services.context.app.routes.build import _build_hints
        result = _build_hints("coding", "simple")
        assert "focused" in result.lower() or "concise" in result.lower()

    def test_moderate_complexity_no_suffix(self):
        from services.context.app.routes.build import _build_hints
        base = _build_hints("coding", "moderate")
        assert "edge case" not in base.lower()
        assert "focused" not in base.lower()

    def test_two_hints_joined_with_double_space(self):
        from services.context.app.routes.build import _build_hints
        result = _build_hints("coding", "complex")
        # Two sentences joined by the double-space separator
        assert "  " in result

    def test_one_retriever_failure_does_not_zero_other(self, context_client):
        """With return_exceptions=True, a failure in one retriever leaves the other intact."""
        from shared.contracts.context import FewShotExample
        few_shot = FewShotExample(problem="q", solution="a", score=0.9, category="coding")
        with (
            patch(
                "services.context.app.routes.build.retrieve_few_shots",
                new_callable=AsyncMock,
                return_value=[few_shot],
            ),
            patch(
                "services.context.app.routes.build.retrieve_repo_snippets",
                new_callable=AsyncMock,
                side_effect=Exception("IO error"),
            ),
        ):
            resp = context_client.post(
                "/v1/context/build",
                json={"prompt": "fix the bug", "intent": "debugging", "complexity": "simple"},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["few_shots"]) == 1   # still returned despite repo failure
        assert body["repo_snippets"] == []
