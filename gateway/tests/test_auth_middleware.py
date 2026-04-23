"""
Tests for the Gateway APIKeyMiddleware.

Covers:
- Auth disabled when no API key configured
- /health path exempt from auth
- Missing Authorization header → 401
- Wrong token → 403
- Correct token → 200 passthrough
"""
import pytest
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _make_app(api_key: str) -> FastAPI:
    """Build a minimal FastAPI app with the middleware and a test route."""
    app = FastAPI()

    # Patch settings before adding middleware
    with patch("services.gateway.app.middleware.auth.settings") as mock_settings:
        mock_settings.api_key = api_key
        middleware = APIKeyMiddleware(app)

    # Manually add the middleware instance
    from starlette.middleware import Middleware
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.responses import JSONResponse

    async def homepage(request):
        return JSONResponse({"ok": True})

    async def health(request):
        return JSONResponse({"status": "ok"})

    inner_app = Starlette(
        routes=[
            Route("/v1/chat", homepage, methods=["POST"]),
            Route("/health", health, methods=["GET"]),
        ],
        middleware=[
            Middleware(
                APIKeyMiddleware,
                # We'll patch settings per test using monkeypatch
            )
        ],
    )
    return inner_app


class TestAuthMiddleware:
    """Test auth middleware via the full Gateway app with patched settings."""

    @pytest.fixture
    def client_no_auth(self):
        """App with no API key configured → auth disabled."""
        from services.gateway.app.middleware.auth import APIKeyMiddleware
        with patch("services.gateway.app.middleware.auth.settings") as s:
            s.api_key = ""
            from fastapi import FastAPI
            from fastapi.testclient import TestClient
            from starlette.responses import JSONResponse

            app = FastAPI()
            app.add_middleware(APIKeyMiddleware)

            @app.get("/v1/chat")
            async def chat():
                return {"ok": True}

            @app.get("/health")
            async def health():
                return {"status": "ok"}

            yield TestClient(app)

    @pytest.fixture
    def client_with_auth(self):
        """App with API key = 'secret'."""
        from services.gateway.app.middleware.auth import APIKeyMiddleware
        with patch("services.gateway.app.middleware.auth.settings") as s:
            s.api_key = "secret"
            from fastapi import FastAPI
            from fastapi.testclient import TestClient

            app = FastAPI()
            app.add_middleware(APIKeyMiddleware)

            @app.get("/v1/chat")
            async def chat():
                return {"ok": True}

            @app.get("/health")
            async def health():
                return {"status": "ok"}

            yield TestClient(app)

    def test_no_auth_key_allows_all_requests(self, client_no_auth):
        resp = client_no_auth.get("/v1/chat")
        assert resp.status_code == 200

    def test_health_exempt_from_auth(self, client_with_auth):
        resp = client_with_auth.get("/health")
        assert resp.status_code == 200

    def test_missing_auth_header_returns_401(self, client_with_auth):
        resp = client_with_auth.get("/v1/chat")
        assert resp.status_code == 401
        assert "Authorization" in resp.json()["detail"]

    def test_wrong_token_returns_403(self, client_with_auth):
        resp = client_with_auth.get(
            "/v1/chat",
            headers={"Authorization": "Bearer wrongtoken"},
        )
        assert resp.status_code == 403

    def test_correct_token_passes(self, client_with_auth):
        resp = client_with_auth.get(
            "/v1/chat",
            headers={"Authorization": "Bearer secret"},
        )
        assert resp.status_code == 200

    def test_malformed_header_no_bearer_prefix_returns_401(self, client_with_auth):
        resp = client_with_auth.get(
            "/v1/chat",
            headers={"Authorization": "secret"},
        )
        assert resp.status_code == 401
