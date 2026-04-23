"""
API-key authentication middleware
===================================
Checks the ``Authorization: Bearer <key>`` header on every inbound request.

Auth is disabled when ``GATEWAY__API_KEY`` is not set or is an empty string,
which is the default for local development.

Paths exempted from auth:
  - GET /health
"""

from __future__ import annotations

import secrets

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from ..config.settings import settings

_EXEMPT_PATHS = {"/health"}


class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Auth is disabled when no key is configured
        if not settings.api_key:
            return await call_next(request)

        if request.url.path in _EXEMPT_PATHS:
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing or malformed Authorization header"},
            )

        token = auth_header[len("Bearer "):]
        if not secrets.compare_digest(token, settings.api_key):
            return JSONResponse(
                status_code=403,
                content={"detail": "Invalid API key"},
            )

        return await call_next(request)
