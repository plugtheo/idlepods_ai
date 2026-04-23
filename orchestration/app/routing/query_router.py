"""
QueryRouter — Orchestration Service re-export
===============================================
The canonical implementation lives in shared/routing/query_router.py so that
the Gateway and Orchestration Services are guaranteed to use identical logic.

Import directly from this module — the public names are re-exported here so
that existing imports (``from ..routing.query_router import QueryRouter``) and
test fixtures continue to work without change.
"""

from shared.routing.query_router import (  # noqa: F401
    Complexity,
    Intent,
    QueryRouter,
    RouteDecision,
)
