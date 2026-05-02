"""
Redis session store
===================
Persists conversation history across multi-turn requests.

get_session(task_id)               → list[dict]  ([] on miss/error)
save_session(task_id, history, ttl) → None
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

_client = None
_client_lock = threading.Lock()
_redis_ok: bool = True


def is_healthy() -> bool:
    return _redis_ok


def _get_client():
    """Return the singleton async Redis client, initialising on first call."""
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is not None:
            return _client
        from ..config.settings import settings
        import redis.asyncio as aioredis

        _client = aioredis.from_url(settings.redis_url, decode_responses=True)
        logger.info("Redis session store → %s", settings.redis_url)
    return _client


async def get_session(task_id: str) -> list[dict[str, Any]]:
    """Return stored conversation history for task_id, or [] on miss/error."""
    global _redis_ok
    try:
        raw = await _get_client().get(f"session:v2:{task_id}")
        _redis_ok = True
        if raw:
            return json.loads(raw)
    except Exception as exc:
        logger.error("get_session(%s) failed: %s", task_id, exc)
        _redis_ok = False
    return []


async def save_session(task_id: str, history: list[dict[str, Any]], ttl: int) -> None:
    """Upsert history for task_id with the given TTL (seconds)."""
    global _redis_ok
    try:
        await _get_client().setex(f"session:v2:{task_id}", ttl, json.dumps(history))
        _redis_ok = True
    except Exception as exc:
        logger.error("save_session(%s) failed: %s", task_id, exc)
        _redis_ok = False


async def get_fingerprints(task_id: str) -> dict[str, str] | None:
    """Return stored file fingerprints for task_id, or None on miss/error."""
    global _redis_ok
    try:
        raw = await _get_client().get(f"fps:v2:{task_id}")
        _redis_ok = True
        if raw:
            return json.loads(raw)
    except Exception as exc:
        logger.error("get_fingerprints(%s) failed: %s", task_id, exc)
        _redis_ok = False
    return None


async def save_fingerprints(task_id: str, fps: dict[str, str], ttl: int) -> None:
    """Upsert file fingerprints for task_id with the given TTL (seconds)."""
    global _redis_ok
    try:
        await _get_client().setex(f"fps:v2:{task_id}", ttl, json.dumps(fps))
        _redis_ok = True
    except Exception as exc:
        logger.error("save_fingerprints(%s) failed: %s", task_id, exc)
        _redis_ok = False


async def get_snippets(task_id: str) -> list[dict[str, Any]]:
    """Return stored repo snippets for task_id, or [] on miss/error."""
    global _redis_ok
    try:
        raw = await _get_client().get(f"snippets:v2:{task_id}")
        _redis_ok = True
        if raw:
            return json.loads(raw)
    except Exception as exc:
        logger.error("get_snippets(%s) failed: %s", task_id, exc)
        _redis_ok = False
    return []


async def save_snippets(task_id: str, snippets: list[dict[str, Any]], ttl: int) -> None:
    """Upsert repo snippets for task_id with the given TTL (seconds)."""
    global _redis_ok
    try:
        await _get_client().setex(f"snippets:v2:{task_id}", ttl, json.dumps(snippets))
        _redis_ok = True
    except Exception as exc:
        logger.error("save_snippets(%s) failed: %s", task_id, exc)
        _redis_ok = False


_task_state_warned: set[str] = set()

_DEFAULT_TASK_STATE_TTL = 7 * 86400  # 7 days


async def set_plan(task_id: str, plan: Any, ttl_s: int = _DEFAULT_TASK_STATE_TTL) -> None:
    """Persist plan state for task_id. plan is a Plan object or JSON-serialisable dict."""
    global _redis_ok
    try:
        if hasattr(plan, "model_dump"):
            payload = plan.model_dump(mode="json")
        else:
            payload = plan
        await _get_client().setex(f"task_state:{task_id}", ttl_s, json.dumps(payload))
        _redis_ok = True
    except Exception as exc:
        logger.error("set_plan(%s) failed: %s", task_id, exc)
        _redis_ok = False


async def get_plan(task_id: str) -> Any | None:
    """Return stored Plan dict for task_id, or None on miss/error."""
    global _redis_ok
    try:
        raw = await _get_client().get(f"task_state:{task_id}")
        _redis_ok = True
        if raw:
            return json.loads(raw)
    except Exception as exc:
        if task_id not in _task_state_warned:
            logger.warning("task_state degraded for %s: %s", task_id, exc)
            _task_state_warned.add(task_id)
        _redis_ok = False
    return None
