"""
In-flight experience-recording task tracking
=============================================
Tracks every ``recorder.record(...)`` task created by the run routes so that
the FastAPI shutdown handler can drain them cleanly on SIGTERM.  Tasks that
do not complete within the drain window have their original
``ExperienceEvent`` payloads serialised to a spool JSONL file; on the next
startup ``replay_spool`` re-submits them through the recorder.

Scope: this module covers experience-recording tasks ONLY.  Other
fire-and-forget tasks (e.g. Redis ``save_session``) are out of scope.

Replay deduplication is intentionally lightweight: before re-appending a
spooled record, the last 100 lines of the main JSONL file are inspected for
matching ``(session_id, timestamp)`` pairs.  This catches the rare case
where the original task did finish writing but its completion arrived after
the drain timeout.  No persistent dedupe state is kept — downstream
training already de-duplicates by session_id.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Iterable

from shared.contracts.experience import ExperienceEvent

from . import jsonl_store, recorder

logger = logging.getLogger(__name__)

_DEDUPE_TAIL_LINES = 100

_active: "dict[asyncio.Task, ExperienceEvent]" = {}


def register(task: asyncio.Task, event: ExperienceEvent) -> None:
    """Track *task* until it completes; the payload *event* is retained for spooling."""
    _active[task] = event
    task.add_done_callback(_active.pop)


def pending_events() -> list[ExperienceEvent]:
    """Return events whose recording task is still running."""
    return [evt for t, evt in _active.items() if not t.done()]


async def drain(timeout: float) -> list[ExperienceEvent]:
    """Wait up to *timeout* seconds for active tasks; return events still pending."""
    tasks = [t for t in list(_active.keys()) if not t.done()]
    if not tasks:
        return []
    done, pending = await asyncio.wait(tasks, timeout=timeout)
    remaining = [_active[t] for t in pending if t in _active]
    if remaining:
        logger.warning(
            "Shutdown drain timed out — %d experience task(s) still pending; spooling.",
            len(remaining),
        )
    return remaining


def spool_pending(events: Iterable[ExperienceEvent], path: Path) -> None:
    """Append pending *events* to the spool file (synchronous; called at shutdown)."""
    import filelock
    events = list(events)
    if not events:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with filelock.FileLock(str(path) + ".lock", timeout=10):
        with path.open("a", encoding="utf-8") as fh:
            for evt in events:
                fh.write(evt.model_dump_json() + "\n")
    logger.info("Spooled %d undrained experience event(s) to %s", len(events), path)


def _read_spool_sync(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed spool line: %s", exc)
    return out


def _tail_dedupe_keys(path: Path, n: int = _DEDUPE_TAIL_LINES) -> set[tuple[str, str]]:
    """Return ``(session_id, timestamp)`` keys present in the last *n* lines of *path*."""
    if not path.exists():
        return set()
    try:
        with path.open("r", encoding="utf-8") as fh:
            lines = fh.readlines()[-n:]
    except OSError as exc:
        logger.warning("Could not tail %s for dedupe: %s", path, exc)
        return set()
    keys: set[tuple[str, str]] = set()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        sid = obj.get("session_id")
        ts = obj.get("timestamp")
        if sid is not None and ts is not None:
            keys.add((str(sid), str(ts)))
    return keys


def _truncate_spool_sync(path: Path) -> None:
    if not path.exists():
        return
    with path.open("w", encoding="utf-8") as fh:
        from .jsonl_store import _advisory_lock
        _advisory_lock(fh, exclusive=True)
        fh.truncate(0)


async def replay_spool(spool_path: Path, jsonl_path: Path) -> None:
    """Re-submit spooled experience events through the recorder, then truncate the spool."""
    raw = await asyncio.to_thread(_read_spool_sync, spool_path)
    if not raw:
        return
    dedupe_keys = await asyncio.to_thread(_tail_dedupe_keys, jsonl_path)

    replayed = 0
    skipped = 0
    for obj in raw:
        sid = obj.get("session_id")
        ts = obj.get("timestamp")
        if sid is not None and ts is not None and (str(sid), str(ts)) in dedupe_keys:
            skipped += 1
            continue
        try:
            event = ExperienceEvent.model_validate(obj)
        except Exception as exc:
            logger.warning("Skipping invalid spool record: %s", exc)
            continue
        try:
            await recorder.record(event)
            replayed += 1
        except Exception as exc:
            logger.error("Spool replay failed for session %s: %s", str(sid)[:8], exc)

    await asyncio.to_thread(_truncate_spool_sync, spool_path)
    logger.info("Spool replay complete: replayed=%d skipped=%d", replayed, skipped)
