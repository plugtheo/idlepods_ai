"""
JSONL storage — experience records
=====================================
Appends each ExperienceEvent as a single JSON line to the configured JSONL
file.

Single-writer-per-line guarantee
--------------------------------
Concurrent writers are coordinated at two levels:

1. In-process: an ``asyncio.Lock`` serialises appends issued by the same
   event loop, avoiding redundant thread-pool contention.
2. Across processes: an ``fcntl.LOCK_EX`` advisory lock held for the
   duration of the write guarantees that bytes from concurrent writers in
   separate workers / containers (sharing the same volume) cannot
   interleave.  ``fcntl`` is POSIX-only; on Windows the OS-level lock is a
   no-op (development convenience — production runs Linux containers).

JSONL is the authoritative record.  Write failure is a hard error
(propagated to the caller so the background task can log it).
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from shared.contracts.experience import ExperienceEvent
from ..config.settings import settings

logger = logging.getLogger(__name__)

# In-process lock — cross-process coordination is via fcntl below.
_lock = asyncio.Lock()

if sys.platform == "win32":
    def _advisory_lock(fh, exclusive: bool) -> None:
        """No-op on Windows (dev only)."""
        return
else:
    import fcntl

    def _advisory_lock(fh, exclusive: bool) -> None:
        """Acquire an OS advisory lock on *fh*; released when the file closes."""
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)


def _append_sync(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        _advisory_lock(fh, exclusive=True)
        fh.write(line)


async def append(event: ExperienceEvent) -> None:
    """Append one ExperienceEvent to the JSONL file (atomic per-record)."""
    path = Path(settings.jsonl_path)
    line = event.model_dump_json() + "\n"
    async with _lock:
        await asyncio.to_thread(_append_sync, path, line)
    logger.debug("[%s] Experience appended to %s", event.session_id[:8], path)
