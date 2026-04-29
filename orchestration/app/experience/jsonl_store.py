"""
JSONL storage — experience records
=====================================
Appends each ExperienceEvent as a single JSON line to the configured JSONL
file.  Uses an asyncio.Lock covering the full write to prevent concurrent
pipeline completions from interleaving bytes.

Blocking file I/O is offloaded to the default thread pool so the event loop
remains responsive during writes.

JSONL is the authoritative record.  Write failure is a hard error (propagated
to the caller so the background task can log it).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from shared.contracts.experience import ExperienceEvent
from ..config.settings import settings

logger = logging.getLogger(__name__)

# Single lock covering both append and count to guarantee consistent line count
# reads across concurrent pipeline completions.
_lock = asyncio.Lock()


def _append_sync(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line)


async def append(event: ExperienceEvent) -> None:
    """Append one ExperienceEvent to the JSONL file (atomic per-record)."""
    path = Path(settings.jsonl_path)
    line = event.model_dump_json() + "\n"
    async with _lock:
        await asyncio.to_thread(_append_sync, path, line)
    logger.debug("[%s] Experience appended to %s", event.session_id[:8], path)


def _count_lines_sync(path: Path) -> int:
    with path.open("r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def count() -> int:
    """Return the current number of stored experience records (sync, fast)."""
    path = Path(settings.jsonl_path)
    if not path.exists():
        return 0
    try:
        return _count_lines_sync(path)
    except OSError:
        return 0
