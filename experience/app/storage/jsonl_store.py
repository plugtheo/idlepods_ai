"""
JSONL storage
================
Appends each ExperienceEvent as a single JSON line to the configured
JSONL file.  Thread-safe via asyncio.Lock (single-process service).
Blocking file I/O is offloaded to the default thread pool so the event
loop remains responsive during writes and counts.

Schema: every line is a JSON object produced by ``event.model_dump_json()``.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from shared.contracts.experience import ExperienceEvent
from ..config.settings import settings

logger = logging.getLogger(__name__)

_lock = asyncio.Lock()


def _append_sync(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line)


async def append_experience(event: ExperienceEvent) -> None:
    """Append one ExperienceEvent to the JSONL file."""
    path = Path(settings.jsonl_path)
    line = event.model_dump_json() + "\n"
    loop = asyncio.get_running_loop()
    async with _lock:
        await loop.run_in_executor(None, _append_sync, path, line)
    logger.debug("[%s] Experience appended to %s", event.session_id[:8], path)


def _count_lines_sync(path: Path) -> int:
    with path.open("r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


async def count_experiences() -> int:
    """Return the total number of stored experience records."""
    path = Path(settings.jsonl_path)
    if not path.exists():
        return 0
    loop = asyncio.get_running_loop()
    async with _lock:
        return await loop.run_in_executor(None, _count_lines_sync, path)



