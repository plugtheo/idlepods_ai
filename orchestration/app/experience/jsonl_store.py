"""
JSONL storage — experience records
=====================================
Appends each ExperienceEvent as a single JSON line to a daily shard file.

Single-writer-per-line guarantee
--------------------------------
Concurrent writers are coordinated at two levels:

1. In-process: an ``asyncio.Lock`` serialises appends from the same event loop.
2. Across processes: ``filelock.FileLock`` (cross-platform) on ``<shard>.lock``
   prevents interleaving from separate workers / containers sharing a volume.

Daily rotation
--------------
Writes go to ``experiences-YYYYMMDD.jsonl`` in ``settings.jsonl_dir``.
The legacy ``experiences.jsonl`` is kept in place and remains readable by
the cursor until manually pruned.

Reads (``read_all_for_role``) walk all ``experiences-*.jsonl`` shards in
chronological filename order plus the legacy file if present.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import filelock

from shared.contracts.experience import ExperienceEvent
from ..config.settings import settings

logger = logging.getLogger(__name__)

# In-process lock — cross-process coordination is via filelock below.
_lock = asyncio.Lock()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _resolve_jsonl_path() -> Path:
    today = _utcnow().strftime("%Y%m%d")
    return Path(settings.jsonl_dir) / f"experiences-{today}.jsonl"


def _shard_lock(path: Path) -> filelock.FileLock:
    return filelock.FileLock(str(path) + ".lock", timeout=10)


def _append_sync(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _shard_lock(path):
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)


async def append(event: ExperienceEvent) -> None:
    """Append one ExperienceEvent to today's daily shard (atomic per-record)."""
    path = _resolve_jsonl_path()
    line = event.model_dump_json() + "\n"
    async with _lock:
        await asyncio.to_thread(_append_sync, path, line)
    logger.debug("[%s] Experience appended to %s", event.session_id[:8], path)


def _iter_shards() -> list[Path]:
    """Return all experience JSONL paths in chronological order."""
    base = Path(settings.jsonl_dir)
    dated = sorted(base.glob("experiences-*.jsonl"))
    legacy = Path(settings.jsonl_path)
    out: list[Path] = []
    if legacy.exists() and legacy not in dated:
        out.append(legacy)
    out.extend(dated)
    return out


def read_all_for_role(role: str) -> list[dict]:
    """Read all experience records matching *role* across all shards."""
    results: list[dict] = []
    for shard in _iter_shards():
        with _shard_lock(shard):
            try:
                for line in shard.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if record.get("role") == role or record.get("capability") == role:
                        results.append(record)
            except OSError:
                pass
    return results
