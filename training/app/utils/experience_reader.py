"""
Experience reader
==================
Reads the shared JSONL shards that the Experience Service writes.
The Training Service mounts the same ``/data`` volume so this is a
direct local file read — no HTTP call needed.

Shard layout
------------
Daily shards:  ``experiences-YYYYMMDD.jsonl`` in the same directory as
``settings.jsonl_path``.  The legacy ``experiences.jsonl`` is readable until
manually pruned.  Both are walked in chronological order.

Cursor support
--------------
``iter_after(cursor)`` skips records up to the stored ``{shard, offset}``
position so the scheduler can replay only unprocessed records.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import filelock

from shared.contracts.experience import SCORER_RULE_VERSION
from ..config.settings import settings

logger = logging.getLogger(__name__)

PROMPT_FINGERPRINT_MAX_CHARS = 120


def _jsonl_dir() -> Path:
    return Path(settings.jsonl_path).parent


def _iter_shards() -> List[Path]:
    """Return all experience JSONL paths in chronological order."""
    base = _jsonl_dir()
    dated = sorted(base.glob("experiences-*.jsonl"))
    legacy = Path(settings.jsonl_path)
    out: List[Path] = []
    if legacy.exists() and legacy not in dated:
        out.append(legacy)
    out.extend(dated)
    return out


def iter_records() -> Generator[Tuple[Path, int, dict], None, None]:
    """Yield ``(shard_path, line_offset, record)`` for every record across all shards."""
    for shard in _iter_shards():
        lock = filelock.FileLock(str(shard) + ".lock", timeout=10)
        with lock.acquire(poll_interval=0.1):
            try:
                lines = shard.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
        for offset, raw in enumerate(lines):
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                continue
            yield shard, offset, record


def iter_after(cursor: Optional[dict]) -> Generator[Tuple[Path, int, dict], None, None]:
    """Yield records that come after *cursor* (``{shard, offset}`` dict or ``None``)."""
    if cursor is None:
        yield from iter_records()
        return
    cursor_shard = Path(cursor["shard"])
    cursor_offset = int(cursor["offset"])
    shards = _iter_shards()
    try:
        cursor_shard_idx = shards.index(cursor_shard)
    except ValueError:
        # cursor shard no longer present — replay everything
        yield from iter_records()
        return
    for shard, offset, record in iter_records():
        try:
            shard_idx = shards.index(shard)
        except ValueError:
            continue
        if shard_idx < cursor_shard_idx:
            continue
        if shard_idx == cursor_shard_idx and offset <= cursor_offset:
            continue
        yield shard, offset, record


def load_experiences() -> List[dict]:
    """Return all experience records from all shards."""
    return [record for _, _, record in iter_records()]


def check_diversity(records: List[dict]) -> Tuple[bool, str]:
    """
    Evaluate whether the experience batch is diverse enough for training.

    Three criteria (all must pass):
    1. Batch size  ≥ settings.min_batch_size
    2. Score spread  (max - min) ≥ settings.min_score_spread
    3. Unique fingerprint ratio  ≥ settings.min_diversity_ratio

    Returns (passes: bool, reason: str).
    """
    n = len(records)
    if n < settings.min_batch_size:
        return False, f"too few experiences: {n} < {settings.min_batch_size}"

    scores = [float(r.get("final_score", 0.0)) for r in records]
    spread = max(scores) - min(scores)
    if spread < settings.min_score_spread:
        return False, (
            f"score spread too small: {spread:.3f} < {settings.min_score_spread} "
            f"(range {min(scores):.2f}–{max(scores):.2f})"
        )

    fingerprints = {_fingerprint(r.get("prompt", "")) for r in records}
    diversity_ratio = len(fingerprints) / n
    if diversity_ratio < settings.min_diversity_ratio:
        return False, (
            f"low prompt diversity: {diversity_ratio:.2f} < {settings.min_diversity_ratio} "
            f"({len(fingerprints)} unique / {n} total)"
        )

    return True, f"criteria met: n={n}, spread={spread:.2f}, diversity={diversity_ratio:.2f}"


def _fingerprint(prompt: str) -> str:
    normalised = "".join(prompt.lower().split())
    return normalised[:PROMPT_FINGERPRINT_MAX_CHARS]
