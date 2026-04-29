"""
Context builder
================
Replaces the HTTP call to the Context Service.  Runs few-shot retrieval,
repo scanning, and hint generation concurrently, then bundles the results
into a BuiltContext.

Any component failure is treated as an empty result for that component —
context enrichment failure must never fail the pipeline.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

from shared.contracts.context import BuiltContext, RepoSnippet
from . import few_shot, hints, repo
from .repo import CONTENT_PREVIEW_BYTES, _file_fingerprint

logger = logging.getLogger(__name__)


def _compute_fingerprints(allowed_files: list[str], repo_root: Path) -> dict[str, str]:
    return {f: _file_fingerprint(repo_root / f) for f in allowed_files}


def _validate_snippet(s: dict, repo_root: Path) -> bool:
    try:
        raw = (repo_root / s["file"]).read_bytes()[:CONTENT_PREVIEW_BYTES].decode("utf-8", errors="ignore")
        return raw.replace("\n", " ").strip() == s.get("snippet", "")
    except OSError:
        return False


async def build(
    prompt: str,
    intent: str,
    complexity: str,
    task_id: str = "",
    allowed_files: Optional[list[str]] = None,
) -> BuiltContext:
    """
    Build enriched context for one request.

    Runs all three retrievals concurrently via asyncio.gather with
    return_exceptions=True so a failure in one component does not prevent
    the others from completing.
    """
    new_fps: dict[str, str] = {}
    scan_gate: set[str] | None = None
    cached_snippets: list[dict] = []

    if task_id and allowed_files:
        from ..config.settings import settings
        from ..db import redis as session_store

        repo_root = Path(settings.repo_path).resolve()

        old_fps = await session_store.get_fingerprints(task_id) or {}

        new_fps = await asyncio.to_thread(_compute_fingerprints, allowed_files, repo_root)

        changed = {f for f in allowed_files if new_fps.get(f) != old_fps.get(f)}

        # Carry forward cached snippets for unchanged files, with content validation
        old_snippets = await session_store.get_snippets(task_id)
        invalidated: set[str] = set()
        cached_snippets = []
        for s in old_snippets:
            if s.get("file") in changed:
                continue
            if not _validate_snippet(s, repo_root):
                logger.warning("Cache content mismatch for %s — forcing re-scan", s.get("file"))
                invalidated.add(s.get("file", ""))
            else:
                cached_snippets.append(s)
        # empty set → no files changed/invalidated, scan returns [] immediately via early-exit in repo.scan
        scan_gate = changed | invalidated

        await session_store.save_fingerprints(task_id, new_fps, settings.redis_session_ttl_s)
    elif allowed_files is not None:
        scan_gate = set(allowed_files)

    few_shots_result, repo_snippets_result, hints_result = await asyncio.gather(
        few_shot.search(prompt),
        repo.scan(prompt, intent, allowed_files=scan_gate),
        hints.generate(intent, complexity),
        return_exceptions=True,
    )

    if isinstance(few_shots_result, Exception):
        logger.warning("Few-shot retrieval raised: %s", few_shots_result)
        few_shots_result = []

    if isinstance(repo_snippets_result, Exception):
        logger.warning("Repo scan raised: %s", repo_snippets_result)
        repo_snippets_result = []

    if isinstance(hints_result, Exception):
        logger.warning("Hint generation raised: %s", hints_result)
        hints_result = ""

    # Merge fresh snippets with cached snippets for unchanged files
    fresh_dicts = [s.model_dump() for s in repo_snippets_result]
    merged_snippets = cached_snippets + fresh_dicts

    if task_id:
        from ..config.settings import settings
        from ..db import redis as session_store
        await session_store.save_snippets(task_id, merged_snippets, settings.redis_session_ttl_s)

    return BuiltContext(
        few_shots=few_shots_result,
        repo_snippets=[RepoSnippet(**s) for s in merged_snippets],
        system_hints=hints_result,
        task_id=task_id,
        allowed_files=allowed_files or [],
        file_fingerprints=new_fps,
    )
