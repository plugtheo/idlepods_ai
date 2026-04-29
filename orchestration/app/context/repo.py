"""
Repository context retrieval
==============================
Scans the local repository for code files semantically relevant to the
current prompt.  Only activated for CODING, DEBUGGING, and ANALYSIS intents,
and only when the prompt signals that existing codebase context is needed.

File-system scan runs in asyncio.to_thread() to avoid blocking the event loop.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from pathlib import Path
from typing import List

from shared.contracts.context import RepoSnippet
from ..config.settings import settings

logger = logging.getLogger(__name__)

_CODE_INTENTS = {"coding", "debugging", "analysis"}
_EXTENSIONS = {".py", ".ts", ".js", ".md"}
_MAX_FILES = 200

_EXISTING_CODE_SIGNALS = re.compile(
    r"\b(fix|debug|update|modify|change|refactor|review|improve|optimize|"
    r"existing|current|where|which\s+file|"
    r"how\s+does|what\s+does|why\s+does|why\s+is|why\s+are|"
    r"our\s+code|our\s+codebase|our\s+project|our\s+repo|"
    r"the\s+codebase|this\s+codebase|the\s+repo|this\s+repo|the\s+project)\b",
    re.IGNORECASE,
)
_FILE_EXT_SIGNAL = re.compile(r"\b\w+\.(py|ts|js|md)\b")


def _tokenise(text: str) -> set[str]:
    return set(re.findall(r"[a-z_]+", text.lower()))


def _should_scan_repo(prompt: str) -> bool:
    return bool(
        _EXISTING_CODE_SIGNALS.search(prompt) or _FILE_EXT_SIGNAL.search(prompt)
    )


def _file_fingerprint(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return ""


def _scan_files_sync(
    prompt_tokens: set[str],
    repo_root: Path,
    max_snippets: int,
    allowed_files: set[str] | None = None,
) -> list[RepoSnippet]:
    candidates: list[tuple[float, Path]] = []
    file_count = 0

    for path in repo_root.rglob("*"):
        if not path.is_file() or path.suffix not in _EXTENSIONS:
            continue
        if "__pycache__" in path.parts or ".venv" in path.parts:
            continue

        if allowed_files is not None:
            try:
                rel = path.relative_to(repo_root).as_posix()
            except ValueError:
                continue
            if rel not in allowed_files:
                continue

        if file_count >= _MAX_FILES:
            break
        file_count += 1

        try:
            content_preview = path.read_bytes()[:300].decode("utf-8", errors="ignore")
        except OSError:
            content_preview = ""
        fp_tokens = _tokenise(path.stem + " " + content_preview)
        overlap = len(prompt_tokens & fp_tokens)
        if overlap > 0 or allowed_files is not None:
            relevance = overlap / max(len(prompt_tokens), len(fp_tokens), 1)
            candidates.append((max(relevance, 0.001), path))

    candidates.sort(key=lambda x: x[0], reverse=True)
    snippets: list[RepoSnippet] = []

    for relevance, path in candidates[:max_snippets]:
        try:
            raw = path.read_text(encoding="utf-8", errors="ignore")
            snippet = raw[:300].replace("\n", " ").strip()
            rel_path = path.relative_to(repo_root).as_posix()
            snippets.append(
                RepoSnippet(file=rel_path, snippet=snippet, relevance=round(relevance, 3))
            )
        except OSError:
            continue

    logger.info("Repo scan: files_walked=%d  returned=%d", file_count, len(snippets))
    return snippets


async def scan(
    prompt: str, intent: str, allowed_files: set[str] | None = None
) -> List[RepoSnippet]:
    """
    Return up to settings.max_repo_snippets relevant file snippets.

    When ``allowed_files`` is an empty set, returns immediately with no results.
    When ``allowed_files`` is a non-empty set, bypasses intent/prompt heuristics
    and scans only the listed paths.
    When ``allowed_files`` is None, applies the normal intent + prompt guards.
    """
    if allowed_files is not None and len(allowed_files) == 0:
        return []

    if allowed_files is None:
        if intent.lower() not in _CODE_INTENTS:
            return []
        if not _should_scan_repo(prompt):
            logger.debug("Repo scan skipped — prompt does not reference existing code.")
            return []

    repo_root = Path(settings.repo_path).resolve()
    if not repo_root.exists():
        logger.warning("Repo path does not exist: %s", repo_root)
        return []

    prompt_tokens = _tokenise(prompt)
    return await asyncio.to_thread(
        _scan_files_sync, prompt_tokens, repo_root, settings.max_repo_snippets, allowed_files
    )
