"""
Repository context retrieval
==============================
Scans the local repository for code files that are semantically relevant
to the current prompt.  Only activated for code-related intents (coding,
debugging, analysis) **and** only when the prompt signals that existing
codebase context is actually needed (e.g. "fix", "debug", "update",
"refactor" an existing thing, or a specific file is referenced).

Prompts that ask to create something new from scratch ("implement a
rate-limiter", "write a parser") skip the scan entirely — there is no
existing code to surface.

Approach
--------
1. Reject if intent is not code-related.
2. Reject if the prompt does not contain signals that imply existing code
   context (see `_should_scan_repo`).
3. Walk all .py / .ts / .js / .md files under `settings.repo_path` in a
   thread-pool executor so the async event loop is not blocked.
4. For each file, build a lightweight fingerprint from filename + first-line
   comment.  (Full-file embedding is too slow per-request.)
5. Keyword overlap between token-set(prompt) and token-set(fingerprint)
   selects candidate files.
6. Top-`settings.max_repo_snippets` candidates are returned with their
   first 300 chars as the snippet.

This is intentionally simple — the goal is fast, good-enough context
injection without adding a per-request embedding call per file.
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import List

from shared.contracts.context import RepoSnippet
from ..config.settings import settings

logger = logging.getLogger(__name__)

# Intents that warrant repo scanning (subject to prompt signal check below)
_CODE_INTENTS = {"coding", "debugging", "analysis"}

# File extensions to consider
_EXTENSIONS = {".py", ".ts", ".js", ".md"}

# Maximum files to walk (safety cap for very large repos)
_MAX_FILES = 200

# Patterns that signal the prompt is about *existing* code — fix/debug/
# update/modify/change/refactor/review/improve/optimize, investigative questions,
# explicit "existing/current/our" qualifiers, or a file extension reference.
# NOTE: bare "find" is intentionally excluded — it is too broad
# ("find the factorial of n") and the remaining patterns cover all legitimate
# "find where X is in the codebase" cases via "where", "which file", etc.
_EXISTING_CODE_SIGNALS = re.compile(
    r"\b(fix|debug|update|modify|change|refactor|review|improve|optimize|"
    r"existing|current|where|which\s+file|"
    r"how\s+does|what\s+does|why\s+does|why\s+is|why\s+are|"
    r"our\s+code|our\s+codebase|our\s+project|our\s+repo|"
    r"the\s+codebase|this\s+codebase|the\s+repo|this\s+repo|the\s+project)\b",
    re.IGNORECASE,
)

# A file extension reference (e.g. "utils.py", "auth.ts") also signals existing code
_FILE_EXT_SIGNAL = re.compile(r"\b\w+\.(py|ts|js|md)\b")


def _tokenise(text: str) -> set[str]:
    """Lowercase word tokens from text (strips punctuation)."""
    return set(re.findall(r"[a-z_]+", text.lower()))


def _should_scan_repo(prompt: str) -> bool:
    """
    Return True only when the prompt signals that existing codebase context
    is relevant — e.g. the user is asking to fix, debug, or find something
    that already exists, or has referenced a specific file.

    Prompts that ask to create something new from scratch are rejected so the
    repo walk is skipped entirely for those requests.
    """
    return bool(
        _EXISTING_CODE_SIGNALS.search(prompt) or _FILE_EXT_SIGNAL.search(prompt)
    )


def _file_fingerprint(path: Path) -> str:
    """First non-empty line of the file (usually a docstring or comment)."""
    try:
        with path.open(encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped and not stripped.startswith("#!"):
                    return stripped[:120]
    except OSError:
        pass
    return ""


def _scan_files_sync(
    prompt_tokens: set[str], repo_root: Path, max_snippets: int
) -> list[RepoSnippet]:
    """
    Synchronous file-system walk — called via run_in_executor so the async
    event loop is not blocked during I/O-bound directory traversal and file
    reads.
    """
    candidates: list[tuple[float, Path]] = []
    file_count = 0

    for path in repo_root.rglob("*"):
        if not path.is_file() or path.suffix not in _EXTENSIONS:
            continue
        if "__pycache__" in path.parts or ".venv" in path.parts:
            continue
        if file_count >= _MAX_FILES:
            break
        file_count += 1

        fp_tokens = _tokenise(path.stem + " " + _file_fingerprint(path))
        overlap = len(prompt_tokens & fp_tokens)
        if overlap > 0:
            relevance = overlap / max(len(prompt_tokens), len(fp_tokens), 1)
            candidates.append((relevance, path))

    candidates.sort(key=lambda x: x[0], reverse=True)
    snippets: list[RepoSnippet] = []

    for relevance, path in candidates[:max_snippets]:
        try:
            raw = path.read_text(encoding="utf-8", errors="ignore")
            snippet = raw[:300].replace("\n", " ").strip()
            # Use as_posix() to ensure forward slashes regardless of host OS
            rel_path = path.relative_to(repo_root).as_posix()
            snippets.append(
                RepoSnippet(file=rel_path, snippet=snippet, relevance=round(relevance, 3))
            )
        except OSError:
            continue

    logger.info(
        "Repo scan: files_walked=%d  returned=%d",
        file_count, len(snippets),
    )
    return snippets


async def retrieve_repo_snippets(prompt: str, intent: str) -> List[RepoSnippet]:
    """
    Return up to `settings.max_repo_snippets` relevant file snippets.

    Returns an empty list if the intent is not code-related, or if the
    prompt does not contain signals indicating that existing codebase
    context is needed, or if the repo path is unavailable.
    """
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
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, _scan_files_sync, prompt_tokens, repo_root, settings.max_repo_snippets
    )
