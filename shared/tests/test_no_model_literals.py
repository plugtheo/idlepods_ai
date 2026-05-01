"""
Grep guard: asserts that no Python source file under inference/, orchestration/,
training/, shared/, or scripts/ contains hardcoded model-name strings that the
backend registry was designed to eliminate.

Failing this test means Plan A (Backend Registry) is incomplete.
"""

import os
import re
from pathlib import Path

# Directories to scan (relative to project root)
_SCAN_DIRS = ["inference", "orchestration", "training", "shared", "scripts"]

# Literals that must not appear (case-insensitive)
_BANNED = re.compile(r"\b(qwen|deepseek|mistral)\b", re.IGNORECASE)

# Files allowed to contain these strings (by absolute-path suffix match)
_ALLOWLIST = {
    # This file itself references them as banned patterns
    os.path.join("shared", "tests", "test_no_model_literals.py"),
    # _LEGACY_BACKEND_ALIASES in contracts/inference.py has these as map keys (one-release shim)
    os.path.join("shared", "contracts", "inference.py"),
    # The models.py registry loader may reference them as example values
    os.path.join("shared", "contracts", "models.py"),
    # models.yaml is YAML, not Python — not scanned here
}


def _project_root() -> Path:
    return Path(__file__).parents[2]


def test_no_hardcoded_model_name_literals():
    root = _project_root()
    violations: list[str] = []

    for scan_dir in _SCAN_DIRS:
        base = root / scan_dir
        if not base.exists():
            continue
        for py_file in base.rglob("*.py"):
            # Skip test directories and cache dirs
            parts = py_file.relative_to(root).parts
            if any(p in ("tests", "__pycache__", ".claude") for p in parts):
                continue
            # Check allowlist by suffix
            rel = str(py_file.relative_to(root))
            if any(rel.endswith(allowed) or rel == allowed for allowed in _ALLOWLIST):
                continue
            text = py_file.read_text(encoding="utf-8", errors="replace")
            for lineno, line in enumerate(text.splitlines(), start=1):
                if _BANNED.search(line):
                    violations.append(f"{rel}:{lineno}: {line.strip()}")

    assert not violations, (
        "Hardcoded model-name literals found — update to use the backend registry:\n"
        + "\n".join(violations[:40])
        + ("\n... (truncated)" if len(violations) > 40 else "")
    )
