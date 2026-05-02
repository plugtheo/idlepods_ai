"""Parse plans/current-task.md → Plan."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..tools.runner import ToolError, _safe_path
from .schema import Plan, PlanStep

_PLANS_ROOT = "plans"

_GOAL_RE = re.compile(r"^##\s+Task\s+Goal\s*$", re.IGNORECASE | re.MULTILINE)
_STEPS_RE = re.compile(r"^##\s+Steps\s*", re.IGNORECASE | re.MULTILINE)
_SECTION_RE = re.compile(r"^##\s+", re.MULTILINE)
_NUMBERED_RE = re.compile(r"^\s*(\d+)\.\s+(.+)$")
_STATUS_MARKER_RE = re.compile(r"^\[[~x! ]\]\s+")


def _safe_plans_path(raw: str) -> Path:
    p = _safe_path(raw)
    parts = p.parts
    if not parts or parts[0] != _PLANS_ROOT:
        raise ToolError(
            f"Plan path must be rooted under '{_PLANS_ROOT}/': got {raw!r}"
        )
    return p


def parse_plan(text: str, source_path: str = "plans/current-task.md") -> Plan:
    """Parse plan markdown → Plan. Raises ValueError on unrecognisable format."""
    goal = _extract_goal(text)
    steps = _extract_steps(text)
    now = datetime.now(timezone.utc)
    return Plan(goal=goal, steps=steps, created_at=now, updated_at=now)


def read_plan(path: str) -> Optional[Plan]:
    """Read and parse a plan markdown file. Returns None if file is missing."""
    resolved = _safe_plans_path(path)
    if not resolved.exists():
        return None
    text = resolved.read_text(encoding="utf-8")
    return parse_plan(text, source_path=path)


# ── internal helpers ─────────────────────────────────────────────────────────


def _extract_goal(text: str) -> str:
    m = _GOAL_RE.search(text)
    if not m:
        first_line = text.strip().splitlines()[0] if text.strip() else ""
        return first_line.lstrip("#").strip() or "Unnamed task"

    after = text[m.end():]
    next_section = _SECTION_RE.search(after)
    block = after[: next_section.start()].strip() if next_section else after.strip()
    return block.splitlines()[0].strip() if block else "Unnamed task"


def _extract_steps(text: str) -> list[PlanStep]:
    m = _STEPS_RE.search(text)
    if not m:
        return []

    after = text[m.end():]
    next_section = _SECTION_RE.search(after)
    block = after[: next_section.start()] if next_section else after

    steps: list[PlanStep] = []
    for line in block.splitlines():
        nm = _NUMBERED_RE.match(line)
        if nm:
            n = int(nm.group(1))
            desc = _STATUS_MARKER_RE.sub("", nm.group(2).strip())
            steps.append(PlanStep(id=f"step-{n}", description=desc))

    return steps
