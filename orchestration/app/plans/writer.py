"""Atomic markdown write-back for Plan objects."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from .schema import Plan, PlanStep

_STATUS_EMOJI = {
    "pending": "[ ]",
    "in_progress": "[~]",
    "done": "[x]",
    "blocked": "[!]",
}

_VALID_TRANSITIONS: dict[str, set[str]] = {
    "pending":     {"in_progress", "blocked"},
    "in_progress": {"done", "blocked"},
    "blocked":     {"in_progress"},
    "done":        set(),
}


def validate_transition(old: Plan, new: Plan) -> None:
    """Raise ValueError if any step status transition is illegal."""
    old_by_id = {s.id: s for s in old.steps}
    for new_step in new.steps:
        old_step = old_by_id.get(new_step.id)
        if old_step is None:
            continue
        if old_step.status == new_step.status:
            continue
        allowed = _VALID_TRANSITIONS.get(old_step.status, set())
        if new_step.status not in allowed:
            raise ValueError(
                f"Illegal status transition for step {new_step.id!r}: "
                f"{old_step.status!r} → {new_step.status!r}"
            )


def render_plan(plan: Plan) -> str:
    """Render a Plan back to canonical markdown."""
    lines = []
    lines.append(f"## Task Goal\n\n{plan.goal}\n")
    lines.append("## Steps\n")
    for i, step in enumerate(plan.steps, 1):
        marker = _STATUS_EMOJI.get(step.status, "[ ]")
        lines.append(f"{i}. {marker} {step.description}")
        if step.evidence:
            lines.append(f"   Evidence: {step.evidence}")
        if step.files_touched:
            lines.append(f"   Files: {', '.join(step.files_touched)}")
    lines.append("")
    return "\n".join(lines)


def write_plan_atomic(path: Path, plan: Plan) -> None:
    """Write plan markdown atomically using filelock + os.replace."""
    import filelock

    lock_path = str(path) + ".lock"
    tmp_path = str(path) + ".tmp"

    with filelock.FileLock(lock_path, timeout=30):
        Path(tmp_path).write_text(render_plan(plan), encoding="utf-8")
        os.replace(tmp_path, path)
