"""
Agent output signal extraction.

Scans the most recent non-tool agent output in iteration_history using
compiled regex patterns to surface routing-relevant signals.  These signals
are injected into the LLM supervisor prompt so the model can make
context-aware routing decisions without reading raw agent text.

All functions are pure — no I/O, no side effects.  Patterns are compiled
at module load time for performance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class AgentSignals:
    """Extracted routing signals from the most recent agent output."""

    role: str = ""
    output_length: int = 0
    score_value: Optional[float] = None   # numeric SCORE: 0.xx if present
    has_errors: bool = False              # error/exception/traceback/failure
    has_completion: bool = False          # done/fixed/implemented/resolved
    has_uncertainty: bool = False         # unsure/unclear/need more info
    has_code_changes: bool = False        # wrote/modified/created a file or function
    has_blockers: bool = False            # blocked/cannot proceed/missing dependency

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "output_length": self.output_length,
            "score_value": self.score_value,
            "has_errors": self.has_errors,
            "has_completion": self.has_completion,
            "has_uncertainty": self.has_uncertainty,
            "has_code_changes": self.has_code_changes,
            "has_blockers": self.has_blockers,
        }


# ── Compiled patterns ─────────────────────────────────────────────────────────

_SCORE_RE = re.compile(r"\bSCORE[:\s]+([0-9]+\.[0-9]+)", re.IGNORECASE)

_ERROR_RE = re.compile(
    r"\b(error|exception|traceback|failed|failure|crash|fatal|abort)\b",
    re.IGNORECASE,
)

_COMPLETION_RE = re.compile(
    r"\b(done|complete|completed|fixed|implemented|resolved|finished|successfully)\b",
    re.IGNORECASE,
)

_UNCERTAINTY_RE = re.compile(
    r"(unsure|unclear|uncertain|cannot determine|need more|not sure|might need"
    r"|I don'?t know)",
    re.IGNORECASE,
)

_CODE_CHANGE_RE = re.compile(
    r"\b(wrote|modified|created|updated|added|deleted|changed|refactored)\s+"
    r"(file|function|class|method|module|test|script|config)\b",
    re.IGNORECASE,
)

_BLOCKER_RE = re.compile(
    r"\b(blocked|cannot proceed|impossible|missing dependency|not possible"
    r"|unable to proceed|can't proceed)\b",
    re.IGNORECASE,
)


# ── Extraction ────────────────────────────────────────────────────────────────


def extract_signals(state: Dict[str, Any]) -> AgentSignals:
    """
    Extract routing signals from the most recent non-tool agent output.

    Tool result entries (role='tool') are skipped — they are infrastructure
    output, not agent-generated content.  Returns a zero-value AgentSignals
    if no agent history entry is present.
    """
    history: List[Dict[str, Any]] = state.get("iteration_history") or []
    agent_entries = [h for h in history if h.get("role") != "tool"]
    if not agent_entries:
        return AgentSignals()

    last = agent_entries[-1]
    role = str(last.get("role") or "")
    output: str = str(last.get("output") or last.get("full_output") or "")

    score_value: Optional[float] = None
    m = _SCORE_RE.search(output)
    if m:
        try:
            score_value = float(m.group(1))
        except ValueError:
            pass

    return AgentSignals(
        role=role,
        output_length=len(output),
        score_value=score_value,
        has_errors=bool(_ERROR_RE.search(output)),
        has_completion=bool(_COMPLETION_RE.search(output)),
        has_uncertainty=bool(_UNCERTAINTY_RE.search(output)),
        has_code_changes=bool(_CODE_CHANGE_RE.search(output)),
        has_blockers=bool(_BLOCKER_RE.search(output)),
    )


def format_signals_for_prompt(signals: AgentSignals) -> str:
    """
    Render AgentSignals as a compact Markdown section for the supervisor prompt.

    Returns an empty string when no agent history is available (role is empty).
    """
    if not signals.role:
        return ""

    active: List[str] = []
    if signals.score_value is not None:
        active.append(f"score={signals.score_value:.2f}")
    if signals.has_completion:
        active.append("completion_detected")
    if signals.has_errors:
        active.append("errors_detected")
    if signals.has_uncertainty:
        active.append("uncertainty_detected")
    if signals.has_code_changes:
        active.append("code_changes_detected")
    if signals.has_blockers:
        active.append("blockers_detected")
    if not active:
        active.append("no_strong_signals")

    return (
        f"## Output Signals (last_role={signals.role}, "
        f"output_len={signals.output_length})\n"
        + ", ".join(active)
    )
