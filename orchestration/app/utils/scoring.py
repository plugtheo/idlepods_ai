"""
Inline scoring utilities
=========================
Extracts quality signals from agent text outputs and computes a
per-iteration score used by the convergence check.

Design principle: CPU-only, no external calls, no ML models.
The scorer reads well-known output patterns from reviewer/critic agents
and falls back to heuristic estimates when those patterns are absent.

Score bands
-----------
0.0 – 0.4   Poor: reviewer/critic found blockers, or output is very short.
0.4 – 0.7   Acceptable: output present but reviewer found issues.
0.7 – 0.85  Good: reviewer scored it reasonably, few objections.
0.85 – 1.0  Excellent: score above threshold, ready to converge.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

# Regex to extract "SCORE: 0.82" style annotations from reviewer / critic output.
_SCORE_RE = re.compile(r"\bSCORE\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)

# Structural markers for code presence — used to reward coder/debugger outputs
_CODE_PRESENT_RE = re.compile(r"```|def |class |import |\bfunction\b|\breturn\b", re.I)

# Orchestration metadata leakage — adapter trained on pipeline experience data
# outputs score/metrics JSON instead of code.  Detect and heavily penalise.
_METADATA_LEAKAGE_RE = re.compile(
    r"'agent_name'\s*:|\"agent_name\"\s*:|'iteration_number'\s*:|\"iteration_number\"\s*:"
    r"|'quality_score'\s*:|\"quality_score\"\s*:|'execution_time_ms'\s*:|\"execution_time_ms\"\s*:"
    r"|'session_id'\s*:|\"session_id\"\s*:|'final_output'\s*:|\"final_output\"\s*:"
    r"|'agent_chain'\s*:|\"agent_chain\"\s*:|'contributions'\s*:|\"contributions\"\s*:"
    r"|'iteration_scores'\s*:|\"iteration_scores\"\s*:|'final_score'\s*:|\"final_score\"\s*:"
    r"|'converged'\s*:|\"converged\"\s*:",
    re.I,
)

# Negative signals that lower the heuristic score
_BLOCKER_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bBLOCKERS?\s*[:=](?!\s*none)",
        r"\bCRITICAL\s+(?:ISSUE|BUG|ERROR)\b",
        r"\bFAILS?\s+(?:to|the|all)\b",
        r"\bDOES\s+NOT\s+WORK\b",
    ]
]

# Positive signals that raise the heuristic score
_POSITIVE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bLOOKS\s+GOOD\b",
        r"\bWELL\s+(?:STRUCTURED|WRITTEN|TESTED)\b",
        r"\bNO\s+ISSUES?\b",
        r"\bBLOCKERS?\s*[:=]\s*(?:None|none|N\/A)\b",
    ]
]


_EVALUATOR_REQUIRED_FIELDS: Dict[str, List[str]] = {
    "reviewer": ["ISSUES", "SUGGESTIONS"],
    "critic":   ["BLOCKERS", "IMPROVEMENT"],
}


def _required_fields_present(text: str, role: str) -> bool:
    """Return True if all non-SCORE structured fields for *role* appear in *text*."""
    fields = _EVALUATOR_REQUIRED_FIELDS.get(role, [])
    return all(re.search(rf"(?m)^{field}:", text) for field in fields)


def extract_score_from_text(text: str) -> float | None:
    """
    Try to read an explicit SCORE annotation from *text*.

    Returns the float if found and in [0, 1], otherwise None.
    """
    match = _SCORE_RE.search(text)
    if match:
        value = float(match.group(1))
        if 0.0 <= value <= 1.0:
            return value
        if 0 < value <= 10:
            return value / 10.0  # convert 0–10 scale to 0–1
    return None


# Heuristic baseline scores — module constants so the values are searchable
# and self-documenting.  Not promoted to settings because they are tied to the
# specific output patterns the regexes below detect, not to deployment policy.
_SCORE_SHORT_TEXT = 0.30          # near-empty output
_SCORE_METADATA_LEAK = 0.10       # adapter outputting pipeline JSON instead of code
_SCORE_EVALUATOR_HEURISTIC = 0.55 # reviewer/critic present but no explicit SCORE field
_SCORE_CODER_NO_CODE = 0.60       # coder/debugger output with no code markers
_SCORE_CODER_BASE = 0.65          # coder/debugger base score when code is present
_SCORE_CODER_MAX = 0.75           # cap for length-boosted coder score
_CODER_LENGTH_SCALE = 12000       # characters-per-unit for the length bonus
_SCORE_GENERIC_HEURISTIC = 0.62   # planner, researcher, consensus baseline
_BLOCKER_SCORE_PENALTY = 0.12     # subtracted per matched blocker pattern
_POSITIVE_SCORE_BONUS = 0.06      # added per matched positive pattern


def heuristic_score(text: str, role: str) -> float:
    """
    Estimate quality of *text* using lightweight pattern matching.

    Reviewer and critic output is scored explicitly (SCORE: N.NN).
    For generative roles (coder, planner, etc.) a heuristic is applied.

    Role-aware baselines (when no explicit SCORE annotation):
      reviewer / critic — 0.55: evaluative roles that omit the SCORE field
                          produced an incomplete response; stay well below the
                          convergence threshold to force a re-evaluation.
      coder / debugger  — 0.60–0.75: reward code presence and output length
                          since these roles don't emit scores by design.
      planner / others  — 0.62: modest baseline for prose-only roles.
    """
    if not text or len(text.strip()) < 30:
        return _SCORE_SHORT_TEXT  # Near-empty output is always low quality

    # Detect adapter metadata leakage before honoring any explicit annotation.
    if _METADATA_LEAKAGE_RE.search(text):
        return _SCORE_METADATA_LEAK

    # Explicit score annotations — only honored when required structured fields
    # are present for evaluator roles (prevents a bare SCORE: bypassing the gate).
    if role not in ("reviewer", "critic") or _required_fields_present(text, role):
        explicit = extract_score_from_text(text)
        if explicit is not None:
            return explicit

    # Role-specific baseline when no explicit SCORE annotation
    if role in ("reviewer", "critic"):
        # These roles SHOULD produce SCORE: — absence means incomplete output
        score = _SCORE_EVALUATOR_HEURISTIC
    elif role in ("coder", "debugger"):
        has_code = bool(_CODE_PRESENT_RE.search(text))
        length = len(text.strip())
        # Reward code presence + output substance
        score = _SCORE_CODER_NO_CODE if not has_code else min(_SCORE_CODER_MAX, _SCORE_CODER_BASE + length / _CODER_LENGTH_SCALE)
    else:
        # planner, researcher, consensus
        score = _SCORE_GENERIC_HEURISTIC

    # Apply negative signals
    for pattern in _BLOCKER_PATTERNS:
        if pattern.search(text):
            score -= _BLOCKER_SCORE_PENALTY

    # Apply positive signals
    for pattern in _POSITIVE_PATTERNS:
        if pattern.search(text):
            score += _POSITIVE_SCORE_BONUS

    return max(0.0, min(1.0, score))


def score_iteration(iteration_history: List[Dict[str, Any]], current_iteration: int) -> float:
    """
    Compute a quality score for the most recent iteration.

    Priority order:
    1. Explicit SCORE annotations from reviewer/critic — most reliable.
    2. Blend evaluator heuristic + generative agent (coder/debugger) heuristic
       when evaluators ran but produced no explicit SCORE (e.g. broken adapter).
    3. Heuristic average of all agents when no evaluators ran.

    Parameters
    ----------
    iteration_history:
        Full history list from AgentState.
    current_iteration:
        Index of the iteration just completed (1-based).

    Returns
    -------
    float in [0, 1]
    """
    this_iteration = [
        entry for entry in iteration_history
        if entry.get("iteration") == current_iteration
    ]

    if not this_iteration:
        return 0.0

    evaluator_roles = {"reviewer", "critic"}
    generative_roles = {"coder", "debugger"}

    # Separate evaluators that gave an EXPLICIT score from those that didn't.
    explicit_eval_scores: List[float] = []
    heuristic_eval_scores: List[float] = []
    for e in this_iteration:
        if e.get("role") not in evaluator_roles:
            continue
        # Prefer full_output (if stored) so SCORE: annotations aren't trimmed.
        text = e.get("full_output") or e.get("output", "")
        role_e = e["role"]
        explicit = extract_score_from_text(text) if _required_fields_present(text, role_e) else None
        if explicit is not None:
            explicit_eval_scores.append(explicit)
        else:
            heuristic_eval_scores.append(heuristic_score(text, role_e))

    # 1. Explicit evaluator scores — highest confidence signal.
    if explicit_eval_scores:
        return sum(explicit_eval_scores) / len(explicit_eval_scores)

    # 2. Evaluators ran but gave no explicit SCORE (broken adapter, wrong format).
    #    Blend with generative agent quality — coder heuristic rewards good code.
    gen_scores = [
        heuristic_score(e.get("full_output") or e.get("output", ""), e.get("role", ""))
        for e in this_iteration
        if e.get("role") in generative_roles
    ]
    if heuristic_eval_scores and gen_scores:
        blended = heuristic_eval_scores + gen_scores
        return sum(blended) / len(blended)
    if gen_scores:
        return sum(gen_scores) / len(gen_scores)
    if heuristic_eval_scores:
        return sum(heuristic_eval_scores) / len(heuristic_eval_scores)

    # 3. No evaluator agents ran — average all agent scores.
    all_scores = [
        heuristic_score(e.get("full_output") or e.get("output", ""), e.get("role", ""))
        for e in this_iteration
    ]
    return sum(all_scores) / len(all_scores)


def score_per_entry(history_entry: dict) -> float:
    """
    Compute a quality score for a single history entry.

    For evaluator roles (reviewer, critic) the explicit SCORE annotation is
    preferred; heuristic is used as fallback.  For all other roles a
    heuristic score based on output content and role is returned.

    Parameters
    ----------
    history_entry:
        One dict from AgentState's ``iteration_history`` list.

    Returns
    -------
    float in [0, 1]
    """
    role = history_entry.get("role", "")
    text = history_entry.get("full_output") or history_entry.get("output", "")

    if role in ("reviewer", "critic") and _required_fields_present(text, role):
        explicit = extract_score_from_text(text)
        if explicit is not None:
            return explicit

    return heuristic_score(text, role)


def validate_output(text: str, role: str) -> tuple[bool, list[str]]:
    """
    Run all post-generation validation rules against *text*.

    Returns (is_valid, reasons).  When not valid, the caller should replace
    the output with a sentinel so downstream agents are not polluted; the
    original full_output should be retained for accurate scoring.
    """
    reasons: list[str] = []
    if not text or len(text.strip()) < 30:
        reasons.append("short_text")
    if _METADATA_LEAKAGE_RE.search(text):
        reasons.append("metadata_leakage")
    if role in _EVALUATOR_REQUIRED_FIELDS and not _required_fields_present(text, role):
        reasons.append("missing_required_fields")
    return len(reasons) == 0, reasons
