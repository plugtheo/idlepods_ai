"""
Supervisor routing telemetry.

Computes and logs structured routing statistics from the supervisor_decisions
log so ops can monitor routing quality, LLM utilization, and fallback rates
without parsing free-text log lines.

Terminology:
  fallback    — a decision where the LLM was attempted but failed (budget
                exhausted, inference exception, no/bad tool call).  Marked by
                rule suffixes in _FALLBACK_SUFFIXES.
  guard       — a deliberate deterministic delegation (e.g. HybridSupervisor
                handling R3).  Not a failure — marked by _GUARD_SUFFIXES.
  shortcircuit — forced dispatch to a single legal target with no LLM call.

All computation functions are pure — no I/O.
log_routing_stats() is the only function with side effects.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Rule-suffix classifiers ────────────────────────────────────────────────────

_FALLBACK_SUFFIXES: frozenset[str] = frozenset({
    "_budget_fallback",
    "_exc_fallback",
    "_llm_fallback",
    "_notc_fallback",
    "_badpick_fallback",
})

_GUARD_SUFFIXES: frozenset[str] = frozenset({
    "_hybrid_guard",
})


def _has_any_suffix(rule: str, suffixes: frozenset) -> bool:
    return any(rule.endswith(s) for s in suffixes)


# ── RoutingStats ──────────────────────────────────────────────────────────────


@dataclass
class RoutingStats:
    """Aggregated statistics over a window of supervisor decisions."""

    total_decisions: int = 0
    llm_called_count: int = 0         # decisions where an LLM call was made
    fallback_count: int = 0           # LLM was attempted but fell back (failure)
    guard_count: int = 0              # deliberate deterministic delegations
    shortcircuit_count: int = 0       # single-target forced dispatch
    rule_distribution: Dict[str, int] = field(default_factory=dict)
    avg_confidence: Optional[float] = None   # mean confidence from LLM decisions
    avg_decide_ms: Optional[float] = None    # mean wall-clock time per decision
    max_decide_ms: Optional[float] = None    # worst-case wall-clock time

    @property
    def llm_utilization_rate(self) -> float:
        """Fraction of decisions that made an LLM call."""
        return self.llm_called_count / self.total_decisions if self.total_decisions else 0.0

    @property
    def fallback_rate(self) -> float:
        """Fraction of decisions that fell back from a failed LLM attempt."""
        return self.fallback_count / self.total_decisions if self.total_decisions else 0.0

    @property
    def shortcircuit_rate(self) -> float:
        """Fraction of decisions that shortcircuited (single forced target)."""
        return self.shortcircuit_count / self.total_decisions if self.total_decisions else 0.0


# ── Computation ───────────────────────────────────────────────────────────────


def compute_routing_stats(decisions: List[Dict[str, Any]]) -> RoutingStats:
    """
    Compute RoutingStats from a list of supervisor_decisions log entries.

    Pure function — safe to call with any slice of the supervisor_decisions list.
    """
    if not decisions:
        return RoutingStats()

    total = len(decisions)

    llm_called = sum(
        1 for d in decisions
        if (d.get("metadata") or {}).get("llm_called") is True
    )
    shortcircuits = sum(
        1 for d in decisions
        if (d.get("metadata") or {}).get("shortcircuit") is True
    )
    fallbacks = sum(
        1 for d in decisions
        if _has_any_suffix(d.get("rule", ""), _FALLBACK_SUFFIXES)
    )
    guards = sum(
        1 for d in decisions
        if _has_any_suffix(d.get("rule", ""), _GUARD_SUFFIXES)
    )

    rule_dist: Dict[str, int] = {}
    for d in decisions:
        rule = d.get("rule", "unknown")
        rule_dist[rule] = rule_dist.get(rule, 0) + 1

    confidences = [
        d["metadata"]["confidence"]
        for d in decisions
        if (d.get("metadata") or {}).get("confidence") is not None
    ]
    avg_confidence = sum(confidences) / len(confidences) if confidences else None

    elapsed_vals = [
        d["metadata"]["decide_ms"]
        for d in decisions
        if (d.get("metadata") or {}).get("decide_ms") is not None
    ]
    avg_decide_ms = sum(elapsed_vals) / len(elapsed_vals) if elapsed_vals else None
    max_decide_ms = max(elapsed_vals) if elapsed_vals else None

    return RoutingStats(
        total_decisions=total,
        llm_called_count=llm_called,
        fallback_count=fallbacks,
        guard_count=guards,
        shortcircuit_count=shortcircuits,
        rule_distribution=rule_dist,
        avg_confidence=avg_confidence,
        avg_decide_ms=avg_decide_ms,
        max_decide_ms=max_decide_ms,
    )


# ── Logging ───────────────────────────────────────────────────────────────────


def log_routing_stats(
    stats: RoutingStats,
    session_id: str,
    *,
    at_level: str = "debug",
) -> None:
    """
    Emit a single structured log line summarising routing statistics.

    at_level: 'debug' (default) or 'info'.
    """
    emit = logger.info if at_level == "info" else logger.debug
    emit(
        "[%s] supervisor_metrics total=%d llm_rate=%.2f fallback_rate=%.2f "
        "guard_rate=%.2f shortcircuit_rate=%.2f "
        "avg_confidence=%s avg_decide_ms=%s max_decide_ms=%s rule_dist=%s",
        session_id,
        stats.total_decisions,
        stats.llm_utilization_rate,
        stats.fallback_rate,
        stats.guard_count / stats.total_decisions if stats.total_decisions else 0.0,
        stats.shortcircuit_rate,
        f"{stats.avg_confidence:.2f}" if stats.avg_confidence is not None else "n/a",
        f"{stats.avg_decide_ms:.1f}" if stats.avg_decide_ms is not None else "n/a",
        f"{stats.max_decide_ms:.1f}" if stats.max_decide_ms is not None else "n/a",
        stats.rule_distribution,
    )
