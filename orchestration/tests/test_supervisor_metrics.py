"""
Tests for supervisor_metrics.py.

Covers:
  - compute_routing_stats(): empty decisions, llm_called counting, fallback/guard
    suffix detection, shortcircuit counting, rule distribution, confidence stats,
    elapsed-time stats, derived rate properties
  - log_routing_stats(): debug and info levels, no crash on zero stats
  - _has_any_suffix(): suffix classification correctness
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from orchestration.app.graph.supervisor_metrics import (
    RoutingStats,
    _FALLBACK_SUFFIXES,
    _GUARD_SUFFIXES,
    _has_any_suffix,
    compute_routing_stats,
    log_routing_stats,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _decision(
    rule: str = "R_llm",
    llm_called: bool = True,
    shortcircuit: bool = False,
    confidence: float | None = None,
    decide_ms: float | None = None,
    next_node: str = "coder",
) -> Dict[str, Any]:
    meta: Dict[str, Any] = {"llm_called": llm_called}
    if shortcircuit:
        meta["shortcircuit"] = True
    if confidence is not None:
        meta["confidence"] = confidence
    if decide_ms is not None:
        meta["decide_ms"] = decide_ms
    return {
        "iteration": 1,
        "next_node": next_node,
        "reason": "test",
        "rule": rule,
        "metadata": meta,
        "ts": _now(),
    }


# ─── _has_any_suffix ─────────────────────────────────────────────────────────

class TestHasAnySuffix:

    def test_matches_budget_fallback(self):
        assert _has_any_suffix("R3_budget_fallback", _FALLBACK_SUFFIXES)

    def test_matches_exc_fallback(self):
        assert _has_any_suffix("R4_exc_fallback", _FALLBACK_SUFFIXES)

    def test_matches_llm_fallback(self):
        assert _has_any_suffix("R5_llm_fallback", _FALLBACK_SUFFIXES)

    def test_matches_hybrid_guard(self):
        assert _has_any_suffix("R3_hybrid_guard", _GUARD_SUFFIXES)

    def test_no_match_plain_rule(self):
        assert not _has_any_suffix("R_llm", _FALLBACK_SUFFIXES)
        assert not _has_any_suffix("R_forced", _FALLBACK_SUFFIXES)
        assert not _has_any_suffix("R3", _FALLBACK_SUFFIXES)

    def test_hybrid_guard_not_in_fallback_suffixes(self):
        assert not _has_any_suffix("R3_hybrid_guard", _FALLBACK_SUFFIXES)

    def test_fallback_suffixes_not_in_guard_suffixes(self):
        assert not _has_any_suffix("R3_budget_fallback", _GUARD_SUFFIXES)


# ─── compute_routing_stats ────────────────────────────────────────────────────

class TestComputeRoutingStats:

    def test_empty_list_returns_zero_stats(self):
        stats = compute_routing_stats([])
        assert stats.total_decisions == 0
        assert stats.llm_called_count == 0
        assert stats.fallback_count == 0
        assert stats.guard_count == 0
        assert stats.shortcircuit_count == 0
        assert stats.rule_distribution == {}
        assert stats.avg_confidence is None
        assert stats.avg_decide_ms is None
        assert stats.max_decide_ms is None

    def test_total_decisions_count(self):
        decisions = [_decision() for _ in range(5)]
        stats = compute_routing_stats(decisions)
        assert stats.total_decisions == 5

    def test_llm_called_count(self):
        decisions = [
            _decision(llm_called=True),
            _decision(llm_called=True),
            _decision(llm_called=False),
        ]
        stats = compute_routing_stats(decisions)
        assert stats.llm_called_count == 2

    def test_llm_utilization_rate(self):
        decisions = [_decision(llm_called=True)] * 3 + [_decision(llm_called=False)] * 7
        stats = compute_routing_stats(decisions)
        assert stats.llm_utilization_rate == pytest.approx(0.3)

    def test_llm_utilization_rate_zero_when_empty(self):
        assert compute_routing_stats([]).llm_utilization_rate == 0.0

    def test_fallback_count_budget_fallback(self):
        decisions = [
            _decision(rule="R4_budget_fallback", llm_called=False),
            _decision(rule="R_llm", llm_called=True),
        ]
        stats = compute_routing_stats(decisions)
        assert stats.fallback_count == 1

    def test_fallback_count_exc_fallback(self):
        decisions = [
            _decision(rule="R5_exc_fallback", llm_called=False),
            _decision(rule="R5_llm_fallback", llm_called=False),
        ]
        stats = compute_routing_stats(decisions)
        assert stats.fallback_count == 2

    def test_fallback_rate(self):
        decisions = [
            _decision(rule="R4_budget_fallback", llm_called=False),
            _decision(rule="R_llm", llm_called=True),
            _decision(rule="R_llm", llm_called=True),
            _decision(rule="R4_exc_fallback", llm_called=False),
        ]
        stats = compute_routing_stats(decisions)
        assert stats.fallback_rate == pytest.approx(0.5)

    def test_guard_count_hybrid_guard(self):
        decisions = [
            _decision(rule="R3_hybrid_guard", llm_called=False),
            _decision(rule="R3_hybrid_guard", llm_called=False),
            _decision(rule="R_llm", llm_called=True),
        ]
        stats = compute_routing_stats(decisions)
        assert stats.guard_count == 2

    def test_guard_not_counted_as_fallback(self):
        decisions = [_decision(rule="R3_hybrid_guard", llm_called=False)]
        stats = compute_routing_stats(decisions)
        assert stats.fallback_count == 0
        assert stats.guard_count == 1

    def test_shortcircuit_count(self):
        decisions = [
            _decision(rule="R_forced", llm_called=False, shortcircuit=True),
            _decision(rule="R_forced", llm_called=False, shortcircuit=True),
            _decision(rule="R_llm", llm_called=True),
        ]
        stats = compute_routing_stats(decisions)
        assert stats.shortcircuit_count == 2

    def test_shortcircuit_rate(self):
        decisions = [
            _decision(rule="R_forced", llm_called=False, shortcircuit=True),
            _decision(rule="R_llm", llm_called=True),
            _decision(rule="R_llm", llm_called=True),
            _decision(rule="R_llm", llm_called=True),
        ]
        stats = compute_routing_stats(decisions)
        assert stats.shortcircuit_rate == pytest.approx(0.25)

    def test_rule_distribution_counts_each_rule(self):
        decisions = [
            _decision(rule="R_llm"),
            _decision(rule="R_llm"),
            _decision(rule="R_forced"),
            _decision(rule="R3_hybrid_guard"),
        ]
        stats = compute_routing_stats(decisions)
        assert stats.rule_distribution["R_llm"] == 2
        assert stats.rule_distribution["R_forced"] == 1
        assert stats.rule_distribution["R3_hybrid_guard"] == 1

    def test_avg_confidence_from_llm_decisions(self):
        decisions = [
            _decision(confidence=0.9),
            _decision(confidence=0.7),
            _decision(llm_called=False),  # no confidence key
        ]
        stats = compute_routing_stats(decisions)
        assert stats.avg_confidence == pytest.approx(0.8)

    def test_avg_confidence_none_when_no_llm_calls(self):
        decisions = [_decision(llm_called=False)]
        stats = compute_routing_stats(decisions)
        assert stats.avg_confidence is None

    def test_avg_decide_ms(self):
        decisions = [
            _decision(decide_ms=50.0),
            _decision(decide_ms=150.0),
        ]
        stats = compute_routing_stats(decisions)
        assert stats.avg_decide_ms == pytest.approx(100.0)

    def test_max_decide_ms(self):
        decisions = [
            _decision(decide_ms=30.0),
            _decision(decide_ms=200.0),
            _decision(decide_ms=80.0),
        ]
        stats = compute_routing_stats(decisions)
        assert stats.max_decide_ms == pytest.approx(200.0)

    def test_decide_ms_none_when_absent(self):
        decisions = [_decision()]  # no decide_ms in metadata
        stats = compute_routing_stats(decisions)
        assert stats.avg_decide_ms is None
        assert stats.max_decide_ms is None

    def test_mixed_decisions_comprehensive(self):
        decisions = [
            _decision(rule="R_forced", llm_called=False, shortcircuit=True, decide_ms=1.0),
            _decision(rule="R_forced", llm_called=False, shortcircuit=True, decide_ms=1.2),
            _decision(rule="R_llm", llm_called=True, confidence=0.85, decide_ms=340.0),
            _decision(rule="R3_hybrid_guard", llm_called=False, decide_ms=2.0),
            _decision(rule="R4_budget_fallback", llm_called=False, decide_ms=3.0),
        ]
        stats = compute_routing_stats(decisions)
        assert stats.total_decisions == 5
        assert stats.llm_called_count == 1
        assert stats.shortcircuit_count == 2
        assert stats.guard_count == 1
        assert stats.fallback_count == 1
        assert stats.avg_confidence == pytest.approx(0.85)
        assert stats.max_decide_ms == pytest.approx(340.0)


# ─── log_routing_stats ────────────────────────────────────────────────────────

class TestLogRoutingStats:

    def test_log_debug_does_not_raise(self, caplog):
        stats = RoutingStats(
            total_decisions=5, llm_called_count=3, fallback_count=1,
            guard_count=0, shortcircuit_count=1,
            rule_distribution={"R_llm": 3, "R_forced": 2},
            avg_confidence=0.80, avg_decide_ms=120.0, max_decide_ms=250.0,
        )
        import logging
        with caplog.at_level(logging.DEBUG, logger="orchestration.app.graph.supervisor_metrics"):
            log_routing_stats(stats, "abc12345", at_level="debug")
        assert any("supervisor_metrics" in r.message for r in caplog.records)

    def test_log_info_does_not_raise(self, caplog):
        stats = RoutingStats(total_decisions=0)
        import logging
        with caplog.at_level(logging.INFO, logger="orchestration.app.graph.supervisor_metrics"):
            log_routing_stats(stats, "xyz99999", at_level="info")

    def test_log_contains_session_id(self, caplog):
        stats = RoutingStats(total_decisions=3, llm_called_count=2)
        import logging
        with caplog.at_level(logging.DEBUG, logger="orchestration.app.graph.supervisor_metrics"):
            log_routing_stats(stats, "sess-id1", at_level="debug")
        assert any("sess-id1" in r.message for r in caplog.records)

    def test_log_contains_total_decisions(self, caplog):
        stats = RoutingStats(total_decisions=42)
        import logging
        with caplog.at_level(logging.DEBUG, logger="orchestration.app.graph.supervisor_metrics"):
            log_routing_stats(stats, "sess", at_level="debug")
        assert any("total=42" in r.message for r in caplog.records)

    def test_log_na_when_confidence_absent(self, caplog):
        stats = RoutingStats(total_decisions=1, avg_confidence=None)
        import logging
        with caplog.at_level(logging.DEBUG, logger="orchestration.app.graph.supervisor_metrics"):
            log_routing_stats(stats, "sess", at_level="debug")
        assert any("avg_confidence=n/a" in r.message for r in caplog.records)

    def test_log_na_when_decide_ms_absent(self, caplog):
        stats = RoutingStats(total_decisions=1, avg_decide_ms=None, max_decide_ms=None)
        import logging
        with caplog.at_level(logging.DEBUG, logger="orchestration.app.graph.supervisor_metrics"):
            log_routing_stats(stats, "sess", at_level="debug")
        assert any("avg_decide_ms=n/a" in r.message for r in caplog.records)
