"""
Tests for supervisor_safety.py.

Covers:
  - consecutive_same_role_count(): zero prior decisions, trailing match,
    reset on different role, proposed not yet appended
  - token_spend_total(): sums only llm_called=True decisions, handles None metadata
  - is_token_budget_exhausted(): disabled at 0, triggers at threshold
  - is_hard_decision_cap_reached(): disabled at 0, triggers at threshold
  - session_rollout_bucket(): consistent hash, empty session_id maps to 0
  - is_in_rollout(): 100=always, 0=never, hash-based mid-range
  - detect_role_loop(): disabled at 0, worker-only, returns reason string
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

import pytest

from orchestration.app.graph.supervisor_safety import (
    consecutive_same_role_count,
    detect_role_loop,
    is_hard_decision_cap_reached,
    is_in_rollout,
    is_token_budget_exhausted,
    session_rollout_bucket,
    token_spend_total,
)
from orchestration.app.graph.supervisor import _VALID_WORKER_ROLES


# ─── helpers ──────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _decision(
    next_node: str = "coder",
    llm_called: bool = False,
    tokens_generated: int | None = None,
) -> Dict[str, Any]:
    meta: Dict[str, Any] = {"llm_called": llm_called}
    if tokens_generated is not None:
        meta["tokens_generated"] = tokens_generated
    return {"next_node": next_node, "rule": "R_test", "metadata": meta, "ts": _now()}


def _decisions(*next_nodes: str) -> List[Dict[str, Any]]:
    return [_decision(n) for n in next_nodes]


# ─── consecutive_same_role_count ──────────────────────────────────────────────

class TestConsecutiveSameRoleCount:

    def test_returns_one_when_no_prior_decisions(self):
        assert consecutive_same_role_count([], "coder") == 1

    def test_returns_one_when_different_last_role(self):
        decisions = _decisions("planner", "reviewer")
        assert consecutive_same_role_count(decisions, "coder") == 1

    def test_counts_one_prior_match(self):
        decisions = _decisions("coder")
        assert consecutive_same_role_count(decisions, "coder") == 2

    def test_counts_three_consecutive_matches(self):
        decisions = _decisions("coder", "coder", "coder")
        assert consecutive_same_role_count(decisions, "coder") == 4

    def test_stops_at_non_matching_entry(self):
        decisions = _decisions("planner", "coder", "coder")
        assert consecutive_same_role_count(decisions, "coder") == 3

    def test_resets_on_different_role_in_middle(self):
        decisions = _decisions("coder", "planner", "coder")
        # trailing: only the last "coder" matches, then "planner" breaks it
        assert consecutive_same_role_count(decisions, "coder") == 2

    def test_proposed_node_not_in_decisions_returns_one(self):
        decisions = _decisions("planner", "reviewer", "debugger")
        assert consecutive_same_role_count(decisions, "coder") == 1


# ─── token_spend_total ────────────────────────────────────────────────────────

class TestTokenSpendTotal:

    def test_zero_when_no_decisions(self):
        assert token_spend_total([]) == 0

    def test_sums_only_llm_called_true(self):
        decisions = [
            _decision("coder", llm_called=True, tokens_generated=100),
            _decision("planner", llm_called=True, tokens_generated=50),
            _decision("reviewer", llm_called=False, tokens_generated=200),  # excluded
        ]
        assert token_spend_total(decisions) == 150

    def test_zero_when_no_llm_calls(self):
        decisions = [_decision("coder", llm_called=False, tokens_generated=500)]
        assert token_spend_total(decisions) == 0

    def test_handles_missing_tokens_generated(self):
        decisions = [_decision("coder", llm_called=True)]  # no tokens_generated key
        assert token_spend_total(decisions) == 0

    def test_handles_none_metadata(self):
        decisions = [{"next_node": "coder", "rule": "R4", "metadata": None, "ts": _now()}]
        assert token_spend_total(decisions) == 0

    def test_accumulates_multiple_llm_calls(self):
        decisions = [
            _decision("coder", llm_called=True, tokens_generated=120),
            _decision("researcher", llm_called=True, tokens_generated=80),
            _decision("coder", llm_called=True, tokens_generated=200),
        ]
        assert token_spend_total(decisions) == 400


# ─── is_token_budget_exhausted ────────────────────────────────────────────────

class TestIsTokenBudgetExhausted:

    def test_not_exhausted_below_threshold(self):
        decisions = [_decision("coder", llm_called=True, tokens_generated=100)]
        assert not is_token_budget_exhausted(decisions, max_tokens=500)

    def test_exhausted_at_threshold(self):
        decisions = [_decision("coder", llm_called=True, tokens_generated=500)]
        assert is_token_budget_exhausted(decisions, max_tokens=500)

    def test_exhausted_above_threshold(self):
        decisions = [_decision("coder", llm_called=True, tokens_generated=600)]
        assert is_token_budget_exhausted(decisions, max_tokens=500)

    def test_disabled_when_max_tokens_is_zero(self):
        decisions = [_decision("coder", llm_called=True, tokens_generated=9999)]
        assert not is_token_budget_exhausted(decisions, max_tokens=0)

    def test_disabled_when_max_tokens_is_negative(self):
        decisions = [_decision("coder", llm_called=True, tokens_generated=9999)]
        assert not is_token_budget_exhausted(decisions, max_tokens=-1)


# ─── is_hard_decision_cap_reached ────────────────────────────────────────────

class TestIsHardDecisionCapReached:

    def test_not_reached_below_cap(self):
        decisions = _decisions("coder", "planner", "reviewer")
        assert not is_hard_decision_cap_reached(decisions, max_decisions=10)

    def test_reached_at_cap(self):
        decisions = _decisions(*["coder"] * 10)
        assert is_hard_decision_cap_reached(decisions, max_decisions=10)

    def test_reached_above_cap(self):
        decisions = _decisions(*["coder"] * 12)
        assert is_hard_decision_cap_reached(decisions, max_decisions=10)

    def test_disabled_when_max_decisions_is_zero(self):
        decisions = _decisions(*["coder"] * 1000)
        assert not is_hard_decision_cap_reached(decisions, max_decisions=0)

    def test_not_reached_empty_decisions(self):
        assert not is_hard_decision_cap_reached([], max_decisions=10)


# ─── session_rollout_bucket ───────────────────────────────────────────────────

class TestSessionRolloutBucket:

    def test_empty_session_id_maps_to_zero(self):
        assert session_rollout_bucket("") == 0

    def test_returns_value_in_range_0_99(self):
        for sid in ["abc", "sess-001", "test-session-xyz", "a" * 64]:
            bucket = session_rollout_bucket(sid)
            assert 0 <= bucket <= 99, f"bucket={bucket} out of range for sid={sid!r}"

    def test_same_session_id_always_same_bucket(self):
        sid = "consistent-session-id"
        b1 = session_rollout_bucket(sid)
        b2 = session_rollout_bucket(sid)
        assert b1 == b2

    def test_different_session_ids_can_differ(self):
        buckets = {session_rollout_bucket(f"sess-{i}") for i in range(50)}
        assert len(buckets) > 1  # hashes should spread across buckets


# ─── is_in_rollout ────────────────────────────────────────────────────────────

class TestIsInRollout:

    def test_100pct_always_in_rollout(self):
        assert is_in_rollout("any-session", 100)
        assert is_in_rollout("", 100)

    def test_0pct_never_in_rollout(self):
        assert not is_in_rollout("any-session", 0)
        assert not is_in_rollout("", 0)

    def test_session_in_rollout_when_bucket_below_pct(self):
        # Find a session whose bucket is < 50
        for i in range(200):
            sid = f"sess-{i}"
            bucket = session_rollout_bucket(sid)
            if bucket < 50:
                assert is_in_rollout(sid, 50)
                return
        pytest.skip("could not find session with bucket < 50 in 200 attempts")

    def test_session_not_in_rollout_when_bucket_above_pct(self):
        # Find a session whose bucket is >= 50
        for i in range(200):
            sid = f"sess-{i}"
            bucket = session_rollout_bucket(sid)
            if bucket >= 50:
                assert not is_in_rollout(sid, 50)
                return
        pytest.skip("could not find session with bucket >= 50 in 200 attempts")

    def test_negative_pct_treated_as_zero(self):
        assert not is_in_rollout("any-session", -10)

    def test_pct_above_100_treated_as_100(self):
        assert is_in_rollout("any-session", 150)


# ─── detect_role_loop ─────────────────────────────────────────────────────────

class TestDetectRoleLoop:

    def test_returns_none_when_disabled(self):
        decisions = _decisions("coder", "coder", "coder", "coder", "coder")
        assert detect_role_loop(decisions, "coder", 0, _VALID_WORKER_ROLES) is None

    def test_returns_none_for_non_worker_role(self):
        decisions = _decisions("tool_executor", "tool_executor", "tool_executor")
        result = detect_role_loop(decisions, "tool_executor", 2, _VALID_WORKER_ROLES)
        assert result is None

    def test_returns_none_for_check_convergence(self):
        decisions = _decisions("check_convergence", "check_convergence")
        result = detect_role_loop(decisions, "check_convergence", 2, _VALID_WORKER_ROLES)
        assert result is None

    def test_returns_none_below_threshold(self):
        decisions = _decisions("coder", "coder")  # 2 prior + 1 proposed = 3, threshold=5
        result = detect_role_loop(decisions, "coder", 5, _VALID_WORKER_ROLES)
        assert result is None

    def test_returns_reason_at_threshold(self):
        decisions = _decisions("coder", "coder", "coder", "coder")  # 4 prior + proposed = 5
        result = detect_role_loop(decisions, "coder", 5, _VALID_WORKER_ROLES)
        assert result is not None
        assert "coder" in result
        assert "5x" in result

    def test_returns_reason_above_threshold(self):
        decisions = _decisions(*["debugger"] * 7)
        result = detect_role_loop(decisions, "debugger", 5, _VALID_WORKER_ROLES)
        assert result is not None
        assert "debugger" in result

    def test_returns_none_when_different_role_breaks_streak(self):
        decisions = _decisions("coder", "coder", "planner", "coder", "coder")
        # trailing streak: coder, coder = 2, + proposed = 3, threshold=5
        result = detect_role_loop(decisions, "coder", 5, _VALID_WORKER_ROLES)
        assert result is None

    def test_threshold_of_one_always_triggers_for_worker(self):
        result = detect_role_loop([], "reviewer", 1, _VALID_WORKER_ROLES)
        assert result is not None
        assert "reviewer" in result
