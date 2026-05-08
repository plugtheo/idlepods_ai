"""
Safety bounds and rollout gating for the supervisor pipeline.

Provides pure functions for:
  - Hard decision cap: detect when total decisions exceed a per-run limit.
  - Role-loop detection: detect when the same agent role is dispatched
    consecutively more times than allowed.
  - Token budget: sum LLM supervisor call tokens to enforce a per-run cap.
  - Rollout gating: hash-based session bucketing for gradual strategy rollout.

None of these functions perform I/O or import from other project modules —
this avoids circular imports with supervisor.py, supervisor_llm.py, etc.
Callers are responsible for logging and for constructing SupervisorDecision
overrides when a violation is detected.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Role-loop detection ───────────────────────────────────────────────────────


def consecutive_same_role_count(
    decisions: List[Dict[str, Any]],
    proposed_next_node: str,
) -> int:
    """
    Count how many trailing decisions (including the proposed next) would
    consecutively target the same node.

    The proposed_next_node is treated as the decision about to be appended —
    it is not yet in the decisions list.  Returns 1 when no prior decisions
    share the same node.
    """
    count = 1  # count the proposed decision itself
    for d in reversed(decisions):
        if d.get("next_node") == proposed_next_node:
            count += 1
        else:
            break
    return count


def detect_role_loop(
    decisions: List[Dict[str, Any]],
    proposed_next_node: str,
    max_consecutive: int,
    valid_worker_roles: frozenset,
) -> Optional[str]:
    """
    Return a violation reason string if a role-dispatch loop is detected, else None.

    A loop is detected when the same worker role would be dispatched
    consecutively >= max_consecutive times.  Non-worker nodes (tool_executor,
    check_convergence, etc.) are exempt.  Set max_consecutive <= 0 to disable.
    """
    if max_consecutive <= 0:
        return None
    if proposed_next_node not in valid_worker_roles:
        return None

    count = consecutive_same_role_count(decisions, proposed_next_node)
    if count >= max_consecutive:
        return f"role_loop_{proposed_next_node}_{count}x"
    return None


# ── Token budget ──────────────────────────────────────────────────────────────


def token_spend_total(decisions: List[Dict[str, Any]]) -> int:
    """
    Sum tokens_generated across all decisions where llm_called=True.

    Decisions without a tokens_generated key contribute 0.
    """
    return sum(
        int((d.get("metadata") or {}).get("tokens_generated", 0))
        for d in decisions
        if (d.get("metadata") or {}).get("llm_called") is True
    )


def is_token_budget_exhausted(
    decisions: List[Dict[str, Any]],
    max_tokens: int,
) -> bool:
    """
    Return True when cumulative supervisor LLM tokens have reached max_tokens.
    max_tokens <= 0 disables the check (always returns False).
    """
    if max_tokens <= 0:
        return False
    return token_spend_total(decisions) >= max_tokens


# ── Hard decision cap ─────────────────────────────────────────────────────────


def is_hard_decision_cap_reached(
    decisions: List[Dict[str, Any]],
    max_decisions: int,
) -> bool:
    """
    Return True when the total number of decisions has reached max_decisions.
    max_decisions <= 0 disables the check (always returns False).
    """
    if max_decisions <= 0:
        return False
    return len(decisions) >= max_decisions


# ── Rollout gating ────────────────────────────────────────────────────────────


def session_rollout_bucket(session_id: str) -> int:
    """
    Map session_id to a deterministic integer bucket in [0, 99].

    Uses the first 8 hex digits of the MD5 hash for stability.
    Empty session_id always maps to bucket 0.
    """
    if not session_id:
        return 0
    return int(hashlib.md5(session_id.encode(), usedforsecurity=False).hexdigest(), 16) % 100


def is_in_rollout(session_id: str, rollout_pct: int) -> bool:
    """
    Return True when this session is within the rollout percentage window.

    rollout_pct=100 (default) → all sessions included.
    rollout_pct=0 → no sessions included.
    """
    if rollout_pct >= 100:
        return True
    if rollout_pct <= 0:
        return False
    return session_rollout_bucket(session_id) < rollout_pct
