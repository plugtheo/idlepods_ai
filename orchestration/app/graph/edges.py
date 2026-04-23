"""
LangGraph conditional edges
============================
Edge functions determine which node to visit next after each node runs.
They read AgentState and return the name of the next node as a string.

Flow overview
-------------

START
  └─► route_entry            --- decides chain[0] or "consensus"
        │
        ▼ (one of the agent nodes)
  planner / researcher / coder / debugger / reviewer / critic
        │
        ▼
  _next_in_chain             --- advance through agent_chain or → check_convergence
        │
        ├─► (next agent node)
        └─► check_convergence
                │
                ├─► "consensus"  (converged or max iterations)
                └─► "planner"    (loop again from chain[0])
                    (possibly with "debugger" injected when score is low)
consensus
  └─► END
"""

from __future__ import annotations

import logging

from .state import AgentState
from ..utils.scoring import score_iteration

logger = logging.getLogger(__name__)

# All node names registered in the graph (except END, START, check_convergence)
_AGENT_NODES = {
    "planner":      "planner",
    "researcher":   "researcher",
    "coder":        "coder",
    "debugger":     "debugger",
    "reviewer":     "reviewer",
    "critic":       "critic",
    "review_critic": "review_critic",
    "consensus":    "consensus",
}


def route_entry(state: AgentState) -> str:
    """
    Decide where to start the agent chain.

    If agent_chain is empty or index is past the end, go straight to
    consensus (degenerate case — should not happen in normal operation).
    """
    chain = state.get("agent_chain", [])
    if not chain:
        logger.warning("agent_chain is empty — routing to consensus")
        return "consensus"
    first_role = chain[0]
    return _AGENT_NODES.get(first_role, "consensus")


def next_in_chain(state: AgentState) -> str:
    """
    After an agent node runs, decide which node is next.

    - If there are more agents in the chain → return the next agent.
    - Otherwise → return "check_convergence".
    """
    chain = state.get("agent_chain", [])
    index = state.get("agent_chain_index", 0)

    if index < len(chain):
        role = chain[index]
        return _AGENT_NODES.get(role, "check_convergence")

    return "check_convergence"


def check_convergence(state: AgentState) -> str:
    """
    Score the just-completed iteration and decide whether to continue.

    Returns
    -------
    "consensus"  — pipeline converged (score >= threshold) or max_iterations
    first_role   — keep iterating (reset chain index and go again)
    """
    current_iteration = state.get("current_iteration", 1)
    max_iterations = state.get("max_iterations", 5)
    threshold = state.get("convergence_threshold") or 0.85
    history = state.get("iteration_history", [])
    best_score = state.get("best_score", 0.0)
    session_id = state.get("session_id", "")[:8]

    iter_score = score_iteration(history, current_iteration)

    logger.info(
        "[%s] check_convergence: iter=%d  score=%.3f  best=%.3f  threshold=%.2f",
        session_id, current_iteration, iter_score, best_score, threshold,
    )

    # Converged: quality threshold met
    if iter_score >= threshold:
        logger.info("[%s] Converged at iter=%d  score=%.3f", session_id, current_iteration, iter_score)
        return "consensus"

    # Forced stop: reached max iterations
    if current_iteration >= max_iterations:
        logger.info("[%s] Max iterations reached (%d)", session_id, max_iterations)
        return "consensus"

    # Continue: more iterations needed
    chain = state.get("agent_chain", [])
    first_role = chain[0] if chain else "planner"
    return _AGENT_NODES.get(first_role, "planner")
