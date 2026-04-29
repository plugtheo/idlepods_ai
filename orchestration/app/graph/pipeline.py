"""
LangGraph pipeline
===================
Builds and compiles the multi-agent StateGraph.

Graph topology
--------------

START
  └─[route_entry]─► planner | researcher | coder | debugger | reviewer | critic
                         │
                   [next_in_chain] ───► (next agent in chain)
                         │
                         └───► check_convergence ─[check_convergence]─► consensus
                                      │
                                      └─► (loop back to chain[0])
consensus ──► END

There is one node per agent role.  Nodes not in the active agent_chain
for a given request are simply never routed to — the graph is defined
statically but only the nodes that the chain selects are executed.

A dedicated `update_loop_state` node (runs after check_convergence when
looping) increments `current_iteration`, updates `best_score`, and resets
`agent_chain_index` before routing back to the first agent.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .edges import (
    _AGENT_NODES,
    check_convergence as _check_convergence_edge,
    next_in_chain,
    route_entry,
)
from .nodes import (
    coder_node,
    consensus_node,
    critic_node,
    debugger_node,
    planner_node,
    researcher_node,
    review_critic_node,
    reviewer_node,
)
from .state import AgentState
from ..utils.scoring import score_iteration

logger = logging.getLogger(__name__)

SHORT_CHAIN_MAX_AGENTS = 2   # chains at or below this length skip the consensus node
MIN_PER_ITERATION_NODES = 5  # minimum node count assumed per iteration for recursion limit
PIPELINE_HELPER_NODES = 4    # non-agent nodes per iteration (check_conv, finalize, update_loop, …)
RECURSION_LIMIT_BUFFER = 15  # extra node budget above the worst-case estimate


# ── State-updating node for "continue looping" branch ──────────────────────


async def _update_loop_state(state: AgentState) -> dict:
    """
    Called when the pipeline decides to run another iteration.

    Responsibilities:
    - Update best_score / best_output from the iteration just completed.
    - Increment current_iteration.
    - Reset agent_chain_index to 0.

    Note: this node does NOT call any agent or inference service.
    """
    current_iteration = state.get("current_iteration", 1)
    history = state.get("iteration_history", [])
    best_score = state.get("best_score", 0.0)
    best_output = state.get("best_output", "")

    iter_score = score_iteration(history, current_iteration)
    last_output = state.get("last_output", "")

    if iter_score > best_score:
        best_score = iter_score
        best_output = last_output

    return {
        "current_iteration": current_iteration + 1,
        "agent_chain_index": 0,
        "best_score": best_score,
        "best_output": best_output,
        "iteration_scores": list(state.get("iteration_scores", [])) + [iter_score],
    }


async def _finalize_state(state: AgentState) -> dict:
    """
    Sets converged/quality_converged flags before routing to consensus.
    Also updates best_score from the final completed iteration.

    Skips consensus for short chains (≤ 2 agents) that have already
    quality-converged — those chains produce a clean, focused output and
    running a synthesis agent on top just paraphrases it.
    """
    current_iteration = state.get("current_iteration", 1)
    history = state.get("iteration_history", [])
    # 0.85 fallback is dead in normal flow: routes/run.py always seeds state from settings.convergence_threshold
    threshold = state.get("convergence_threshold", 0.85)
    best_score = state.get("best_score", 0.0)
    best_output = state.get("best_output", "")
    last_output = state.get("last_output", "")
    chain = state.get("agent_chain", [])

    iter_score = score_iteration(history, current_iteration)
    if iter_score > best_score:
        best_score = iter_score
        best_output = last_output

    quality_converged = best_score >= threshold

    # Skip the consensus inference call when:
    #   1. The chain is short (≤ 2 agents) — nothing complex to synthesise; AND
    #   2. Quality threshold was actually met — the output is already polished.
    # For longer chains (planner + coder + review_critic etc.) the consensus
    # node provides meaningful integration value and should always run.
    skip_consensus = quality_converged and len(chain) <= SHORT_CHAIN_MAX_AGENTS

    delta: dict = {
        "best_score": best_score,
        "best_output": best_output,
        "iteration_scores": list(state.get("iteration_scores", [])) + [iter_score],
        "converged": True,
        "quality_converged": quality_converged,
        "skip_consensus": skip_consensus,
    }

    if skip_consensus:
        # Populate final_output now so run.py can read it without consensus
        delta["final_output"] = best_output

    return delta


# ── Edge wrapper that distinguishes "converge" vs "loop" ─────────────────


def _noop_convergence_anchor(state: AgentState) -> dict:
    """No-op node.  Exists solely as a named routing anchor; all logic is in the edge."""
    return {}


def _check_convergence_with_update(state: AgentState) -> str:
    """
    Wraps the convergence edge logic.

    Returns:
      "finalize"       — about to converge; update flags then go to consensus
      "update_loop"    — keep iterating; update state then go to next agent
    """
    decision = _check_convergence_edge(state)
    if decision == "consensus":
        return "finalize"
    return "update_loop"


def _next_in_chain_or_convergence(state: AgentState) -> str:
    """
    After an agent node, advance in the chain or go to check_convergence.
    """
    return next_in_chain(state)


# ── Graph construction ────────────────────────────────────────────────────


def build_pipeline() -> CompiledStateGraph:
    """
    Build and compile the multi-agent LangGraph pipeline.

    Returns a compiled graph ready for `await graph.ainvoke(state)`.
    """
    graph = StateGraph(AgentState)

    # Register agent nodes
    graph.add_node("planner",       planner_node)
    graph.add_node("researcher",    researcher_node)
    graph.add_node("coder",         coder_node)
    graph.add_node("debugger",      debugger_node)
    graph.add_node("reviewer",      reviewer_node)
    graph.add_node("critic",        critic_node)
    graph.add_node("review_critic", review_critic_node)
    graph.add_node("consensus",     consensus_node)

    # Helper nodes (no inference call)
    graph.add_node("check_convergence", _noop_convergence_anchor)  # no-op; routing done via edge
    graph.add_node("finalize",     _finalize_state)
    graph.add_node("update_loop",  _update_loop_state)

    # Entry: START → first agent (determined by agent_chain[0])
    graph.add_conditional_edges(
        START,
        route_entry,
        {role: role for role in _AGENT_NODES},
    )

    # After each agent node: advance chain or go to convergence check
    _all_agent_roles = list(_AGENT_NODES.keys())
    _non_consensus_roles = [r for r in _all_agent_roles if r != "consensus"]

    for role in _non_consensus_roles:
        targets = {r: r for r in _non_consensus_roles}
        targets["check_convergence"] = "check_convergence"
        graph.add_conditional_edges(role, _next_in_chain_or_convergence, targets)

    # check_convergence: either finalize (→ consensus) or loop
    graph.add_conditional_edges(
        "check_convergence",
        _check_convergence_with_update,
        {"finalize": "finalize", "update_loop": "update_loop"},
    )

    # finalize → consensus (always) or END (when skip_consensus=True)
    graph.add_conditional_edges(
        "finalize",
        lambda s: END if s.get("skip_consensus", False) else "consensus",
        {END: END, "consensus": "consensus"},
    )

    # update_loop → back to first agent in chain (via conditional entry)
    graph.add_conditional_edges(
        "update_loop",
        route_entry,
        {role: role for role in _AGENT_NODES},
    )

    # consensus → END
    graph.add_edge("consensus", END)

    return graph.compile()


def _recursion_limit(max_iterations: int, chain_length: int) -> int:
    """
    Compute a safe LangGraph recursion limit.

    LangGraph counts every node invocation.  With N agents and M iterations:
      worst case = M × (N agents + 1 convergence + 2 helpers) + buffer
    """
    per_iteration = max(chain_length, MIN_PER_ITERATION_NODES) + PIPELINE_HELPER_NODES
    return (max_iterations + 1) * per_iteration + RECURSION_LIMIT_BUFFER
