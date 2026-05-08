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
    route_after_tool_user,
    route_entry,
)
from .nodes import (
    _tool_using_roles,
    coder_node,
    consensus_node,
    critic_node,
    debugger_node,
    planner_node,
    researcher_node,
    review_critic_node,
    reviewer_node,
    tool_executor_node,
)
from .supervisor import supervisor_anchor, supervisor_decide
from .state import AgentState
from ..utils.scoring import score_iteration

logger = logging.getLogger(__name__)

SHORT_CHAIN_MAX_AGENTS = 2   # chains at or below this length skip the consensus node
MIN_PER_ITERATION_NODES = 5  # minimum node count assumed per iteration for recursion limit
PIPELINE_HELPER_NODES = 4    # non-agent nodes per iteration (check_conv, finalize, update_loop, …)
RECURSION_LIMIT_BUFFER = 50  # extra node budget above the worst-case estimate (includes tool ReAct steps)


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
    from ..config.settings import settings as _settings
    threshold = state.get("convergence_threshold", _settings.convergence_threshold)
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

    When settings.pipeline_use_supervisor is True, agent dispatch is driven by
    the supervisor node (DeterministicSupervisor) rather than the static
    agent_chain index.  The legacy path remains the default.
    """
    from ..config.settings import settings as _settings
    if _settings.pipeline_use_supervisor:
        return _build_supervisor_pipeline()
    return _build_legacy_pipeline()


def _build_legacy_pipeline() -> CompiledStateGraph:
    """Original chain-index-driven pipeline (default)."""
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
    graph.add_node("tool_executor", tool_executor_node)

    # Helper nodes (no inference call)
    graph.add_node("check_convergence", _noop_convergence_anchor)
    graph.add_node("finalize",     _finalize_state)
    graph.add_node("update_loop",  _update_loop_state)

    # Entry: START → first agent (determined by agent_chain[0])
    graph.add_conditional_edges(
        START,
        route_entry,
        {role: role for role in _AGENT_NODES},
    )

    _all_agent_roles = list(_AGENT_NODES.keys())
    _non_consensus_roles = [r for r in _all_agent_roles if r != "consensus"]
    _tool_using = _tool_using_roles() & set(_non_consensus_roles)
    targets = {r: r for r in _non_consensus_roles}
    targets["check_convergence"] = "check_convergence"
    targets["tool_executor"] = "tool_executor"

    for role in _non_consensus_roles:
        edge_fn = route_after_tool_user if role in _tool_using else _next_in_chain_or_convergence
        graph.add_conditional_edges(role, edge_fn, targets)

    graph.add_conditional_edges(
        "tool_executor",
        lambda s: s.get("tool_originating_role") or next(iter(_tool_using_roles() & set(_non_consensus_roles)), _non_consensus_roles[0]),
        {role: role for role in _AGENT_NODES},
    )

    graph.add_conditional_edges(
        "check_convergence",
        _check_convergence_with_update,
        {"finalize": "finalize", "update_loop": "update_loop"},
    )

    graph.add_conditional_edges(
        "finalize",
        lambda s: END if s.get("skip_consensus", False) else "consensus",
        {END: END, "consensus": "consensus"},
    )

    graph.add_conditional_edges(
        "update_loop",
        route_entry,
        {role: role for role in _AGENT_NODES},
    )

    graph.add_edge("consensus", END)
    return graph.compile()


def _build_supervisor_pipeline() -> CompiledStateGraph:
    """
    Supervisor-driven pipeline.

    Topology: START → supervisor → (any worker | tool_executor | check_convergence)
              every worker → supervisor (ring)
              tool_executor → supervisor
              check_convergence → finalize | update_loop
              update_loop → supervisor
              finalize → consensus | END
              consensus → END

    "consensus" is deliberately excluded from the supervisor's routing targets —
    it is reached only via finalize, never dispatched directly.
    """
    graph = StateGraph(AgentState)

    # Agent nodes
    graph.add_node("planner",       planner_node)
    graph.add_node("researcher",    researcher_node)
    graph.add_node("coder",         coder_node)
    graph.add_node("debugger",      debugger_node)
    graph.add_node("reviewer",      reviewer_node)
    graph.add_node("critic",        critic_node)
    graph.add_node("review_critic", review_critic_node)
    graph.add_node("consensus",     consensus_node)
    graph.add_node("tool_executor", tool_executor_node)

    # Supervisor + helpers
    graph.add_node("supervisor",        supervisor_anchor)
    graph.add_node("check_convergence", _noop_convergence_anchor)
    graph.add_node("finalize",          _finalize_state)
    graph.add_node("update_loop",       _update_loop_state)

    # Roles the supervisor may route to directly (consensus excluded).
    _supervised_roles = [r for r in _AGENT_NODES if r != "consensus"]

    supervisor_targets: dict[str, str] = {r: r for r in _supervised_roles}
    supervisor_targets["tool_executor"]     = "tool_executor"
    supervisor_targets["check_convergence"] = "check_convergence"

    # START → supervisor
    graph.add_edge(START, "supervisor")

    # supervisor → next agent | tool_executor | check_convergence
    graph.add_conditional_edges("supervisor", supervisor_decide, supervisor_targets)

    # Every supervised worker → supervisor (the ring)
    for role in _supervised_roles:
        graph.add_edge(role, "supervisor")

    # tool_executor → supervisor
    graph.add_edge("tool_executor", "supervisor")

    # check_convergence → finalize | update_loop
    graph.add_conditional_edges(
        "check_convergence",
        _check_convergence_with_update,
        {"finalize": "finalize", "update_loop": "update_loop"},
    )

    # finalize → consensus | END
    graph.add_conditional_edges(
        "finalize",
        lambda s: END if s.get("skip_consensus", False) else "consensus",
        {END: END, "consensus": "consensus"},
    )

    # update_loop → supervisor (not route_entry — supervisor handles re-dispatch)
    graph.add_edge("update_loop", "supervisor")

    # consensus → END
    graph.add_edge("consensus", END)

    return graph.compile()


def _recursion_limit(max_iterations: int, chain_length: int) -> int:
    """
    Compute a safe LangGraph recursion limit.

    Legacy mode: worst case = M × (N agents + helpers) + buffer
    Supervisor mode: each step adds a supervisor node invocation, plus tool
    round-trips.  Conservative estimate based on max plan steps per iteration.
    """
    from ..config.settings import settings as _settings

    if _settings.pipeline_use_supervisor:
        max_steps = _settings.pipeline_supervisor_max_steps
        # per iter: supervisor + plan_steps × (supervisor + worker + possible tool pair)
        # + check_convergence + finalize/update_loop + buffer per iter
        per_iter = (max_steps * 4) + 12
        limit = (max_iterations + 2) * per_iter + RECURSION_LIMIT_BUFFER
        logger.info(
            "recursion_limit mode=supervisor max_iter=%d max_steps=%d per_iter=%d limit=%d",
            max_iterations, max_steps, per_iter, limit,
        )
        return limit

    per_iteration = max(chain_length, MIN_PER_ITERATION_NODES) + PIPELINE_HELPER_NODES
    limit = (max_iterations + 1) * per_iteration + RECURSION_LIMIT_BUFFER
    logger.info(
        "recursion_limit mode=legacy max_iter=%d chain=%d per_iter=%d limit=%d",
        max_iterations, chain_length, per_iteration, limit,
    )
    return limit
