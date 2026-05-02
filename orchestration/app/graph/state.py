"""
AgentState — LangGraph state schema
=====================================
Every node in the pipeline reads from and writes to this TypedDict.
LangGraph merges partial dicts returned by each node into the shared state.

Design rules
------------
- All fields are plain Python types (str, int, float, list, dict) so the
  state is JSON-serialisable for observability and future message-bus use.
- No model instances, no framework objects.  The state carries DATA only.
- `iteration_history` accumulates every agent output across all iterations;
  it's the only "memory" the pipeline has within a request.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

# Plan is imported lazily where used to avoid circular imports at module load.


class AgentState(TypedDict, total=False):
    # ── Request identity ──────────────────────────────────────────────────
    session_id: str
    """Unique identifier for this pipeline run (set by Orchestration at start)."""

    task_id: str
    """Stable multi-turn identifier; falls back to session_id when not supplied by the caller."""

    user_prompt: str
    """Original raw prompt from the user."""

    # ── Routing ───────────────────────────────────────────────────────────
    agent_chain: List[str]
    """Ordered list of agent role names to run this iteration, e.g. ['coder', 'reviewer']."""

    agent_chain_index: int
    """Current position within agent_chain (0-based). Incremented by each agent node."""

    # ── File gate & diff state ────────────────────────────────────────────
    allowed_files: List[str]
    """Repo-relative POSIX paths the context builder is permitted to scan."""

    file_fingerprints: Dict[str, Any]
    """Per-file fingerprints from the last context build (path → first-line string)."""

    # ── Context enrichment (set once before the loop starts) ─────────────
    few_shots: List[Dict[str, Any]]
    """Past similar experiences retrieved by the Context Service (RAG)."""

    repo_snippets: List[Dict[str, Any]]
    """Relevant code snippets from the local repo."""

    system_hints: str
    """Guidance text injected into every agent system prompt this request."""

    # ── Iteration tracking ────────────────────────────────────────────────
    current_iteration: int
    """Which iteration the loop is on (1-based)."""

    max_iterations: int
    """Maximum number of iterations before forced stop."""

    convergence_threshold: float
    """Quality score threshold to consider the output converged."""

    # ── Agent outputs ─────────────────────────────────────────────────────
    conversation_history: List[Dict[str, Any]]
    """Prior multi-turn exchanges loaded from Redis at request start."""

    iteration_history: List[Dict[str, Any]]
    """
    Every agent step across all iterations, in order.
    Each entry: {role, iteration, output, score, timestamp}
    """

    last_output: str
    """Most recent output from the last agent that ran."""

    # ── Scoring / convergence ─────────────────────────────────────────────
    iteration_scores: List[float]
    """Score produced after each complete iteration."""

    best_score: float
    """Highest score achieved so far."""

    best_output: str
    """Output that produced the best_score."""

    converged: bool
    """True once the quality threshold is reached or max_iterations is exceeded."""

    quality_converged: bool
    """True ONLY when score >= convergence_threshold (not just max-iter exit)."""

    # ── Final result (set by last node before END) ────────────────────────
    final_output: str
    """The definitive output to return to the user."""

    final_score: float
    """Quality score of the final output."""

    skip_consensus: bool
    """
    When True the pipeline routes from finalize → END, bypassing the
    consensus agent.  Set by _finalize_state for short chains (≤ 2 agents)
    that have already quality-converged — consensus would only paraphrase
    an already-clean output, wasting an inference call.
    """

    # ── Tool use (ReAct loop) ─────────────────────────────────────────────────
    pending_tool_calls: List[Dict[str, Any]]
    """Tool calls emitted by the last coder output; cleared after tool_executor runs."""

    tool_steps_used: int
    """Number of tool execution rounds completed in the current pipeline run."""

    tool_originating_role: str
    """Role that emitted the pending tool call; used to route back after tool_executor."""

    # ── Plan tracking (Phase 2) ───────────────────────────────────────────────
    plan: Optional[Dict[str, Any]]
    """Serialised Plan dict (model_dump) for the current task; None when no plan loaded."""

    plan_changed: bool
    """Set True by any node that mutates plan; finalize writes back on convergence."""

    current_step_id: Optional[str]
    """The step id the planner selected for this turn; injected into tool-role system msgs."""
