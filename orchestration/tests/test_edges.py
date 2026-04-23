"""
Tests for LangGraph edge functions.

Covers:
- route_entry: first role in chain, empty chain → consensus
- next_in_chain: next role, end of chain → check_convergence  
- check_convergence: score above threshold → consensus, max iterations → consensus, continue
"""
import pytest


def _base_state(**overrides):
    state = {
        "session_id": "test-session",
        "user_prompt": "write code",
        "agent_chain": ["planner", "coder", "reviewer"],
        "agent_chain_index": 0,
        "few_shots": [],
        "repo_snippets": [],
        "system_hints": "",
        "current_iteration": 1,
        "max_iterations": 5,
        "convergence_threshold": 0.85,
        "iteration_history": [],
        "last_output": "",
        "iteration_scores": [],
        "best_score": 0.0,
        "best_output": "",
        "converged": False,
        "quality_converged": False,
        "final_output": "",
        "final_score": 0.0,
    }
    state.update(overrides)
    return state


class TestRouteEntry:
    def test_returns_first_role_in_chain(self):
        from services.orchestration.app.graph.edges import route_entry
        state = _base_state(agent_chain=["planner", "coder"])
        result = route_entry(state)
        assert result == "planner"

    def test_empty_chain_returns_consensus(self):
        from services.orchestration.app.graph.edges import route_entry
        state = _base_state(agent_chain=[])
        result = route_entry(state)
        assert result == "consensus"

    def test_coder_chain_starts_with_coder(self):
        from services.orchestration.app.graph.edges import route_entry
        state = _base_state(agent_chain=["coder", "reviewer"])
        assert route_entry(state) == "coder"

    def test_unknown_role_falls_back_to_consensus(self):
        from services.orchestration.app.graph.edges import route_entry
        state = _base_state(agent_chain=["unknown_role"])
        result = route_entry(state)
        assert result == "consensus"


class TestNextInChain:
    def test_returns_next_role_when_more_exist(self):
        from services.orchestration.app.graph.edges import next_in_chain
        state = _base_state(agent_chain=["planner", "coder", "reviewer"], agent_chain_index=1)
        result = next_in_chain(state)
        assert result == "coder"

    def test_returns_check_convergence_at_end(self):
        from services.orchestration.app.graph.edges import next_in_chain
        state = _base_state(agent_chain=["planner", "coder"], agent_chain_index=2)
        result = next_in_chain(state)
        assert result == "check_convergence"

    def test_first_role_index_zero(self):
        from services.orchestration.app.graph.edges import next_in_chain
        state = _base_state(agent_chain=["debugger"], agent_chain_index=0)
        assert next_in_chain(state) == "debugger"


class TestCheckConvergence:
    def test_high_score_returns_consensus(self):
        from services.orchestration.app.graph.edges import check_convergence
        history = [
            {"iteration": 1, "role": "reviewer", "output": "This implementation is excellent and meets all requirements. SCORE: 0.90 no issues found"},
        ]
        state = _base_state(
            iteration_history=history,
            current_iteration=1,
            convergence_threshold=0.85,
        )
        result = check_convergence(state)
        assert result == "consensus"

    def test_max_iterations_returns_consensus(self):
        from services.orchestration.app.graph.edges import check_convergence
        state = _base_state(
            iteration_history=[],
            current_iteration=5,
            max_iterations=5,
        )
        result = check_convergence(state)
        assert result == "consensus"

    def test_low_score_continues_with_first_role(self):
        from services.orchestration.app.graph.edges import check_convergence
        history = [
            {"iteration": 1, "role": "reviewer", "output": "There are many issues with error handling that need to be fixed. SCORE: 0.40 needs work"},
        ]
        state = _base_state(
            iteration_history=history,
            current_iteration=1,
            max_iterations=5,
            convergence_threshold=0.85,
            agent_chain=["planner", "coder"],
        )
        result = check_convergence(state)
        assert result == "planner"

    def test_empty_history_does_not_converge_early(self):
        from services.orchestration.app.graph.edges import check_convergence
        state = _base_state(current_iteration=1, max_iterations=5)
        result = check_convergence(state)
        # Score will be 0.0, should continue
        assert result != "consensus"
