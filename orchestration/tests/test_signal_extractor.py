"""
Tests for signal_extractor.py.

Covers:
  - extract_signals(): empty/tool-only history, each signal flag, score parsing,
    multi-signal outputs, no-signal bland output, tool entries skipped correctly
  - format_signals_for_prompt(): empty role, each active signal, no-signal fallback,
    score formatting, role+length in header
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

import pytest

from orchestration.app.graph.signal_extractor import (
    AgentSignals,
    extract_signals,
    format_signals_for_prompt,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _history_entry(role: str, output: str, iteration: int = 1) -> Dict[str, Any]:
    return {"role": role, "output": output, "iteration": iteration, "timestamp": _now()}


def _state(entries) -> Dict[str, Any]:
    return {"iteration_history": entries}


# ─── extract_signals ──────────────────────────────────────────────────────────

class TestExtractSignals:

    def test_empty_history_returns_zero_signals(self):
        signals = extract_signals(_state([]))
        assert signals == AgentSignals()
        assert signals.role == ""
        assert signals.output_length == 0

    def test_none_history_returns_zero_signals(self):
        signals = extract_signals({"iteration_history": None})
        assert signals == AgentSignals()

    def test_tool_only_history_returns_zero_signals(self):
        entries = [
            _history_entry("tool", "file content from read_file"),
            _history_entry("tool", "another tool result"),
        ]
        signals = extract_signals(_state(entries))
        assert signals == AgentSignals()

    def test_role_captured_from_last_agent_entry(self):
        entries = [
            _history_entry("planner", "I made a plan."),
            _history_entry("coder", "I wrote the function."),
        ]
        signals = extract_signals(_state(entries))
        assert signals.role == "coder"

    def test_output_length_measured(self):
        output = "This is a short output."
        signals = extract_signals(_state([_history_entry("coder", output)]))
        assert signals.output_length == len(output)

    def test_score_extracted_from_reviewer_output(self):
        output = "The code looks good. SCORE: 0.87 — minor issues only."
        signals = extract_signals(_state([_history_entry("reviewer", output)]))
        assert signals.score_value == pytest.approx(0.87)

    def test_score_case_insensitive(self):
        output = "score: 0.55 overall."
        signals = extract_signals(_state([_history_entry("critic", output)]))
        assert signals.score_value == pytest.approx(0.55)

    def test_no_score_in_output_is_none(self):
        signals = extract_signals(_state([_history_entry("reviewer", "Looks fine, no issues.")]))
        assert signals.score_value is None

    def test_has_errors_detected(self):
        output = "I found a traceback in the test run."
        signals = extract_signals(_state([_history_entry("debugger", output)]))
        assert signals.has_errors is True

    def test_has_errors_exception_keyword(self):
        signals = extract_signals(_state([_history_entry("coder", "An exception was raised.")]))
        assert signals.has_errors is True

    def test_has_errors_false_when_clean(self):
        signals = extract_signals(_state([_history_entry("coder", "All tests passed.")]))
        assert signals.has_errors is False

    def test_has_completion_done_keyword(self):
        signals = extract_signals(_state([_history_entry("coder", "The task is done.")]))
        assert signals.has_completion is True

    def test_has_completion_implemented_keyword(self):
        signals = extract_signals(_state([_history_entry("coder", "I implemented the feature.")]))
        assert signals.has_completion is True

    def test_has_completion_false_in_progress(self):
        signals = extract_signals(_state([_history_entry("coder", "Still working on it.")]))
        assert signals.has_completion is False

    def test_has_uncertainty_detected(self):
        output = "I'm unsure about the best approach here."
        signals = extract_signals(_state([_history_entry("researcher", output)]))
        assert signals.has_uncertainty is True

    def test_has_uncertainty_unclear_keyword(self):
        output = "The requirements are unclear."
        signals = extract_signals(_state([_history_entry("planner", output)]))
        assert signals.has_uncertainty is True

    def test_has_uncertainty_false_when_confident(self):
        signals = extract_signals(_state([_history_entry("coder", "The fix is straightforward.")]))
        assert signals.has_uncertainty is False

    def test_has_code_changes_wrote_file(self):
        signals = extract_signals(_state([_history_entry("coder", "I wrote file auth.py.")]))
        assert signals.has_code_changes is True

    def test_has_code_changes_modified_function(self):
        signals = extract_signals(_state([_history_entry("coder", "Modified function login().")]))
        assert signals.has_code_changes is True

    def test_has_code_changes_false_no_changes(self):
        signals = extract_signals(_state([_history_entry("reviewer", "The code quality is good.")]))
        assert signals.has_code_changes is False

    def test_has_blockers_blocked_keyword(self):
        signals = extract_signals(_state([_history_entry("coder", "I am blocked by missing dependency.")]))
        assert signals.has_blockers is True

    def test_has_blockers_cannot_proceed(self):
        signals = extract_signals(_state([_history_entry("coder", "Cannot proceed without the config.")]))
        assert signals.has_blockers is True

    def test_has_blockers_false_when_unblocked(self):
        signals = extract_signals(_state([_history_entry("coder", "Ready to proceed with the next step.")]))
        assert signals.has_blockers is False

    def test_multiple_signals_simultaneously(self):
        output = (
            "I wrote file auth.py but an exception was raised. "
            "SCORE: 0.60. The fix is done."
        )
        signals = extract_signals(_state([_history_entry("coder", output)]))
        assert signals.has_code_changes is True
        assert signals.has_errors is True
        assert signals.score_value == pytest.approx(0.60)
        assert signals.has_completion is True

    def test_bland_output_has_no_signals(self):
        output = "I reviewed the situation and have some thoughts."
        signals = extract_signals(_state([_history_entry("reviewer", output)]))
        assert signals.score_value is None
        assert signals.has_errors is False
        assert signals.has_completion is False
        assert signals.has_uncertainty is False
        assert signals.has_code_changes is False
        assert signals.has_blockers is False

    def test_tool_entries_skipped_uses_preceding_agent_entry(self):
        """Tool result after coder should not override coder as signal source."""
        entries = [
            _history_entry("coder", "I wrote file main.py."),
            {"role": "tool", "output": "file read result", "iteration": 1, "timestamp": _now()},
        ]
        signals = extract_signals(_state(entries))
        assert signals.role == "coder"
        assert signals.has_code_changes is True

    def test_full_output_field_fallback(self):
        """full_output key is used when output key is absent."""
        entry = {"role": "coder", "full_output": "I fixed the bug.", "iteration": 1}
        signals = extract_signals({"iteration_history": [entry]})
        assert signals.has_completion is True

    def test_signals_to_dict_roundtrip(self):
        output = "SCORE: 0.75. Implementation done."
        signals = extract_signals(_state([_history_entry("reviewer", output)]))
        d = signals.to_dict()
        assert d["role"] == "reviewer"
        assert d["score_value"] == pytest.approx(0.75)
        assert d["has_completion"] is True


# ─── format_signals_for_prompt ────────────────────────────────────────────────

class TestFormatSignalsForPrompt:

    def test_empty_role_returns_empty_string(self):
        signals = AgentSignals()
        assert format_signals_for_prompt(signals) == ""

    def test_includes_role_in_header(self):
        signals = AgentSignals(role="coder", output_length=100)
        result = format_signals_for_prompt(signals)
        assert "last_role=coder" in result

    def test_includes_output_length_in_header(self):
        signals = AgentSignals(role="reviewer", output_length=250)
        result = format_signals_for_prompt(signals)
        assert "output_len=250" in result

    def test_no_strong_signals_fallback(self):
        signals = AgentSignals(role="planner", output_length=50)
        result = format_signals_for_prompt(signals)
        assert "no_strong_signals" in result

    def test_score_formatted_two_decimals(self):
        signals = AgentSignals(role="critic", output_length=80, score_value=0.9)
        result = format_signals_for_prompt(signals)
        assert "score=0.90" in result

    def test_completion_detected_in_output(self):
        signals = AgentSignals(role="coder", output_length=60, has_completion=True)
        result = format_signals_for_prompt(signals)
        assert "completion_detected" in result

    def test_errors_detected_in_output(self):
        signals = AgentSignals(role="debugger", output_length=120, has_errors=True)
        result = format_signals_for_prompt(signals)
        assert "errors_detected" in result

    def test_uncertainty_detected_in_output(self):
        signals = AgentSignals(role="researcher", output_length=90, has_uncertainty=True)
        result = format_signals_for_prompt(signals)
        assert "uncertainty_detected" in result

    def test_code_changes_detected_in_output(self):
        signals = AgentSignals(role="coder", output_length=150, has_code_changes=True)
        result = format_signals_for_prompt(signals)
        assert "code_changes_detected" in result

    def test_blockers_detected_in_output(self):
        signals = AgentSignals(role="coder", output_length=70, has_blockers=True)
        result = format_signals_for_prompt(signals)
        assert "blockers_detected" in result

    def test_multiple_active_signals_comma_separated(self):
        signals = AgentSignals(
            role="coder", output_length=200,
            score_value=0.75, has_errors=True, has_completion=True,
        )
        result = format_signals_for_prompt(signals)
        assert "score=0.75" in result
        assert "errors_detected" in result
        assert "completion_detected" in result

    def test_starts_with_markdown_header(self):
        signals = AgentSignals(role="reviewer", output_length=100)
        result = format_signals_for_prompt(signals)
        assert result.startswith("## Output Signals")
