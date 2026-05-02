"""Tests for recorder._build_contributions tool-call grouping."""
import pytest

from orchestration.app.experience.recorder import _build_contributions


def _assistant(content=None, tool_calls=None):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def _tool(tool_call_id, content):
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


def test_no_tool_calls():
    history = [
        _assistant(content="Here is the answer."),
    ]
    contribs = _build_contributions(history, "coder", 0.8, 1, [])
    assert len(contribs) == 1
    assert contribs[0].output == "Here is the answer."
    assert contribs[0].tool_calls is None
    assert contribs[0].tool_results is None


def test_single_tool_round_grouped():
    tc = [{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]
    history = [
        _assistant(tool_calls=tc),
        _tool("c1", "file contents"),
        _assistant(content="Done."),
    ]
    contribs = _build_contributions(history, "coder", 0.9, 1, [])
    assert len(contribs) == 1
    c = contribs[0]
    assert c.tool_calls == tc
    assert len(c.tool_results) == 1
    assert c.tool_results[0]["tool_call_id"] == "c1"
    assert c.output == "Done."


def test_tool_result_role_skipped_for_compat():
    """Rows with role='tool_result' (Plan 1 legacy) must be silently skipped."""
    tc = [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
    history = [
        _assistant(tool_calls=tc),
        {"role": "tool_result", "content": "old format"},
        _assistant(content="Final."),
    ]
    contribs = _build_contributions(history, "coder", 0.8, 1, [])
    assert contribs[0].output == "Final."


def test_unpaired_tool_row_skipped():
    history = [
        _tool("orphan", "result with no prior assistant+tool_calls"),
        _assistant(content="Answer."),
    ]
    contribs = _build_contributions(history, "coder", 0.7, 1, [])
    assert contribs[0].output == "Answer."
    assert contribs[0].tool_calls is None
