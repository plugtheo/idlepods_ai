"""Tests for smoke_gate tool-call verification."""
import json
from unittest.mock import MagicMock, patch

import pytest

from shared.contracts.training import AdapterRecipe
from training.training.smoke_gate import run_smoke, _tool_call_shape_ok


def _make_recipe(tool_call_style="openai_native"):
    return AdapterRecipe(
        target_modules=["q_proj"],
        tool_call_style=tool_call_style,
    )


# ---------------------------------------------------------------------------
# _tool_call_shape_ok unit tests
# ---------------------------------------------------------------------------

def test_tool_call_shape_ok_valid():
    resp = {
        "choices": [{
            "message": {
                "tool_calls": [{"id": "c1", "type": "function",
                                "function": {"name": "read_file", "arguments": "{}"}}]
            }
        }]
    }
    assert _tool_call_shape_ok(resp) is True


def test_tool_call_shape_ok_missing_tool_calls():
    resp = {"choices": [{"message": {"content": "plain text"}}]}
    assert _tool_call_shape_ok(resp) is False


def test_tool_call_shape_ok_empty_list():
    resp = {"choices": [{"message": {"tool_calls": []}}]}
    assert _tool_call_shape_ok(resp) is False


# ---------------------------------------------------------------------------
# run_smoke integration tests (mocked HTTP)
# ---------------------------------------------------------------------------

def _mock_httpx_post(response_json: dict, status_code: int = 200):
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = response_json
    mock_resp.raise_for_status.return_value = None
    return mock_resp


def test_smoke_fails_when_adapter_outputs_raw_text():
    """Gate must fail when a tool-using role emits plain text instead of tool_calls."""
    recipe = _make_recipe()
    bad_response = {
        "choices": [{"message": {"content": "I will read the file for you.", "tool_calls": None}}]
    }
    with patch("httpx.post", return_value=_mock_httpx_post(bad_response)):
        result = run_smoke("http://inference:8010", "coder", "coder__staging",
                           recipe=recipe)
    assert result["pass"] is False
    assert "tool_call_shape_fail" in result["reason"]


def test_smoke_passes_when_tool_calls_present():
    recipe = _make_recipe()
    good_response = {
        "choices": [{
            "message": {
                "tool_calls": [{"id": "c1", "type": "function",
                                "function": {"name": "read_file", "arguments": '{"path":"/tmp/f.py"}'}}]
            }
        }]
    }
    with patch("httpx.post", return_value=_mock_httpx_post(good_response)):
        result = run_smoke("http://inference:8010", "coder", "coder__staging",
                           recipe=recipe)
    assert result["pass"] is True
    assert result["reason"] == "tool_call_ok"


def test_smoke_non_tool_role_uses_completions_path():
    """Non-tool roles (planner, researcher) use the legacy /v1/completions path."""
    recipe = _make_recipe(tool_call_style="none")
    completions_response = {
        "choices": [{"text": "1. Set up the project structure.\n2. Implement endpoints."}]
    }
    with patch("httpx.post", return_value=_mock_httpx_post(completions_response)) as mock_post:
        result = run_smoke("http://inference:8010", "planner", "planner__staging",
                           recipe=recipe)
    url_called = mock_post.call_args[0][0]
    assert "/v1/completions" in url_called
    assert result["pass"] is True
