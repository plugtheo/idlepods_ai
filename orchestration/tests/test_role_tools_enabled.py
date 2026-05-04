"""
Step 12/13 readiness test — test_role_tools_enabled.py

- Configure role_tools_enabled["researcher"] = ["fake_browser"]
- Assert _tool_using_roles() contains "researcher"
- Assert build_tool_schemas(["fake_browser"]) returns exactly one schema
  (after registering the fake tool)
- Assert no {"coder"} literal in nodes.py / pipeline.py / runner.py / recorder.py
  for routing/scope decisions (grep guard)
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# ── Grep guard: no hardcoded {"coder"} for tool routing ────────────────────

_GUARDED_FILES = [
    "orchestration/app/graph/nodes.py",
    "orchestration/app/graph/pipeline.py",
    "orchestration/app/tools/runner.py",
    "orchestration/app/experience/recorder.py",
]

PROJECT_ROOT = Path(__file__).parents[2]

_LITERAL_PATTERN = re.compile(r'\{"coder"\}')


@pytest.mark.parametrize("rel_path", _GUARDED_FILES)
def test_no_coder_literal_in_tool_routing(rel_path):
    """No {'coder'} set literal must appear in the guarded files."""
    source = (PROJECT_ROOT / rel_path).read_text(encoding="utf-8")
    matches = _LITERAL_PATTERN.findall(source)
    assert not matches, (
        f"Found hardcoded {{'coder'}} literal in {rel_path}. "
        "Tool routing must be derived from settings.role_tools_enabled."
    )


# ── _tool_using_roles() picks up dynamically configured roles ───────────────

def test_tool_using_roles_includes_researcher_when_configured():
    """_tool_using_roles() returns 'researcher' when researcher has non-empty tools."""
    from orchestration.app.graph.nodes import _tool_using_roles

    mock_settings = MagicMock()
    mock_settings.role_tools_enabled = {
        "coder":      ["read_file"],
        "researcher": ["fake_browser"],
        "planner":    [],
    }

    with patch("orchestration.app.graph.nodes.settings", mock_settings):
        roles = _tool_using_roles()

    assert "researcher" in roles
    assert "coder" in roles
    assert "planner" not in roles


# ── build_tool_schemas(allowlist) filters by name ───────────────────────────

def test_build_tool_schemas_allowlist_filters():
    """build_tool_schemas(allowlist) returns only schemas for allowlisted tool names."""
    from orchestration.app.tools.runner import build_tool_schemas, _TOOL_SCHEMAS

    # Register a fake tool schema for the test
    _TOOL_SCHEMAS["fake_browser"] = {
        "type": "function",
        "function": {
            "name": "fake_browser",
            "description": "Fake browser for testing.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }
    try:
        schemas = build_tool_schemas(["fake_browser"])
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "fake_browser"
    finally:
        _TOOL_SCHEMAS.pop("fake_browser", None)


def test_build_tool_schemas_no_allowlist_returns_all():
    """build_tool_schemas() with no allowlist returns all registered schemas."""
    from orchestration.app.tools.runner import build_tool_schemas, _TOOL_SCHEMAS
    schemas = build_tool_schemas()
    assert len(schemas) == len(_TOOL_SCHEMAS)


def test_build_tool_schemas_unknown_tool_ignored():
    """build_tool_schemas(['nonexistent_tool']) returns empty list."""
    from orchestration.app.tools.runner import build_tool_schemas
    schemas = build_tool_schemas(["nonexistent_tool"])
    assert schemas == []
