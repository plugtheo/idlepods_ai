"""
orchestration/app/tools/runner.py
===================================
Sandboxed tool runner for any tool-using agent role (coder, researcher, etc.).

Supported tools
---------------
read_file(path, start?, end?)   — return lines from a file
write_file(path, content)       — write (overwrite) a file
list_files(glob)                — return matching paths
run_command(command)            — run an allowed command

Safety
------
_safe_path() rejects: path traversal (..), absolute paths, and paths
that touch sensitive dirs/files ({.git, .env, secrets/}).

Wire format
-----------
The model emits native OpenAI tool calls:
    {"id": "call_xyz", "function": {"name": "read_file", "arguments": "{\"path\": \"src/foo.py\"}"}}

build_tool_schemas(allowlist) returns the OpenAI function schemas filtered by the
per-role allowlist from settings.role_tools_enabled.  New tools (e.g. researcher's
browser tools) register here by adding an entry to _TOOL_REGISTRY and
_TOOL_SCHEMAS — no graph or pipeline change needed.

execute_tool_call() accepts one OpenAI tool_call object and returns {tool, output, error, id}.
"""

from __future__ import annotations

import glob as _glob_mod
import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DENYLIST = {".git", ".env", "secrets"}
_TOOL_OUTPUT_TRUNCATE_CHARS = 4000


class ToolError(Exception):
    """Raised when a tool call fails; message is already truncated."""


def _safe_path(raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute():
        raise ToolError(f"Absolute paths are not allowed: {raw!r}")
    if ".." in p.parts:
        raise ToolError(f"Path traversal is not allowed: {raw!r}")
    if any(part in _DENYLIST for part in p.parts):
        raise ToolError(f"Path touches a denied location: {raw!r}")
    return p


def _truncate(text: str) -> str:
    cap = _TOOL_OUTPUT_TRUNCATE_CHARS
    if len(text) > cap:
        return text[:cap] + f"\n… (truncated at {cap} chars)"
    return text


def read_file(path: str, start: Optional[int] = None, end: Optional[int] = None) -> str:
    safe = _safe_path(path)
    try:
        lines = safe.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        raise ToolError(f"File not found: {path!r}")
    except Exception as exc:
        raise ToolError(_truncate(str(exc)))
    if start is not None or end is not None:
        lines = lines[(start or 0):end]
    return _truncate("\n".join(lines))


def write_file(path: str, content: str) -> str:
    safe = _safe_path(path)
    try:
        safe.parent.mkdir(parents=True, exist_ok=True)
        safe.write_text(content, encoding="utf-8")
    except Exception as exc:
        raise ToolError(_truncate(str(exc)))
    return f"Written {len(content)} bytes to {path!r}"


def list_files(glob: str) -> str:
    safe_matches = []
    for m in sorted(_glob_mod.glob(glob, recursive=True)):
        try:
            _safe_path(m)
            safe_matches.append(m)
        except ToolError:
            pass
    return _truncate("\n".join(safe_matches) if safe_matches else "(no matches)")


_RUN_COMMAND_ALLOWLIST = {"pytest", "ruff", "mypy"}


def run_command(command: str) -> str:
    parts = command.split()
    if not parts or parts[0] not in _RUN_COMMAND_ALLOWLIST:
        raise ToolError(f"Command not in allowlist {sorted(_RUN_COMMAND_ALLOWLIST)}: {parts[0]!r}")
    try:
        result = subprocess.run(parts, capture_output=True, text=True, timeout=60)
        combined = result.stdout + ("\n" + result.stderr if result.stderr.strip() else "")
        return _truncate(combined or "(no output)")
    except subprocess.TimeoutExpired:
        raise ToolError("Command timed out after 60s")
    except Exception as exc:
        raise ToolError(_truncate(str(exc)))


# ── Registry ─────────────────────────────────────────────────────────────

_TOOL_REGISTRY: Dict[str, Any] = {
    "read_file":   read_file,
    "write_file":  write_file,
    "list_files":  list_files,
    "run_command": run_command,
}

_TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "read_file": {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read lines from a repo-relative file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path":  {"type": "string", "description": "Repo-relative file path."},
                    "start": {"type": "integer", "description": "Start line index (0-based, optional)."},
                    "end":   {"type": "integer", "description": "End line index (exclusive, optional)."},
                },
                "required": ["path"],
            },
        },
    },
    "write_file": {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write (overwrite) a repo-relative file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path":    {"type": "string", "description": "Repo-relative file path."},
                    "content": {"type": "string", "description": "Full file content to write."},
                },
                "required": ["path", "content"],
            },
        },
    },
    "list_files": {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List repo-relative file paths matching a glob pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "glob": {"type": "string", "description": "Glob pattern (e.g. 'src/**/*.py')."},
                },
                "required": ["glob"],
            },
        },
    },
    "run_command": {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run an allowed command (pytest, ruff, mypy) and return combined stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Full command string, e.g. 'pytest src/tests/ -x'"},
                },
                "required": ["command"],
            },
        },
    },
}


def build_tool_schemas(allowlist: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Return OpenAI function schemas for registered tools, filtered by *allowlist*.

    When *allowlist* is provided, only tools whose names appear in it are returned.
    New tools register by adding entries to both _TOOL_REGISTRY and _TOOL_SCHEMAS.
    """
    names = list(_TOOL_SCHEMAS.keys()) if allowlist is None else [n for n in allowlist if n in _TOOL_SCHEMAS]
    return [_TOOL_SCHEMAS[n] for n in names]


def execute_tool_call(call: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch one OpenAI-format tool call. Returns {tool, output, error, id}."""
    func = call.get("function", {})
    name = func.get("name", "")
    call_id = call.get("id", "")
    try:
        kwargs = json.loads(func.get("arguments", "{}"))
    except json.JSONDecodeError as exc:
        return {"tool": name, "output": _truncate(f"Bad arguments JSON: {exc}"), "error": True, "id": call_id}

    fn = _TOOL_REGISTRY.get(name)
    if fn is None:
        return {"tool": name, "output": _truncate(f"Unknown tool: {name!r}"), "error": True, "id": call_id}
    try:
        return {"tool": name, "output": fn(**kwargs), "error": False, "id": call_id}
    except ToolError as exc:
        return {"tool": name, "output": str(exc), "error": True, "id": call_id}
    except TypeError as exc:
        return {"tool": name, "output": _truncate(f"Bad arguments for {name!r}: {exc}"), "error": True, "id": call_id}
    except Exception as exc:
        return {"tool": name, "output": _truncate(str(exc)), "error": True, "id": call_id}
