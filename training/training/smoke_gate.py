"""
Post-training smoke gate.

Sends a canonical per-role chat-completion prompt to the vLLM server using
the staged adapter. Validates the response for emptiness, BPE artifacts,
contamination, and role-shape.

Uses /v1/chat/completions (OpenAI format) to match the ChatML training format
produced by _format_messages_as_prompt in trainer_entry.py.

Returns a result dict: {"pass": bool, "reason": str, "response_len": int}
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

import httpx

from shared.contracts.agent_prompts import AGENT_PROMPTS
from shared.contracts.training import AdapterRecipe

logger = logging.getLogger(__name__)

# Minimal deterministic prompts — fast, no warm-up needed.
_SMOKE_PROMPTS: Dict[str, str] = {
    "coder":      "Write a Python function that returns the sum of two numbers.",
    "debugger":   "Debug this code: def add(a, b): return a - b",
    "reviewer":   "Review this code for bugs: x = 1/0",
    "planner":    "Create a brief plan to implement a login endpoint.",
    "researcher": "Summarize the key concept of LoRA fine-tuning in one paragraph.",
    "critic":     "Evaluate this response: The answer is 42.",
}

_CONTAMINATION_MARKERS = (
    "session_id", "pipeline_metadata", "agent_chain",
    "iteration_scores", "ORCHESTRATION_",
)


def _has_bpe_artifacts(text: str) -> bool:
    return any("Ā" <= ch <= "Ł" for ch in text)


def _is_clean_output(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped[0] in ("{", "[") and stripped[-1] in ("}", "]"):
        try:
            if isinstance(json.loads(stripped), (dict, list)):
                return False
        except json.JSONDecodeError:
            pass
    lower = stripped.lower()
    return not any(m.lower() in lower for m in _CONTAMINATION_MARKERS)


def _shape_ok(role: str, text: str) -> bool:
    t = text.strip()
    if role == "coder":
        return any(kw in t for kw in ("def ", "class ", "```", "return", "import"))
    if role in ("debugger", "reviewer"):
        return len(t) > 20
    if role == "planner":
        return any(c in t for c in ("1.", "-", "*", "Step"))
    if role == "researcher":
        return len(t) > 30
    if role == "critic":
        return len(t) > 10
    return len(t) > 0


def _tool_call_shape_ok(response_json: dict) -> bool:
    """Return True if the response contains a structurally-valid tool_calls array."""
    choices = response_json.get("choices", [])
    if not choices:
        return False
    msg = choices[0].get("message", {})
    tool_calls = msg.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        return False
    first = tool_calls[0]
    return (
        isinstance(first.get("id"), str)
        and isinstance(first.get("function", {}).get("name"), str)
    )


_MINIMAL_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command",
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"],
            },
        },
    }
]


def run_smoke(
    inference_url: str,
    role: str,
    staging_name: str,
    timeout: float = 30.0,
    recipe: Optional[AdapterRecipe] = None,
) -> Dict[str, Any]:
    """
    Synchronous smoke test (runs inside the training subprocess).

    When recipe.tool_call_style == "openai_native": POST /v1/chat/completions
    with tools= and validate that tool_calls are present in the response.
    When recipe.tool_call_style == "none" (or no recipe): POST /v1/completions
    and validate the text response.
    """
    user_prompt = _SMOKE_PROMPTS.get(role, "Say hello.")
    sys_prompt  = AGENT_PROMPTS.get(role, "You are a helpful AI assistant.")

    use_tool_path = recipe is not None and recipe.tool_call_style != "none"

    if use_tool_path:
        payload = {
            "model": staging_name,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 128,
            "temperature": 0.0,
            "tools": _MINIMAL_TOOL_SCHEMAS,
            "tool_choice": "auto",
        }
        try:
            resp = httpx.post(
                f"{inference_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=timeout,
            )
            resp.raise_for_status()
            resp_json = resp.json()
        except Exception as exc:
            return {"pass": False, "reason": f"request_error: {exc}", "response_len": 0}

        if not _tool_call_shape_ok(resp_json):
            return {"pass": False, "reason": "tool_call_shape_fail", "response_len": 0}
        return {"pass": True, "reason": "tool_call_ok", "response_len": 0}
    else:
        payload = {
            "model": staging_name,
            "prompt": (
                f"[SYSTEM]\n{sys_prompt}\n\n"
                f"[USER]\n{user_prompt}\n\n"
                f"[RESPONSE]\n"
            ),
            "max_tokens": 128,
            "temperature": 0.0,
        }
        try:
            resp = httpx.post(
                f"{inference_url}/v1/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=timeout,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["text"]
        except Exception as exc:
            return {"pass": False, "reason": f"request_error: {exc}", "response_len": 0}

        if not text or not text.strip():
            return {"pass": False, "reason": "empty_response", "response_len": 0}
        if _has_bpe_artifacts(text):
            return {"pass": False, "reason": "bpe_artifacts", "response_len": len(text)}
        if not _is_clean_output(text):
            return {"pass": False, "reason": "contaminated_output", "response_len": len(text)}
        if not _shape_ok(role, text):
            return {"pass": False, "reason": f"shape_fail role={role}", "response_len": len(text)}

        return {"pass": True, "reason": "ok", "response_len": len(text)}
