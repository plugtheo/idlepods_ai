"""
Post-training smoke gate.

Loads a staged adapter under '{name}__staging', runs a canonical per-role
prompt against the live vLLM /v1/completions endpoint, and validates the
response for emptiness, BPE artifacts, contamination, and role-shape.

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

_STOP_TOKENS = ["[SYSTEM]", "[USER]", "[ASSISTANT]", "\n[RESPONSE]"]

_CONTAMINATION_MARKERS = (
    "session_id", "pipeline_metadata", "agent_chain",
    "iteration_scores", "ORCHESTRATION_",
)


def _has_bpe_artifacts(text: str) -> bool:
    return any("Ā" <= ch <= "Ń" for ch in text)


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


def run_smoke(
    inference_url: str,
    role: str,
    staging_name: str,
    timeout: float = 30.0,
    recipe: Optional[AdapterRecipe] = None,
) -> Dict[str, Any]:
    """
    Synchronous smoke test (runs inside the training subprocess).

    Sends the canonical prompt for *role* to the vLLM server using the staged
    adapter *staging_name* via POST /v1/completions.
    """
    is_tool_role = recipe is not None and recipe.tool_call_style != "none" and role in ("coder", "debugger")

    prompt_text = _SMOKE_PROMPTS.get(role, "Say hello.")
    sys_prompt  = AGENT_PROMPTS.get(role, "You are a helpful AI assistant.")

    if is_tool_role:
        # Chat-completions endpoint with tools= so vLLM renders the tool-call schema.
        _read_file_tool = {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the contents of a file.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }
        chat_payload = {
            "model": staging_name,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": "Read the file /tmp/main.py and summarise it."},
            ],
            "tools": [_read_file_tool],
            "tool_choice": "auto",
            "max_tokens": 128,
            "temperature": 0.0,
        }
        try:
            resp = httpx.post(
                f"{inference_url}/v1/chat/completions",
                json=chat_payload,
                headers={"Content-Type": "application/json"},
                timeout=timeout,
            )
            resp.raise_for_status()
            resp_json = resp.json()
        except Exception as exc:
            return {"pass": False, "reason": f"request_error: {exc}", "response_len": 0}

        if not _tool_call_shape_ok(resp_json):
            raw = json.dumps(resp_json.get("choices", [{}])[0])
            return {"pass": False, "reason": f"tool_call_shape_fail: {raw[:200]}", "response_len": 0}
        return {"pass": True, "reason": "tool_call_ok", "response_len": len(json.dumps(resp_json))}

    prompt = (
        f"[SYSTEM]\n{sys_prompt}\n\n"
        f"[USER]\n{prompt_text}\n\n"
        f"[RESPONSE]\n"
    )
    payload = {
        "model":       staging_name,
        "prompt":      prompt,
        "max_tokens":  128,
        "temperature": 0.0,
        "stop":        _STOP_TOKENS,
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
