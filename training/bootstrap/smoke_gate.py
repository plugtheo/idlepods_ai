"""
Post-training smoke gate.

Per-role verification of a staged adapter before promotion.  Dispatch is keyed
by role (not by recipe.tool_call_style) so the gate can verify each role
against the inference contract it will actually be used under:

  tool_or_text — coder, debugger: accept either a valid tool_calls array
                 or a clean code/text response. Forcing tool calls would
                 reject otherwise-good adapters that solve the prompt directly.
  json_schema  — reviewer, critic: pass response_schema (same JSON Schema
                 nodes.py uses) so vLLM's guided decoding is active; then
                 validate the response via Pydantic. This is byte-identical
                 to the production path, so the gate fails for any adapter
                 that cannot produce schema-conformant output.
  text         — planner, researcher: chat-completion without tools/schema;
                 shape + cleanliness checks on message.content.

Returns {"pass": bool, "reason": str, "response_len": int}.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional, Tuple

import httpx
from pydantic import ValidationError

from shared.contracts.agent_prompts import AGENT_PROMPTS
from shared.contracts.evaluator_schemas import EVALUATOR_SCHEMAS
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

# Verification mode per role. The smoke gate dispatches on this map — recipe's
# tool_call_style is the *training* contract; smoke mode is the *verification*
# contract. They are allowed to differ (e.g. coder is trained with tool-call
# data but smoke accepts either a tool call or clean code text).
_SMOKE_MODE: Dict[str, str] = {
    "coder":      "tool_or_text",
    "debugger":   "tool_or_text",
    "reviewer":   "json_schema",
    "critic":     "json_schema",
    "planner":    "text",
    "researcher": "text",
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


# ---------------------------------------------------------------------------
# Per-mode verification helpers
# ---------------------------------------------------------------------------

def _validate_text(role: str, text: str) -> Dict[str, Any]:
    """Common text-shape gate used by text and tool_or_text modes."""
    if not text or not text.strip():
        return {"pass": False, "reason": "empty_response", "response_len": 0}
    if _has_bpe_artifacts(text):
        return {"pass": False, "reason": "bpe_artifacts", "response_len": len(text)}
    if not _is_clean_output(text):
        return {"pass": False, "reason": "contaminated_output", "response_len": len(text)}
    if not _shape_ok(role, text):
        return {"pass": False, "reason": f"shape_fail role={role}", "response_len": len(text)}
    return {"pass": True, "reason": "ok", "response_len": len(text)}


def _post_chat(inference_url: str, payload: dict, timeout: float) -> Tuple[Optional[dict], Optional[str]]:
    """POST to /v1/chat/completions. Returns (response_json, error_reason)."""
    try:
        resp = httpx.post(
            f"{inference_url}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json(), None
    except Exception as exc:
        return None, f"request_error: {exc}"


def _smoke_text(
    inference_url: str, role: str, staging_name: str,
    sys_prompt: str, user_prompt: str, timeout: float,
) -> Dict[str, Any]:
    payload = {
        "model": staging_name,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 256,
        "temperature": 0.0,
    }
    rj, err = _post_chat(inference_url, payload, timeout)
    if err:
        return {"pass": False, "reason": err, "response_len": 0}
    text = (rj or {}).get("choices", [{}])[0].get("message", {}).get("content") or ""
    return _validate_text(role, text)


def _smoke_tool_or_text(
    inference_url: str, role: str, staging_name: str,
    sys_prompt: str, user_prompt: str, timeout: float,
) -> Dict[str, Any]:
    payload = {
        "model": staging_name,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 256,
        "temperature": 0.0,
        "tools": _MINIMAL_TOOL_SCHEMAS,
        "tool_choice": "auto",
    }
    rj, err = _post_chat(inference_url, payload, timeout)
    if err:
        return {"pass": False, "reason": err, "response_len": 0}

    if _tool_call_shape_ok(rj or {}):
        return {"pass": True, "reason": "tool_call_ok", "response_len": 0}

    text = (rj or {}).get("choices", [{}])[0].get("message", {}).get("content") or ""
    return _validate_text(role, text)


def _smoke_json_schema(
    inference_url: str, role: str, staging_name: str,
    sys_prompt: str, user_prompt: str, timeout: float,
) -> Dict[str, Any]:
    schema_cls = EVALUATOR_SCHEMAS.get(role)
    if schema_cls is None:
        return {"pass": False, "reason": f"no_schema_for_role:{role}", "response_len": 0}

    json_schema = schema_cls.model_json_schema()
    payload = {
        "model": staging_name,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 512,
        "temperature": 0.0,
        # OpenAI-style structured outputs wrapper. vLLM (with xgrammar) honours
        # this and applies the same grammar mask used at inference time by
        # orchestration/app/graph/nodes.py — so smoke tests the real path.
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": role, "schema": json_schema, "strict": True},
        },
    }
    rj, err = _post_chat(inference_url, payload, timeout)
    if err:
        return {"pass": False, "reason": err, "response_len": 0}
    content = (rj or {}).get("choices", [{}])[0].get("message", {}).get("content") or ""
    if not content.strip():
        return {"pass": False, "reason": "empty_response", "response_len": 0}
    try:
        schema_cls.model_validate_json(content)
    except ValidationError as exc:
        err_summary = exc.errors()[:1]
        return {"pass": False, "reason": f"json_schema_fail:{err_summary}", "response_len": len(content)}
    return {"pass": True, "reason": "json_schema_ok", "response_len": len(content)}


def run_smoke(
    inference_url: str,
    role: str,
    staging_name: str,
    timeout: float = 30.0,
    recipe: Optional[AdapterRecipe] = None,
) -> Dict[str, Any]:
    """
    Synchronous smoke test (runs inside the training subprocess).

    Dispatches on _SMOKE_MODE[role]:
      tool_or_text → accept tool_calls OR clean text matching role shape
      json_schema  → vLLM guided decoding + Pydantic schema validation
      text         → clean text matching role shape (no tools)

    *recipe* is accepted for API stability but no longer drives the dispatch —
    role intent does. The argument can be removed once all callers are updated.
    """
    user_prompt = _SMOKE_PROMPTS.get(role, "Say hello.")
    sys_prompt  = AGENT_PROMPTS.get(role, "You are a helpful AI assistant.")
    mode = _SMOKE_MODE.get(role, "text")

    if mode == "json_schema":
        return _smoke_json_schema(inference_url, role, staging_name, sys_prompt, user_prompt, timeout)
    if mode == "tool_or_text":
        return _smoke_tool_or_text(inference_url, role, staging_name, sys_prompt, user_prompt, timeout)
    return _smoke_text(inference_url, role, staging_name, sys_prompt, user_prompt, timeout)


# ---------------------------------------------------------------------------
# Regression comparison — new vs previous active adapter
# ---------------------------------------------------------------------------

REGRESSION_TOLERANCE: float = 0.02  # allow up to 2 % drop before blocking promotion


def _score_response(role: str, text: str) -> float:
    """Heuristic 0.0–1.0 quality score for a plain-text smoke response."""
    if not text or not text.strip():
        return 0.0
    if _has_bpe_artifacts(text):
        return 0.1
    if not _is_clean_output(text):
        return 0.1
    score = 0.5
    if _shape_ok(role, text):
        score += 0.3
    ln = len(text.strip())
    if 80 <= ln <= 600:
        score += 0.2
    elif ln > 600:
        score += 0.1
    return min(score, 1.0)


def _score_response_json(role: str, text: str) -> float:
    """Score a JSON-mode smoke response by schema validity + content density."""
    schema_cls = EVALUATOR_SCHEMAS.get(role)
    if schema_cls is None or not text or not text.strip():
        return 0.0
    try:
        obj = schema_cls.model_validate_json(text)
    except ValidationError:
        return 0.1
    score = 0.5
    # Content-density bonuses — non-empty fields signal a substantive review.
    list_fields = ("strengths", "issues", "suggestions", "blockers")
    str_fields  = ("verdict", "improvement")
    for f in list_fields:
        v = getattr(obj, f, None)
        if isinstance(v, list) and v:
            score += 0.1
    for f in str_fields:
        v = getattr(obj, f, None)
        if isinstance(v, str) and v.strip():
            score += 0.1
    return min(score, 1.0)


def run_regression_comparison(
    inference_url: str,
    role: str,
    previous_name: Optional[str],
    new_name: str,
    recipe: Optional[AdapterRecipe] = None,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """
    Run the canonical smoke prompt against both the previous active adapter and
    the new candidate, score each via the role-appropriate scorer, and return
    a comparison dict for HistoryEntry / diff report.

    Cold-start (previous_name=None): skips previous query; won=True, delta=None.
    Any query failure returns score=0.0 for that adapter — the delta is still
    computed so a persistent inference error shows as a large regression.
    """
    user_prompt = _SMOKE_PROMPTS.get(role, "Say hello.")
    sys_prompt = AGENT_PROMPTS.get(role, "You are a helpful AI assistant.")
    mode = _SMOKE_MODE.get(role, "text")

    def _query(adapter_name: str) -> Tuple[float, str]:
        if mode == "json_schema":
            schema_cls = EVALUATOR_SCHEMAS.get(role)
            if schema_cls is None:
                return 0.0, f"no_schema_for_role:{role}"
            payload = {
                "model": adapter_name,
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": 512,
                "temperature": 0.0,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": role, "schema": schema_cls.model_json_schema(), "strict": True},
                },
            }
            rj, err = _post_chat(inference_url, payload, timeout)
            if err:
                return 0.0, err
            text = (rj or {}).get("choices", [{}])[0].get("message", {}).get("content") or ""
            return _score_response_json(role, text), text

        if mode == "tool_or_text":
            payload = {
                "model": adapter_name,
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": 256,
                "temperature": 0.0,
                "tools": _MINIMAL_TOOL_SCHEMAS,
                "tool_choice": "auto",
            }
            rj, err = _post_chat(inference_url, payload, timeout)
            if err:
                return 0.0, err
            if _tool_call_shape_ok(rj or {}):
                raw = str((rj or {}).get("choices", [{}])[0].get("message", {}))
                return 0.9, raw
            text = (rj or {}).get("choices", [{}])[0].get("message", {}).get("content") or ""
            return _score_response(role, text), text

        # text mode
        payload = {
            "model": adapter_name,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 256,
            "temperature": 0.0,
        }
        rj, err = _post_chat(inference_url, payload, timeout)
        if err:
            return 0.0, err
        text = (rj or {}).get("choices", [{}])[0].get("message", {}).get("content") or ""
        return _score_response(role, text), text

    new_score, new_text = _query(new_name)

    if not previous_name:
        logger.info(
            "regression_comparison role=%s mode=%s cold_start=True new_score=%.3f",
            role, mode, new_score,
        )
        return {
            "previous_score": None,
            "new_score": round(new_score, 4),
            "delta": None,
            "won": True,
            "prompts_n": 1,
            "details": [{"prompt": user_prompt, "new_score": round(new_score, 4)}],
            "cold_start": True,
            "mode": mode,
        }

    prev_score, prev_text = _query(previous_name)
    delta = round(new_score - prev_score, 4)
    won = delta >= -REGRESSION_TOLERANCE

    logger.info(
        "regression_comparison role=%s mode=%s prev_score=%.3f new_score=%.3f delta=%+.4f won=%s",
        role, mode, prev_score, new_score, delta, won,
    )
    return {
        "previous_score": round(prev_score, 4),
        "new_score": round(new_score, 4),
        "delta": delta,
        "won": won,
        "prompts_n": 1,
        "details": [
            {
                "prompt": user_prompt,
                "prev_score": round(prev_score, 4),
                "prev_len": len(prev_text),
                "new_score": round(new_score, 4),
                "new_len": len(new_text),
            }
        ],
        "cold_start": False,
        "mode": mode,
    }
