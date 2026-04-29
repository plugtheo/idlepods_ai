#!/usr/bin/env python3
"""
Adapter Evaluation Tool
=======================
Compare LoRA adapter performance against the base model (or a previous adapter
version) across all agent roles.

The tool samples N instructions from each role's training dataset, runs them
through the current adapter AND the base model via the vLLM API, computes
per-sample quality metrics, and produces a terminal comparison table plus an
HTML report.

Requirements
------------
  pip install requests           # mandatory
  pip install rich               # optional — gives coloured terminal output

Usage
-----
  # Compare all adapters vs base model (5 samples each)
  python scripts/eval_adapters.py

  # Specific roles, 10 samples
  python scripts/eval_adapters.py --roles coder debugger --n-samples 10

  # Also compare against a previously-loaded named adapter in vLLM
  python scripts/eval_adapters.py --roles coder --prev-adapter coding_lora_v1

  # Save HTML report only
  python scripts/eval_adapters.py --output html

  # All options
  python scripts/eval_adapters.py \\
      --roles coder reviewer \\
      --n-samples 10 \\
      --prev-adapter coding_lora_v1 \\
      --deepseek-url http://localhost:8000 \\
      --mistral-url http://localhost:8001 \\
      --timeout 60 \\
      --output both \\
      --seed 42

Notes on "prev-adapter"
-----------------------
vLLM loads adapters at startup via --lora-modules. To compare against an older
checkpoint you must first load it into vLLM under a distinct name (e.g. by
restarting with an extra --lora-modules entry) and then pass that name via
--prev-adapter.  The eval tool itself does not modify vLLM's running config.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import re
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Optional rich import — degrades gracefully to plain text
# ---------------------------------------------------------------------------
try:
    from rich.console import Console as _RichConsole
    from rich.table import Table as _RichTable
    from rich import box as _rich_box
    from rich.panel import Panel as _RichPanel

    _HAVE_RICH = True
except ImportError:
    _HAVE_RICH = False

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "training_data_curated"
REPORTS_DIR = Path(__file__).parent / "eval_reports"

# ---------------------------------------------------------------------------
# Role / model configuration
# ---------------------------------------------------------------------------
# System prompts imported from the single source of truth so that evaluation
# uses the exact same strings as training and inference.
sys.path.insert(0, str(PROJECT_ROOT))
from shared.contracts.agent_prompts import AGENT_PROMPTS  # noqa: E402

ROLE_MODEL_FAMILY: dict[str, str] = {
    "planner":    "mistral",
    "researcher": "mistral",
    "coder":      "deepseek",
    "debugger":   "deepseek",
    "reviewer":   "deepseek",
    "critic":     "mistral",
    "consensus":  "mistral",
}

# Current adapter name per role (None = use base model in production).
# Mirrors orchestration/app/agents/prompts.py ROLE_ADAPTER.
# Broken adapters are kept here so the eval tool can still benchmark them
# explicitly (use --roles coder to re-test after retraining).
ROLE_ADAPTER: dict[str, str | None] = {
    "planner":    "planning_lora",
    "researcher": "research_lora",
    # coding_lora / debugging_lora retrained on curated coding_dataset.jsonl
    # and debugging_dataset.jsonl (clean code I/O pairs, no orchestration JSON).
    # Smoke test confirmed: code tokens present, zero metadata contamination.
    # Note: debugging_lora outputs prose-style root-cause + fix rather than the
    # strict ISSUE:/FIX: format (training data used code responses, not labels).
    "coder":      "coding_lora",
    "debugger":   "debugging_lora",
    # review_lora retrained on review_dataset.jsonl (8,861 curated samples).
    "reviewer":   "review_lora",
    # criticism_lora follows SCORE:/VERDICT:/BLOCKERS:/IMPROVEMENT: format.
    "critic":     "criticism_lora",
    "consensus":  None,
}

ROLE_DATASET: dict[str, str] = {
    "planner":    "planning_dataset.jsonl",
    "researcher": "research_dataset.jsonl",
    "coder":      "coding_dataset.jsonl",
    "debugger":   "debugging_dataset.jsonl",
    "reviewer":   "review_dataset.jsonl",
    "critic":     "criticism_dataset.jsonl",
    "consensus":  "coding_dataset.jsonl",  # fallback — no consensus dataset
}

ROLE_MAX_TOKENS: dict[str, int] = {
    "planner":    512,
    "researcher": 768,
    "coder":      1536,
    "debugger":   1024,
    "reviewer":   512,
    "critic":     384,
    "consensus":  1024,
}

# Fallback base-model names when vLLM /v1/models cannot be reached
BASE_MODEL_FALLBACKS: dict[str, str] = {
    "deepseek": "deepseek-ai/deepseek-coder-6.7b-instruct",
    "mistral":  "mistralai/Mistral-7B-Instruct-v0.1",
}

# Required structural output fields for evaluator roles
EVALUATOR_REQUIRED_FIELDS: dict[str, list[str]] = {
    "reviewer": ["SCORE", "STRENGTHS", "ISSUES", "SUGGESTIONS"],
    "critic":   ["SCORE", "VERDICT", "BLOCKERS", "IMPROVEMENT"],
}

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------
_SCORE_RE    = re.compile(r"\bSCORE\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
_CODE_RE     = re.compile(r"```|def |class |import |\bfunction\b|\breturn\b", re.I)
# Orchestration metadata leakage pattern — adapter outputs pipeline JSON instead of code
_METADATA_LEAKAGE_RE = re.compile(
    r"'agent_name'\s*:|\"agent_name\"\s*:|'iteration_number'\s*:|\"iteration_number\"\s*:"
    r"|'quality_score'\s*:|\"quality_score\"\s*:|'execution_time_ms'\s*:|\"execution_time_ms\"\s*:",
    re.I,
)
_STRUCT_RE   = re.compile(
    r"^\s*\d+[.)]\s+\S|^#{1,3}\s+\S|^[A-Z][A-Z ]{2,}:\s*\S",
    re.MULTILINE,
)
_BLOCKER_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bBLOCKERS?\s*[:=](?!\s*none)",
        r"\bCRITICAL\s+(?:ISSUE|BUG|ERROR)\b",
        r"\bFAILS?\s+(?:to|the|all)\b",
        r"\bDOES\s+NOT\s+WORK\b",
    ]
]
_POSITIVE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bLOOKS\s+GOOD\b",
        r"\bWELL\s+(?:STRUCTURED|WRITTEN|TESTED)\b",
        r"\bNO\s+ISSUES?\b",
        r"\bBLOCKERS?\s*[:=]\s*(?:None|none|N\/A)\b",
    ]
]

# ---------------------------------------------------------------------------
# Scoring utilities (standalone — no project import required)
# ---------------------------------------------------------------------------

def extract_score_from_text(text: str) -> float | None:
    """Return the first SCORE: annotation as a float in [0, 1], or None."""
    match = _SCORE_RE.search(text)
    if match:
        value = float(match.group(1))
        if 0.0 <= value <= 1.0:  # TODO: Cannot use magic numbers
            return value
        if 0 < value <= 10:  # TODO: Cannot use magic numbers
            return value / 10.0  # normalise 0-10 scale
    return None


def heuristic_score(text: str, role: str) -> float:
    """Estimate output quality via lightweight pattern matching."""
    if not text or len(text.strip()) < 30:   # TODO: Cannot use magic numbers
        return 0.30  # TODO: Cannot use magic numbers

    explicit = extract_score_from_text(text)
    if explicit is not None:
        return explicit

    # Detect adapter metadata leakage (adapter trained on orchestration JSON).
    if _METADATA_LEAKAGE_RE.search(text):
        return 0.10  # TODO: Cannot use magic numbers

    if role in ("reviewer", "critic"):
        score = 0.55  # absence of SCORE: is a red flag for these roles
    elif role in ("coder", "debugger"):
        has_code = bool(_CODE_RE.search(text))
        length = len(text.strip())
        score = 0.60 if not has_code else min(0.75, 0.65 + length / 12000)  # TODO: Cannot use magic numbers
    else:
        score = 0.62  # planner, researcher, consensus

    for pattern in _BLOCKER_PATTERNS:
        if pattern.search(text):
            score -= 0.12  # TODO: Cannot use magic numbers
    for pattern in _POSITIVE_PATTERNS:
        if pattern.search(text):
            score += 0.06  # TODO: Cannot use magic numbers 

    return max(0.0, min(1.0, score))     # TODO: Cannot use magic numbers


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_samples(role: str, n: int, seed: int = 42) -> list[dict]:
    """Load N instruction/response samples for a role from its training dataset."""
    dataset_file = DATA_DIR / ROLE_DATASET[role]
    if not dataset_file.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_file}\n"
            f"Expected one of: {list(ROLE_DATASET.values())}"
        )

    records: list[dict] = []
    with dataset_file.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "instruction" in obj and "response" in obj:
                # Skip instructions that are too long for a quick eval pass
                if len(obj["instruction"]) <= 2000:  # TODO: Cannot use magic numbers
                    records.append(obj)

    if not records:
        raise ValueError(f"No usable records found in {dataset_file}")

    rng = random.Random(seed)
    n = min(n, len(records))
    return rng.sample(records, n)


# ---------------------------------------------------------------------------
# vLLM API calls
# ---------------------------------------------------------------------------

def discover_base_model_name(url: str, family: str, timeout: int = 10) -> str:
    """
    Query vLLM's /v1/models to find the served base model ID.
    Falls back to the known HuggingFace path when the server isn't reachable.
    """
    try:
        resp = requests.get(f"{url}/v1/models", timeout=timeout)
        resp.raise_for_status()
        models = resp.json().get("data", [])
        # Base model IDs contain a '/' (org/name); adapter names don't.
        for m in models:
            mid: str = m.get("id", "")
            if "/" in mid:
                return mid
        # If no '/' found, fall back
        return BASE_MODEL_FALLBACKS[family]
    except Exception:
        return BASE_MODEL_FALLBACKS[family]


def call_vllm(
    url: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    timeout: int = 30,
) -> dict[str, Any]:
    """
    POST to vLLM /v1/chat/completions.

    Returns a dict with:
      output           – str  (empty on error)
      latency_ms       – float
      finish_reason    – str
      prompt_tokens    – int
      completion_tokens – int
      error            – str | None
    """
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,   # deterministic for reproducible benchmarking
        "stream": False,
    }
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            f"{url}/v1/chat/completions",
            json=payload,
            timeout=timeout,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        output = choice["message"]["content"] or ""
        usage = data.get("usage", {})
        return {
            "output": output,
            "latency_ms": latency_ms,
            "finish_reason": choice.get("finish_reason", ""),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "error": None,
        }
    except requests.exceptions.ConnectionError:
        latency_ms = (time.perf_counter() - t0) * 1000
        return _err_result(latency_ms, "Connection refused — is vLLM running?")
    except requests.exceptions.Timeout:
        latency_ms = (time.perf_counter() - t0) * 1000
        return _err_result(latency_ms, f"Request timed out after {timeout}s")
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        return _err_result(latency_ms, str(exc))


def _err_result(latency_ms: float, msg: str) -> dict[str, Any]:
    return {
        "output": "", "latency_ms": latency_ms, "finish_reason": "error",
        "prompt_tokens": 0, "completion_tokens": 0, "error": msg,
    }


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(output: str, role: str) -> dict[str, Any]:
    """Compute quality metrics for a single model output."""
    m: dict[str, Any] = {}
    m["output_length"] = len(output)
    m["heuristic_score"] = heuristic_score(output, role)
    m["explicit_score"] = extract_score_from_text(output)

    # -- Evaluator roles (reviewer, critic) --
    if role in EVALUATOR_REQUIRED_FIELDS:
        required = EVALUATOR_REQUIRED_FIELDS[role]
        m["has_score_field"] = bool(_SCORE_RE.search(output))
        present = [
            f for f in required
            if re.search(rf"\b{re.escape(f)}\s*[:=]", output, re.IGNORECASE)
        ]
        m["field_coverage"] = len(present) / len(required)
        m["required_fields_present"] = present
        m["required_fields_missing"] = [f for f in required if f not in present]

    # -- Code-generating roles (coder, debugger) --
    if role in ("coder", "debugger"):
        code_blocks = re.findall(r"```[\s\S]*?```", output)
        m["has_code"] = bool(_CODE_RE.search(output))
        m["code_block_count"] = len(code_blocks)
        # Try to compile Python blocks for syntax validation
        py_valid = py_total = 0
        for block in code_blocks:
            lang_m = re.match(r"```(\w*)", block)
            lang = lang_m.group(1).lower() if lang_m else ""
            if lang in ("", "python", "py"):
                py_total += 1
                code = re.sub(r"```\w*\n?", "", block).strip()
                try:
                    compile(code, "<string>", "exec")
                    py_valid += 1
                except SyntaxError:
                    pass
        m["python_syntax_valid_ratio"] = (py_valid / py_total) if py_total > 0 else None

    # -- Prose roles (planner, researcher, consensus) --
    if role in ("planner", "researcher", "consensus"):
        m["has_structure"] = bool(_STRUCT_RE.search(output))

    return m


# ---------------------------------------------------------------------------
# Per-role evaluation runner
# ---------------------------------------------------------------------------

def evaluate_role(
    role: str,
    samples: list[dict],
    adapter_name: str | None,
    base_model: str,
    vllm_url: str,
    prev_adapter_name: str | None,
    timeout: int,
) -> list[dict]:
    """
    Run all samples through adapter + base model (+ optional prev adapter).
    Returns a list of result dicts, one per sample.
    """
    system_prompt = AGENT_PROMPTS.get(role, "")
    max_tokens = ROLE_MAX_TOKENS.get(role, 512)
    results: list[dict] = []

    for i, sample in enumerate(samples):
        instruction = sample["instruction"].strip()
        reference = sample.get("response", "").strip()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": instruction},
        ]

        result: dict[str, Any] = {
            "role": role,
            "sample_id": i,
            "instruction_preview": (
                instruction[:120] + "…" if len(instruction) > 120 else instruction
            ),
            "instruction_full": instruction,
            "reference_preview": (
                reference[:120] + "…" if len(reference) > 120 else reference
            ),
            "reference_full": reference,
            "adapter_result": None,
            "base_result": None,
            "prev_adapter_result": None,
        }

        # --- Adapter inference ---
        if adapter_name:
            api_resp = call_vllm(vllm_url, adapter_name, messages, max_tokens, timeout)
            result["adapter_result"] = {
                "model": adapter_name,
                **api_resp,
                "metrics": compute_metrics(api_resp["output"], role)
                if not api_resp["error"] else {},
            }

        # --- Base-model inference ---
        base_resp = call_vllm(vllm_url, base_model, messages, max_tokens, timeout)
        result["base_result"] = {
            "model": base_model,
            **base_resp,
            "metrics": compute_metrics(base_resp["output"], role)
            if not base_resp["error"] else {},
        }

        # --- Previous adapter inference ---
        if prev_adapter_name:
            prev_resp = call_vllm(
                vllm_url, prev_adapter_name, messages, max_tokens, timeout
            )
            result["prev_adapter_result"] = {
                "model": prev_adapter_name,
                **prev_resp,
                "metrics": compute_metrics(prev_resp["output"], role)
                if not prev_resp["error"] else {},
            }

        results.append(result)
        _print_progress(role, i + 1, len(samples))

    return results


def _print_progress(role: str, current: int, total: int) -> None:
    pct = int(current / total * 100)
    bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
    print(f"\r  [{bar}] {current}/{total}", end="", flush=True)
    if current == total:
        print()


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_role_summary(role_results: list[dict]) -> dict[str, Any]:
    """Aggregate per-sample metrics into a role-level summary dict."""

    def _avg(values: list) -> float | None:
        clean = [v for v in values if v is not None]
        return sum(clean) / len(clean) if clean else None

    def _pct(values: list[bool]) -> float:
        return sum(values) / len(values) if values else 0.0

    role = role_results[0]["role"] if role_results else "unknown"
    summary: dict[str, Any] = {"role": role, "n_samples": len(role_results)}

    for variant_key in ("adapter_result", "base_result", "prev_adapter_result"):
        sub = [r[variant_key] for r in role_results if r[variant_key] is not None]
        if not sub:
            summary[variant_key] = None
            continue

        # ms_per_token: latency / completion_tokens — normalized inference
        # speed unaffected by output length differences between variants.
        ms_per_tok_vals = [
            r["latency_ms"] / r["completion_tokens"]
            for r in sub
            if not r["error"] and r.get("completion_tokens", 0) > 0
        ]
        s: dict[str, Any] = {
            "model": sub[0]["model"],
            "avg_latency_ms": _avg([r["latency_ms"] for r in sub]),
            "avg_ms_per_token": _avg(ms_per_tok_vals),
            "avg_completion_tokens": _avg(
                [r["completion_tokens"] for r in sub if not r["error"]]
            ),
            "avg_heuristic_score": _avg(
                [r["metrics"].get("heuristic_score") for r in sub]
            ),
            "avg_output_length": _avg(
                [r["metrics"].get("output_length") for r in sub]
            ),
            "error_rate": _pct([bool(r["error"]) for r in sub]),
        }

        if role in EVALUATOR_REQUIRED_FIELDS:
            s["avg_explicit_score"] = _avg(
                [r["metrics"].get("explicit_score") for r in sub]
            )
            s["field_coverage_rate"] = _avg(
                [r["metrics"].get("field_coverage") for r in sub]
            )
            s["has_score_field_rate"] = _pct(
                [bool(r["metrics"].get("has_score_field")) for r in sub]
            )

        if role in ("coder", "debugger"):
            s["code_presence_rate"] = _pct(
                [bool(r["metrics"].get("has_code")) for r in sub]
            )
            s["avg_code_block_count"] = _avg(
                [r["metrics"].get("code_block_count") for r in sub]
            )
            syntax_vals = [
                r["metrics"].get("python_syntax_valid_ratio")
                for r in sub
                if r["metrics"].get("python_syntax_valid_ratio") is not None
            ]
            s["avg_syntax_valid_ratio"] = _avg(syntax_vals) if syntax_vals else None

        if role in ("planner", "researcher", "consensus"):
            s["structure_rate"] = _pct(
                [bool(r["metrics"].get("has_structure")) for r in sub]
            )

        summary[variant_key] = s

    return summary


# ---------------------------------------------------------------------------
# Terminal rendering
# ---------------------------------------------------------------------------

def render_terminal(all_summaries: list[dict], all_results: list[list[dict]]) -> None:
    if _HAVE_RICH:
        _render_rich(all_summaries)
    else:
        _render_plain(all_summaries)


def _render_rich(summaries: list[dict]) -> None:
    console = _RichConsole()
    console.print()
    console.print(
        _RichPanel.fit(
            f"[bold cyan]Adapter Evaluation Report[/bold cyan]\n"
            f"[dim]{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]",
            border_style="cyan",
        )
    )

    table = _RichTable(
        title="Adapter vs Base Model — Summary",
        box=_rich_box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Role", style="cyan", no_wrap=True)
    table.add_column("Variant")
    table.add_column("Model")
    table.add_column("Heuristic Score", justify="right")
    table.add_column("ms/tok", justify="right")
    table.add_column("Avg Tokens", justify="right")
    table.add_column("Total ms", justify="right")
    table.add_column("Extra Metrics")
    table.add_column("Errors", justify="right")

    VARIANT_LABELS = [
        ("adapter_result",      "Adapter"),
        ("base_result",         "Base"),
        ("prev_adapter_result", "Prev"),
    ]

    for s in summaries:
        role = s["role"]
        first_variant = True
        for variant_key, label in VARIANT_LABELS:
            v = s.get(variant_key)
            if not v:
                continue

            score = v.get("avg_heuristic_score")
            if score is None:
                score_txt = "[dim]N/A[/dim]"
            elif score >= 0.75:
                score_txt = f"[green]{score:.3f}[/green]"
            elif score >= 0.55:
                score_txt = f"[yellow]{score:.3f}[/yellow]"
            else:
                score_txt = f"[red]{score:.3f}[/red]"

            extras: list[str] = []
            if "field_coverage_rate" in v:
                extras.append(f"Format {v['field_coverage_rate']:.0%}")
            if "has_score_field_rate" in v:
                extras.append(f"SCORE✓ {v['has_score_field_rate']:.0%}")
            if "code_presence_rate" in v:
                extras.append(f"Code {v['code_presence_rate']:.0%}")
            if "avg_syntax_valid_ratio" in v and v["avg_syntax_valid_ratio"] is not None:
                extras.append(f"Syntax {v['avg_syntax_valid_ratio']:.0%}")
            if "structure_rate" in v:
                extras.append(f"Struct {v['structure_rate']:.0%}")

            err_rate = v["error_rate"]
            err_txt = (
                f"[red]{err_rate:.0%}[/red]" if err_rate > 0 else "[green]0%[/green]"
            )

            ms_tok = v.get("avg_ms_per_token")
            ms_tok_txt = f"{ms_tok:.1f}" if ms_tok is not None else "N/A"
            avg_toks = v.get("avg_completion_tokens")
            avg_toks_txt = f"{avg_toks:.0f}" if avg_toks is not None else "N/A"
            short_model = v["model"].split("/")[-1]
            table.add_row(
                role if first_variant else "",
                f"[bold]{label}[/bold]",
                short_model,
                score_txt,
                ms_tok_txt,
                avg_toks_txt,
                f"{v['avg_latency_ms']:.0f}" if v.get("avg_latency_ms") is not None else "N/A",
                " | ".join(extras) or "—",
                err_txt,
            )
            first_variant = False

        table.add_section()

    console.print(table)
    console.print()


def _render_plain(summaries: list[dict]) -> None:
    print()
    print("=" * 72)
    print("ADAPTER EVALUATION REPORT")
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 72)

    VARIANT_LABELS = [
        ("adapter_result",      "ADAPTER"),
        ("base_result",         "BASE"),
        ("prev_adapter_result", "PREV ADAPTER"),
    ]

    for s in summaries:
        role = s["role"]
        print(f"\n{'─' * 60}")
        print(f"ROLE: {role.upper()}  (n={s['n_samples']})")
        print(f"{'─' * 60}")
        for variant_key, label in VARIANT_LABELS:
            v = s.get(variant_key)
            if not v:
                continue
            print(f"  [{label}] {v['model']}")
            score = v.get("avg_heuristic_score")
            print(
                f"    Heuristic Score  : {score:.3f}" if score is not None
                else "    Heuristic Score  : N/A"
            )
            ms_tok = v.get("avg_ms_per_token")
            print(
                f"    ms / token       : {ms_tok:.2f}" if ms_tok is not None
                else "    ms / token       : N/A"
            )
            avg_toks = v.get("avg_completion_tokens")
            print(
                f"    Avg Tokens       : {avg_toks:.0f}" if avg_toks is not None
                else "    Avg Tokens       : N/A"
            )
            lat = v.get("avg_latency_ms")
            print(
                f"    Total Latency    : {lat:.0f} ms" if lat is not None
                else "    Total Latency    : N/A"
            )
            olen = v.get("avg_output_length")
            print(
                f"    Avg Output Len   : {olen:.0f} chars" if olen is not None
                else "    Avg Output Len   : N/A"
            )
            if "field_coverage_rate" in v:
                print(f"    Format Coverage  : {v['field_coverage_rate']:.1%}")
            if "has_score_field_rate" in v:
                print(f"    SCORE field rate : {v['has_score_field_rate']:.1%}")
            if "code_presence_rate" in v:
                print(f"    Code Presence    : {v['code_presence_rate']:.1%}")
            if (
                "avg_syntax_valid_ratio" in v
                and v["avg_syntax_valid_ratio"] is not None
            ):
                print(f"    Syntax Valid     : {v['avg_syntax_valid_ratio']:.1%}")
            if "structure_rate" in v:
                print(f"    Structure Rate   : {v['structure_rate']:.1%}")
            if v["error_rate"] > 0:
                print(f"    *** Error Rate   : {v['error_rate']:.1%} ***")

    print()


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Adapter Evaluation Report &mdash; {timestamp}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    line-height: 1.5; color: #1a1a2e; background: #f0f2f5;
  }}
  .page-header {{
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    color: white; padding: 2rem 2.5rem;
  }}
  .page-header h1 {{ font-size: 1.75rem; margin-bottom: 0.25rem; }}
  .page-header p  {{ opacity: 0.65; font-size: 0.9rem; }}
  .container {{ max-width: 1440px; margin: 2rem auto; padding: 0 1.5rem; }}
  h2 {{
    color: #1a1a2e; border-bottom: 3px solid #3498db;
    padding-bottom: 0.4rem; margin: 2rem 0 1rem;
  }}
  /* ── Summary table ────────────────────────────────────────────── */
  .summary-wrap {{
    background: white; border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,.08); overflow: auto;
    margin-bottom: 2rem;
  }}
  table {{
    width: 100%; border-collapse: collapse;
  }}
  th {{
    background: #2c3e50; color: white;
    padding: 0.75rem 1rem; text-align: left; white-space: nowrap;
  }}
  td {{ padding: 0.65rem 1rem; border-bottom: 1px solid #ecf0f1; }}
  tr:last-child td {{ border-bottom: none; }}
  tr.variant-adapter {{ background: #eaf4fd; }}
  tr.variant-base    {{ background: #edfdf4; }}
  tr.variant-prev    {{ background: #fdf6e8; }}
  .score-good {{ color: #27ae60; font-weight: 700; }}
  .score-ok   {{ color: #e67e22; font-weight: 700; }}
  .score-poor {{ color: #e74c3c; font-weight: 700; }}
  /* ── Samples ──────────────────────────────────────────────────── */
  .role-section {{
    background: white; border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,.08);
    padding: 1.5rem; margin-bottom: 1.5rem;
  }}
  .role-section h3 {{ font-size: 1.15rem; margin-bottom: 1rem; color: #16213e; }}
  details.sample {{
    border: 1px solid #dde3f0; border-radius: 6px;
    margin-bottom: 0.6rem; overflow: hidden;
  }}
  details.sample summary {{
    padding: 0.65rem 1rem; cursor: pointer;
    background: #f8f9fc; font-weight: 500; list-style: none;
    display: flex; gap: 0.5rem; align-items: center;
  }}
  details.sample summary::-webkit-details-marker {{ display: none; }}
  details.sample summary::before {{ content: "▶"; font-size: 0.7em; }}
  details.sample[open] summary::before {{ content: "▼"; }}
  details.sample summary:hover {{ background: #edf0f8; }}
  .sample-body {{ padding: 1rem; display: grid; gap: 1rem; }}
  .instr-block {{
    background: #f8f9fc; border-left: 4px solid #8e9cbb;
    padding: 0.75rem; border-radius: 0 4px 4px 0;
  }}
  .instr-block .label {{ font-size: 0.75rem; font-weight: 700; text-transform: uppercase;
                         letter-spacing: .05em; color: #8e9cbb; margin-bottom: 0.25rem; }}
  .variant-card {{
    border-radius: 6px; padding: 0.85rem; border-left-width: 4px;
    border-left-style: solid;
  }}
  .variant-card.adapter {{ border-left-color: #3498db; background: #eaf5fd; }}
  .variant-card.base    {{ border-left-color: #2ecc71; background: #edfcf4; }}
  .variant-card.prev    {{ border-left-color: #e67e22; background: #fdf6e8; }}
  .variant-label {{ font-weight: 700; font-size: 0.9rem; margin-bottom: 0.3rem; }}
  .metrics-bar {{
    font-size: 0.78rem; color: #5d6b8a; margin-bottom: 0.5rem;
    display: flex; flex-wrap: wrap; gap: 0.75rem;
  }}
  .metrics-bar span {{ white-space: nowrap; }}
  pre.out {{
    margin: 0; white-space: pre-wrap; word-break: break-word;
    font-size: 0.82rem; background: #1e2030; color: #cdd6f4;
    padding: 0.85rem; border-radius: 4px;
    max-height: 380px; overflow-y: auto;
  }}
  .error-msg {{
    background: #fdf0f0; border: 1px solid #e74c3c;
    border-radius: 4px; padding: 0.65rem 1rem;
    color: #c0392b; font-size: 0.88rem;
  }}
</style>
</head>
<body>
<div class="page-header">
  <h1>&#x1F50E; Adapter Evaluation Report</h1>
  <p>Generated: {timestamp} &nbsp;&bull;&nbsp; {roles_tag}</p>
</div>
<div class="container">

  <h2>Comparison Summary</h2>
  <div class="summary-wrap">
    <table>
      <thead>
        <tr>
          <th>Role</th><th>Variant</th><th>Model</th>
          <th>Heuristic Score</th><th>ms/tok</th><th>Avg Tokens</th>
          <th>Total ms</th><th>Extra Metrics</th><th>Error Rate</th>
        </tr>
      </thead>
      <tbody>
        {summary_rows}
      </tbody>
    </table>
  </div>

  <h2>Sample Details</h2>
  {detail_sections}
</div>
</body>
</html>
"""


def _he(text: str) -> str:
    """HTML-escape a string."""
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
    )


def render_html(
    all_summaries: list[dict],
    all_results: list[list[dict]],
    output_path: Path,
) -> None:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    roles_tag = ", ".join(s["role"] for s in all_summaries)

    # ── Summary table rows ──────────────────────────────────────────────────
    summary_rows = ""
    VARIANT_META = [
        ("adapter_result",      "Adapter",      "variant-adapter"),
        ("base_result",         "Base Model",   "variant-base"),
        ("prev_adapter_result", "Prev Adapter", "variant-prev"),
    ]

    for s in all_summaries:
        role = s["role"]
        for variant_key, label, row_cls in VARIANT_META:
            v = s.get(variant_key)
            if not v:
                continue

            score = v.get("avg_heuristic_score")
            if score is None:
                score_html = "N/A"
            elif score >= 0.75:
                score_html = f'<span class="score-good">{score:.3f}</span>'
            elif score >= 0.55:
                score_html = f'<span class="score-ok">{score:.3f}</span>'
            else:
                score_html = f'<span class="score-poor">{score:.3f}</span>'

            extras: list[str] = []
            if "field_coverage_rate" in v:
                extras.append(f"Format {v['field_coverage_rate']:.0%}")
            if "has_score_field_rate" in v:
                extras.append(f"SCORE✓ {v['has_score_field_rate']:.0%}")
            if "code_presence_rate" in v:
                extras.append(f"Code {v['code_presence_rate']:.0%}")
            if (
                "avg_syntax_valid_ratio" in v
                and v["avg_syntax_valid_ratio"] is not None
            ):
                extras.append(f"Syntax {v['avg_syntax_valid_ratio']:.0%}")
            if "structure_rate" in v:
                extras.append(f"Structure {v['structure_rate']:.0%}")

            err = v["error_rate"]
            err_html = (
                f'<span style="color:#e74c3c">{err:.0%}</span>' if err > 0
                else '<span style="color:#2ecc71">0%</span>'
            )

            ms_tok = v.get("avg_ms_per_token")
            ms_tok_html = f"{ms_tok:.1f}" if ms_tok is not None else "N/A"
            avg_toks = v.get("avg_completion_tokens")
            avg_toks_html = f"{avg_toks:.0f}" if avg_toks is not None else "N/A"
            summary_rows += (
                f'<tr class="{row_cls}">'
                f"<td>{_he(role)}</td>"
                f"<td><strong>{_he(label)}</strong></td>"
                f"<td>{_he(v['model'].split('/')[-1])}</td>"
                f"<td>{score_html}</td>"
                f"<td>{ms_tok_html}</td>"
                f"<td>{avg_toks_html}</td>"
                f"<td>{v['avg_latency_ms']:.0f}</td>"
                f"<td>{'<br>'.join(_he(e) for e in extras) or '—'}</td>"
                f"<td>{err_html}</td>"
                f"</tr>\n"
            )

    # ── Detail sections ─────────────────────────────────────────────────────
    detail_sections = ""
    CARD_META = [
        ("adapter_result",      "Adapter",     "adapter"),
        ("base_result",         "Base Model",  "base"),
        ("prev_adapter_result", "Prev Adapter","prev"),
    ]

    for role_results in all_results:
        role = role_results[0]["role"] if role_results else "?"
        sample_blocks = ""

        for result in role_results:
            variants_html = ""
            for variant_key, label, card_cls in CARD_META:
                v = result.get(variant_key)
                if not v:
                    continue

                output_text = v.get("output", "") or ""
                error_text = v.get("error") or ""
                mx = v.get("metrics", {})
                score = mx.get("heuristic_score")
                latency = v.get("latency_ms", 0)

                completion_tokens = v.get("completion_tokens", 0)
                ms_per_tok = (
                    latency / completion_tokens
                    if completion_tokens > 0 else None
                )
                metrics_parts: list[str] = []
                metrics_parts.append(
                    f"Score: {score:.3f}" if isinstance(score, float) else "Score: N/A"
                )
                metrics_parts.append(
                    f"ms/tok: {ms_per_tok:.1f}" if ms_per_tok is not None else "ms/tok: N/A"
                )
                metrics_parts.append(f"tokens: {completion_tokens}")
                metrics_parts.append(f"total: {latency:.0f}ms")
                metrics_parts.append(
                    f"Length: {mx.get('output_length', 'N/A')} chars"
                )
                if "field_coverage" in mx:
                    metrics_parts.append(f"Format: {mx['field_coverage']:.0%}")
                if "has_code" in mx:
                    metrics_parts.append(
                        "Code: ✓" if mx.get("has_code") else "Code: ✗"
                    )
                if (
                    "python_syntax_valid_ratio" in mx
                    and mx["python_syntax_valid_ratio"] is not None
                ):
                    metrics_parts.append(
                        f"Syntax: {mx['python_syntax_valid_ratio']:.0%}"
                    )
                if "has_structure" in mx:
                    metrics_parts.append(
                        "Struct: ✓" if mx.get("has_structure") else "Struct: ✗"
                    )

                metrics_html = "".join(
                    f"<span>{_he(p)}</span>" for p in metrics_parts
                )

                if error_text:
                    body_html = f'<div class="error-msg">⚠ {_he(error_text)}</div>'
                else:
                    truncated = output_text[:2000]
                    suffix = "…[truncated]" if len(output_text) > 2000 else ""
                    body_html = (
                        f"<pre class=\"out\">{_he(truncated)}{suffix}</pre>"
                    )

                variants_html += (
                    f'<div class="variant-card {card_cls}">'
                    f'<div class="variant-label">'
                    f"{_he(label)}: <code>{_he(v['model'].split('/')[-1])}</code>"
                    f"</div>"
                    f'<div class="metrics-bar">{metrics_html}</div>'
                    f"{body_html}"
                    f"</div>\n"
                )

            instr_html = _he(result["instruction_full"][:800]) + (
                "…" if len(result["instruction_full"]) > 800 else ""
            )
            instr_preview_html = _he(result["instruction_preview"])

            sample_blocks += (
                f"<details class=\"sample\">"
                f"<summary>Sample {result['sample_id'] + 1}: {instr_preview_html}</summary>"
                f"<div class=\"sample-body\">"
                f'<div class="instr-block">'
                f'<div class="label">Instruction</div>'
                f"<pre class=\"out\">{instr_html}</pre>"
                f"</div>"
                f"{variants_html}"
                f"</div>"
                f"</details>\n"
            )

        detail_sections += (
            f'<div class="role-section">'
            f"<h3>Role: {_he(role)}</h3>"
            f"{sample_blocks}"
            f"</div>\n"
        )

    html = _HTML_TEMPLATE.format(
        timestamp=timestamp,
        roles_tag=roles_tag,
        summary_rows=summary_rows,
        detail_sections=detail_sections,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"\nHTML report → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
ALL_ROLES = list(ROLE_ADAPTER.keys())
ROLES_WITH_ADAPTERS = [r for r, a in ROLE_ADAPTER.items() if a is not None]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate LoRA adapter vs base model for all agent roles.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
examples:
  python scripts/eval_adapters.py
  python scripts/eval_adapters.py --roles coder debugger --n-samples 10
  python scripts/eval_adapters.py --roles all --output html
  python scripts/eval_adapters.py --prev-adapter coding_lora_v1
        """),
    )
    parser.add_argument(
        "--roles",
        nargs="+",
        metavar="ROLE",
        help=(
            f"Roles to evaluate. Use 'all' for every role. "
            f"Default: roles that have adapters configured "
            f"({', '.join(ROLES_WITH_ADAPTERS)})."
        ),
    )
    parser.add_argument(
        "--n-samples", type=int, default=5, metavar="N",
        help="Samples per role (default: 5)",
    )
    parser.add_argument(
        "--prev-adapter", metavar="ADAPTER_NAME",
        help="Name of a previous adapter already loaded in vLLM to compare against",
    )
    parser.add_argument(
        "--deepseek-url", default="http://localhost:8000", metavar="URL",
        help="vLLM DeepSeek endpoint (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--mistral-url", default="http://localhost:8001", metavar="URL",
        help="vLLM Mistral endpoint (default: http://localhost:8001)",
    )
    parser.add_argument(
        "--timeout", type=int, default=30, metavar="SECS",
        help="Per-request timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--output",
        choices=["terminal", "html", "both"],
        default="both",
        help="Output format (default: both)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, metavar="N",
        help="Random seed for sample selection (default: 42)",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    # ── Resolve roles ────────────────────────────────────────────────────────
    if not args.roles:
        roles = ROLES_WITH_ADAPTERS
    elif args.roles == ["all"]:
        roles = ALL_ROLES
    else:
        invalid = [r for r in args.roles if r not in ALL_ROLES]
        if invalid:
            print(
                f"Error: unknown role(s): {invalid}\n"
                f"Valid: {ALL_ROLES}",
                file=sys.stderr,
            )
            sys.exit(1)
        roles = args.roles

    vllm_urls = {
        "deepseek": args.deepseek_url,
        "mistral":  args.mistral_url,
    }

    print()
    print("Adapter Evaluation Tool")
    print("─" * 50)
    print(f"Roles       : {', '.join(roles)}")
    print(f"Samples     : {args.n_samples} per role")
    print(f"Output      : {args.output}")
    print(f"Seed        : {args.seed}")
    if args.prev_adapter:
        print(f"Prev adapter: {args.prev_adapter}")
    print()

    # ── Discover base model names ────────────────────────────────────────────
    print("Discovering base model names from vLLM servers…")
    base_models: dict[str, str] = {}
    for family, url in vllm_urls.items():
        name = discover_base_model_name(url, family, timeout=10)
        base_models[family] = name
        print(f"  {family:8s}: {name}")
    print()

    # ── Evaluate each role ───────────────────────────────────────────────────
    all_summaries: list[dict] = []
    all_results: list[list[dict]] = []

    for role in roles:
        family      = ROLE_MODEL_FAMILY[role]
        url         = vllm_urls[family]
        adapter     = ROLE_ADAPTER[role]
        base_model  = base_models[family]

        print(
            f"Evaluating  : {role}  "
            f"[model={family}, adapter={adapter or 'none'}]"
        )

        try:
            samples = load_samples(role, args.n_samples, seed=args.seed)
        except (FileNotFoundError, ValueError) as exc:
            print(f"  ⚠  Skipping {role}: {exc}")
            continue

        role_results = evaluate_role(
            role=role,
            samples=samples,
            adapter_name=adapter,
            base_model=base_model,
            vllm_url=url,
            prev_adapter_name=args.prev_adapter,
            timeout=args.timeout,
        )

        all_summaries.append(aggregate_role_summary(role_results))
        all_results.append(role_results)

    if not all_summaries:
        print("No results collected — nothing to report.", file=sys.stderr)
        sys.exit(1)

    print()

    # ── Output ───────────────────────────────────────────────────────────────
    if args.output in ("terminal", "both"):
        render_terminal(all_summaries, all_results)

    if args.output in ("html", "both"):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        roles_tag = "_".join(roles) if len(roles) <= 4 else f"{len(roles)}roles"
        report_path = REPORTS_DIR / f"eval_{roles_tag}_{ts}.html"
        render_html(all_summaries, all_results, report_path)


if __name__ == "__main__":
    main()
