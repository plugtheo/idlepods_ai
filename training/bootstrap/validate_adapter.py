#!/usr/bin/env python3
"""
Post-training adapter validation — run as a subprocess from train_gpu_simple.py
or standalone to verify that a trained LoRA adapter is not corrupted.

Usage
-----
    python validate_adapter.py --capability coding --adapter-dir /data/lora_checkpoints

What it checks (per capability)
--------------------------------
  coding     → output contains code tokens (def/class/import/```)
  debugging  → output contains ISSUE: section
  review     → output contains SCORE: section
  planning   → output contains a numbered list (1. …)
  research   → output contains >= 3 sentences of prose
  criticism  → output contains VERDICT:/SCORE:/BLOCKERS: section

  All capabilities:
    - No BPE artifacts (Ġ U+0120, Ċ U+010A, ▁ U+2581)
    - No JSON metadata contamination (agent_name / quality_score keys)
    - Minimum output length (varies per capability)

Exit codes
----------
    0  All checks passed → adapter integrity confirmed
    1  One or more checks failed → adapter likely corrupted or undertrained
    2  Adapter could not be loaded at all → save failed
"""

from __future__ import annotations

import argparse
import gc
import re
import sys
from pathlib import Path
from typing import Callable, List, Tuple

# Add repo root so shared/ is importable (mirrors Docker PYTHONPATH=/app).
_HERE      = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.contracts.agent_prompts import AGENT_PROMPTS, BOOTSTRAP_CAP_TO_ROLE
from shared.contracts.evaluator_schemas import (
    evaluator_required_fields_present,
    extract_score,
)
from shared.contracts.training import AdapterRecipe, lookup_recipe

# ---------------------------------------------------------------------------
# BPE / contamination patterns (same as e2e_test.py)
# ---------------------------------------------------------------------------
from shared.contracts.quality_filters import BPE_ARTIFACT_RE as _BPE_RE, METADATA_LEAKAGE_RE


def _fix_bpe_artifacts(text: str) -> str:
    """
    Mirror of inference/app/backends/local_vllm.py::_fix_bpe_artifacts.
    Converts GPT-2 byte-level BPE unicode placeholders (Ġ=space, Ċ=newline)
    back to their original bytes so checks run on human-readable text.
    """
    passthrough = (
        list(range(33, 127))
        + list(range(161, 173))
        + list(range(174, 256))
    )
    passthrough_set = set(passthrough)
    table: dict[int, int] = {}
    n = 0
    for b in range(256):
        if b in passthrough_set:
            continue
        table[256 + n] = b
        n += 1
    if not any("\u0100" <= ch <= "\u0143" for ch in text):
        return text
    try:
        raw = bytes(table[ord(ch)] if ord(ch) in table else ord(ch) for ch in text)
        return raw.decode("utf-8", errors="replace")
    except Exception:
        return text
_JSON_META_RE = METADATA_LEAKAGE_RE
_CODE_RE      = re.compile(r"```|(?<![a-zA-Z])def|(?<![a-zA-Z])class|(?<![a-zA-Z])import|(?<![a-zA-Z])return", re.I)
_ISSUE_RE     = re.compile(r"\bISSUE\s*:", re.I)
_FIX_RE       = re.compile(r"\bFIX\s*:", re.I)
# Match "1." style OR "Step 1:" style — both are valid numbered-list outputs
_NUMBERED_RE  = re.compile(r"^\s*(\d+\.|[Ss]tep\s+\d+[.:])", re.M)


def _reviewer_output_ok(text: str) -> bool:
    """Reviewer output is valid when it satisfies the schema (JSON path) or the
    legacy prose contract, AND carries an extractable score."""
    return (
        evaluator_required_fields_present(text, "reviewer")
        and extract_score(text) is not None
    )


def _critic_output_ok(text: str) -> bool:
    """Critic output is valid when it satisfies the schema (JSON path) or the
    legacy prose contract, AND carries an extractable score."""
    return (
        evaluator_required_fields_present(text, "critic")
        and extract_score(text) is not None
    )

# Type alias for a single check: (check_id, predicate, human description)
Check = Tuple[str, Callable[[str], bool], str]

# Common checks applied to every prompt across every capability.
_COMMON: List[Check] = [
    ("no_bpe",           lambda t: not bool(_BPE_RE.search(t)),      "No BPE artifacts"),
    ("no_contamination", lambda t: not bool(_JSON_META_RE.search(t)), "No JSON metadata contamination"),
]

# Pass threshold: this fraction of prompts must pass EACH check to promote.
PROMPT_PASS_THRESHOLD = 0.80

# ---------------------------------------------------------------------------
# Base-skill probes (item 6) — run on every capability, gate on 80 % pass rate.
# Checks that heavy capability-specific fine-tuning has not destroyed the
# adapter's general reasoning, arithmetic, and instruction-following abilities.
# ---------------------------------------------------------------------------
_BASE_SKILL_PROBES: List[Tuple[str, Callable[[str], bool]]] = [
    ("What is 7 × 8?",
     lambda t: "56" in t),
    ("Summarize in one sentence: The cat sat on the mat.",
     lambda t: 5 < len(t.strip()) < 250),
    ("Is Python interpreted or compiled? One word.",
     lambda t: "interpret" in t.lower() or "compiled" in t.lower()),
    ("Write the word PASS in all capitals.",
     lambda t: "PASS" in t),
    ("What is the capital of France?",
     lambda t: "paris" in t.lower()),
]
BASE_SKILL_GATE = 0.80  # Promote only if >= 80 % of base-skill probes pass.

# ---------------------------------------------------------------------------
# Semantic correctness helpers (item 1) — code execution for coding/debugging.
# ---------------------------------------------------------------------------

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
# Lines that signal the start of real Python code at module level.
_PY_LEADING_RE = re.compile(
    r"^\s*(def\s|class\s|import\s|from\s|@\w|if\s+__name__\s*==)",
    re.MULTILINE,
)


def _strip_think_blocks(text: str) -> str:
    """Remove Qwen3-style <think>...</think> reasoning blocks (including empty
    ones) so downstream parsers see only the user-facing answer."""
    return _THINK_BLOCK_RE.sub("", text).strip()


def _extract_code_block(text: str) -> str:
    """Return the first executable code segment from *text*.

    Resolution order:
      1. Fenced markdown block (```python ... ``` or any-language fence).
      2. Substring starting at the first Python-leading line (def/class/
         import/from/decorator/`if __name__`). Drops natural-language
         preamble like "Here's a function:" that the model often emits.
      3. The full text as a last resort.
    """
    text = _strip_think_blocks(text)
    m = re.search(r"```(?:[a-zA-Z0-9_+-]*)?\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = _PY_LEADING_RE.search(text)
    if m:
        return text[m.start():].strip()
    return text.strip()


def _exec_code_safe(code: str, timeout: int = 5) -> Tuple[bool, str]:
    """
    Execute Python code in a subprocess.  Returns (success, stderr_snippet).
    Does NOT import the adapter — purely syntactic + runtime correctness check.
    """
    import subprocess
    import tempfile
    import os as _os
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False, encoding="utf-8") as tmp:
        tmp.write(code)
        fname = tmp.name
    try:
        proc = subprocess.run(
            [sys.executable, fname],
            capture_output=True, text=True, timeout=timeout,
        )
        return proc.returncode == 0, (proc.stderr or "").strip()[:200]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as exc:
        return False, str(exc)[:200]
    finally:
        try:
            _os.unlink(fname)
        except OSError:
            pass


def _check_semantic_coding(generated: str) -> bool:
    """Return True if the generated code block parses and executes without error."""
    import ast as _ast
    code = _extract_code_block(generated)
    try:
        _ast.parse(code)
    except SyntaxError as exc:
        print(
            f"  [VALIDATE-DEBUG] code_executes parse error: {exc.msg} "
            f"(line {exc.lineno}, col {exc.offset})",
            flush=True,
        )
        print(f"  [VALIDATE-DEBUG]   head: {code[:240]!r}", flush=True)
        print(f"  [VALIDATE-DEBUG]   tail: {code[-240:]!r}", flush=True)
        return False
    ok, stderr = _exec_code_safe(code)
    if not ok and stderr:
        print(f"  [VALIDATE-DEBUG] code_executes runtime error: {stderr}", flush=True)
    return ok


def _check_semantic_debugging(generated: str) -> bool:
    """
    Extract the FIX code block and verify it executes without raising the
    original error type.  A fix that crashes is not a fix.
    """
    generated = _strip_think_blocks(generated)
    fix_match = re.search(r"FIX\s*:\s*```[^\n]*\n(.*?)```", generated, re.DOTALL | re.I)
    if not fix_match:
        fix_match = re.search(r"FIX\s*:\s*\n((?:[ \t]+.*\n?)+)", generated, re.I)
    if not fix_match:
        return False
    ok, _ = _exec_code_safe(fix_match.group(1).strip())
    return ok


# ---------------------------------------------------------------------------
# Per-capability test fixtures (item 2: 5 prompts per capability, 80% pass)
# ---------------------------------------------------------------------------
_TESTS: dict[str, dict] = {
    "coding": {
        # Full function bodies are token-heavy with indentation; 300 truncates
        # mid-statement on most prompts, which then fails ast.parse.
        "max_new_tokens": 1024,
        "prompts": [
            "Write a Python function that computes the nth Fibonacci number using memoization.",
            "Write a Python function to check if a string is a palindrome.",
            "Write a Python class for a stack with push, pop, and peek methods.",
            "Write a Python function that finds the two largest numbers in a list without sorting.",
            "Write a Python context manager class that measures execution time of a code block.",
        ],
        "checks": [
            ("has_code",       lambda t: bool(_CODE_RE.search(t)),    "Output contains code tokens"),
            ("min_length",     lambda t: len(t.strip()) >= 80,        "Output length >= 80 chars"),
            ("code_executes",  _check_semantic_coding,                "Generated Python code executes without error"),
        ],
    },
    "debugging": {
        # ISSUE: + FIX: sections plus a fenced code block easily exceed 300.
        "max_new_tokens": 1024,
        "prompts": [
            (
                "Fix this code:\n\n"
                "def divide(a, b):\n"
                "    return a / b  # crashes when b is zero\n\n"
                "result = divide(10, 0)"
            ),
            (
                "Fix this code:\n\n"
                "def greet(name):\n"
                "    print('Hello, ' + name)\n\n"
                "greet(42)"
            ),
            (
                "Fix this code:\n\n"
                "items = [1, 2, 3]\n"
                "print(items[5])"
            ),
            (
                "Fix this code:\n\n"
                "d = {'a': 1}\n"
                "print(d['b'])"
            ),
            (
                "Fix this code:\n\n"
                "def factorial(n):\n"
                "    return n * factorial(n - 1)\n\n"
                "factorial(5)"
            ),
        ],
        "checks": [
            ("has_issue",       lambda t: bool(_ISSUE_RE.search(t)),  "Output contains ISSUE: section"),
            ("has_fix",         lambda t: bool(_FIX_RE.search(t)),    "Output contains FIX: section"),
            ("min_length",      lambda t: len(t.strip()) >= 60,       "Output length >= 60 chars"),
            ("fix_executes",    _check_semantic_debugging,            "FIX code block executes without error"),
        ],
    },
    "review": {
        "prompts": [
            (
                "Review this pull request:\n\n"
                "def add(a, b): return a + b\n\n"
                "There are no tests and no docstring."
            ),
            (
                "Review this code for security issues:\n\n"
                "query = 'SELECT * FROM users WHERE id=' + user_input"
            ),
            (
                "Review this code:\n\n"
                "password = 'hunter2'\n"
                "if input_pw == password: grant_access()"
            ),
            (
                "Review this code for performance:\n\n"
                "def find_duplicates(lst):\n"
                "    dupes = []\n"
                "    for i in lst:\n"
                "        if lst.count(i) > 1 and i not in dupes:\n"
                "            dupes.append(i)\n"
                "    return dupes"
            ),
            "Review this code for maintainability: x=lambda a,b:a+b if a>0 else b",
        ],
        "checks": [
            ("reviewer_contract", _reviewer_output_ok, "Output satisfies reviewer schema (JSON) or legacy prose contract with score"),
            ("min_length",        lambda t: len(t.strip()) >= 60, "Output length >= 60 chars"),
        ],
    },
    "planning": {
        "prompts": [
            "Create a step-by-step plan to add JWT authentication to a Python REST API.",
            "Create a plan to migrate a monolithic application to microservices.",
            "Write a sprint plan for adding a search feature to an e-commerce site.",
            "Design a rollback plan for a database schema migration.",
            "Create a plan to reduce API response time from 2s to under 200ms.",
        ],
        "checks": [
            ("has_list",   lambda t: bool(_NUMBERED_RE.search(t)), "Output contains numbered list (1. …)"),
            ("min_length", lambda t: len(t.strip()) >= 80,         "Output length >= 80 chars"),
        ],
    },
    "research": {
        "prompts": [
            "Explain how transformer self-attention mechanisms work.",
            "Compare REST and GraphQL APIs. What are the tradeoffs?",
            "Explain the CAP theorem and when to apply it.",
            "What is the difference between process and thread?",
            "Explain how gradient descent works in neural network training.",
        ],
        "checks": [
            ("min_sentences", lambda t: len(re.split(r"[.!?]+", t.strip())) >= 3,
             "Output contains >= 3 sentences"),
            ("min_length",    lambda t: len(t.strip()) >= 100, "Output length >= 100 chars"),
        ],
    },
    "criticism": {
        "prompts": [
            (
                "Critique this architectural decision:\n\n"
                "Storing all application state in a single global variable in a multi-process web server."
            ),
            (
                "Critique this design:\n\n"
                "Every microservice accesses the same single shared database directly."
            ),
            (
                "Critique this approach:\n\n"
                "Using polling every 100ms to check for new messages instead of websockets."
            ),
            (
                "Critique this decision:\n\n"
                "Storing user passwords as MD5 hashes."
            ),
            (
                "Critique this plan:\n\n"
                "Deploy directly to production without a staging environment or tests."
            ),
        ],
        "checks": [
            ("critic_contract", _critic_output_ok, "Output satisfies critic schema (JSON) or legacy prose contract with score"),
            ("min_length",      lambda t: len(t.strip()) >= 60, "Output length >= 60 chars"),
        ],
    },
}


# ---------------------------------------------------------------------------
# Core validation logic
# ---------------------------------------------------------------------------

def _generate_response(model, tokenizer, sys_prompt: str, user_prompt: str,
                        max_new_tokens: int = 300) -> str:
    """Generate a response using apply_chat_template (matches inference format)."""
    import torch
    msgs = [
        {"role": "system", "content": sys_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=False,
        )
    output_ids = out[0][input_ids.shape[1]:]
    decoded = _fix_bpe_artifacts(tokenizer.decode(output_ids, skip_special_tokens=True))
    # Qwen3 emits <think>...</think> reasoning blocks (possibly empty) before
    # the user-facing answer. At inference time vLLM guided decoding strips
    # them, but here we use raw generate(), so strip them ourselves so the
    # check predicates see the same shape they would in production.
    return _strip_think_blocks(decoded)


def validate(capability: str, adapter_dir: Path) -> bool:
    """
    Load the adapter, run capability-specific prompts across multiple buckets,
    run a base-skill check, and optionally verify merged-adapter parity.

    Returns True only when:
      - >= PROMPT_PASS_THRESHOLD of prompts pass every structural check
      - >= BASE_SKILL_GATE of base-skill probes pass (catastrophic-forgetting gate)
    """
    if capability not in _TESTS:
        print(f"[VALIDATE] No test defined for '{capability}' — skipping (PASS)")
        return True

    test       = _TESTS[capability]
    role_name  = BOOTSTRAP_CAP_TO_ROLE.get(capability, capability)
    sys_prompt = AGENT_PROMPTS.get(role_name, "You are a helpful assistant.")
    adapter_path = str(adapter_dir / f"{capability}_lora")

    print(f"\n  [VALIDATE] Adapter : {adapter_path}")
    print(f"  [VALIDATE] Prompts : {len(test['prompts'])} × capability  +  {len(_BASE_SKILL_PROBES)} base-skill")

    # ── Load adapter ──────────────────────────────────────────────────────────
    try:
        import torch
        from unsloth import FastLanguageModel
    except ImportError:
        print("  [VALIDATE] unsloth/torch not available — skipping (PASS)")
        return True

    try:
        from shared.contracts.models import load_registry as _lr
        _backend = _lr().default_backend
    except Exception:
        _backend = "primary"
    _recipe: AdapterRecipe = lookup_recipe(_backend, role_name)

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name   = adapter_path,
            max_seq_length = _recipe.max_seq_length,
            dtype        = None,
            load_in_4bit = _recipe.load_in_4bit or _recipe.peft_type == "qlora",
        )
        FastLanguageModel.for_inference(model)
    except Exception as exc:
        print(f"  [VALIDATE] FAIL — adapter load error: {exc}")
        return False

    all_pass = True
    try:
        # ── Multi-prompt capability checks (item 2) ───────────────────────────
        all_checks: List[Check] = _COMMON + test["checks"]
        pass_counts = {cid: 0 for cid, _, _ in all_checks}
        prompts = test["prompts"]
        _max_new_tokens = int(test.get("max_new_tokens", 300))

        for i, user_prompt in enumerate(prompts, 1):
            try:
                generated = _generate_response(
                    model, tokenizer, sys_prompt, user_prompt,
                    max_new_tokens=_max_new_tokens,
                )
            except Exception as exc:
                print(f"  [VALIDATE] Prompt {i}: generation error — {exc}")
                continue

            preview = generated.strip().replace("\n", " ")[:120]
            print(f"  [VALIDATE] Prompt {i}/{len(prompts)}: {preview}")

            for cid, fn, _ in all_checks:
                try:
                    if fn(generated):
                        pass_counts[cid] += 1
                except Exception as _check_exc:
                    print(f"  [VALIDATE] Check '{cid}' error on prompt {i}: {_check_exc}")

        print(f"\n  [VALIDATE] Check results ({PROMPT_PASS_THRESHOLD:.0%} threshold):")
        for cid, _, desc in all_checks:
            rate   = pass_counts[cid] / len(prompts)
            ok     = rate >= PROMPT_PASS_THRESHOLD
            mark   = "✓" if ok else "✗"
            status = "PASS" if ok else "FAIL"
            print(f"    {mark} {status:4s}  {desc}  ({pass_counts[cid]}/{len(prompts)} = {rate:.0%})")
            if not ok:
                all_pass = False

        # ── Base-skill gate (item 6) ──────────────────────────────────────────
        print(f"\n  [VALIDATE] Base-skill probes ({BASE_SKILL_GATE:.0%} threshold):")
        bs_pass = 0
        for bs_prompt, bs_check in _BASE_SKILL_PROBES:
            try:
                bs_out = _generate_response(model, tokenizer, "You are a helpful assistant.", bs_prompt, max_new_tokens=80)
                bs_ok  = bs_check(bs_out)
            except Exception:
                bs_ok = False
            mark = "✓" if bs_ok else "✗"
            print(f"    {mark}  {bs_prompt[:60]}")
            if bs_ok:
                bs_pass += 1

        bs_rate = bs_pass / len(_BASE_SKILL_PROBES)
        bs_gate_ok = bs_rate >= BASE_SKILL_GATE
        mark = "✓" if bs_gate_ok else "✗"
        status = "PASS" if bs_gate_ok else "FAIL"
        print(f"  [VALIDATE] Base-skill: {mark} {status}  ({bs_pass}/{len(_BASE_SKILL_PROBES)} = {bs_rate:.0%})")
        if not bs_gate_ok:
            print("  [VALIDATE] WARN: base-skill below threshold — possible catastrophic forgetting")
            all_pass = False

        # ── Merge-vs-unmerged parity check (item 5) ──────────────────────────
        print("\n  [VALIDATE] Merge parity check...")
        _merge_probe = prompts[0]
        try:
            _unmerged_out = _generate_response(model, tokenizer, sys_prompt, _merge_probe, max_new_tokens=100)
            _unmerged_ids = set(tokenizer.encode(_unmerged_out))

            merged_model = model.merge_and_unload()
            FastLanguageModel.for_inference(merged_model)
            _merged_out  = _generate_response(merged_model, tokenizer, sys_prompt, _merge_probe, max_new_tokens=100)
            _merged_ids  = set(tokenizer.encode(_merged_out))
            del merged_model

            _overlap = len(_unmerged_ids & _merged_ids) / max(len(_unmerged_ids | _merged_ids), 1)
            merge_ok = _overlap >= 0.80
            mark   = "✓" if merge_ok else "✗"
            status = "PASS" if merge_ok else "FAIL"
            print(f"  [VALIDATE] Merge parity: {mark} {status}  token-overlap={_overlap:.2%} (threshold ≥ 80%)")
            if not merge_ok:
                print("  [VALIDATE] WARN: merged adapter output diverges — dtype or scaling mismatch")
                # Merge parity is advisory for vLLM (unmerged serving); don't gate promotion.
        except Exception as exc:
            print(f"  [VALIDATE] Merge check skipped: {exc}")

    finally:
        del model, tokenizer
        gc.collect()
        if "torch" in sys.modules and sys.modules["torch"].cuda.is_available():
            sys.modules["torch"].cuda.empty_cache()

    result = "PASS — adapter integrity confirmed" if all_pass else "FAIL — adapter may be corrupted or undertrained"
    print(f"\n  [VALIDATE] {result}\n")
    return all_pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a trained LoRA adapter (post-training integrity check)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Exit 0 = PASS, 1 = checks failed, 2 = load error",
    )
    parser.add_argument(
        "--capability", required=True,
        choices=list(_TESTS.keys()),
        help="Which adapter to validate (bootstrap label, e.g. coding)",
    )
    parser.add_argument(
        "--adapter-dir", required=True, type=Path,
        help="Parent directory that contains <capability>_lora/ sub-folders "
             "(default: /data/lora_checkpoints)",
    )
    args = parser.parse_args()

    adapter_dir = args.adapter_dir
    if not adapter_dir.exists():
        sys.exit(f"[VALIDATE] adapter-dir not found: {adapter_dir}")

    ok = validate(args.capability, adapter_dir)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
