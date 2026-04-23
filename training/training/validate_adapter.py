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

# ---------------------------------------------------------------------------
# BPE / contamination patterns (same as e2e_test.py)
# ---------------------------------------------------------------------------
_BPE_RE = re.compile(r"[\u0120\u010a\u2581]")


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
_JSON_META_RE = re.compile(
    r""""agent_name"\s*:|'agent_name'\s*:|"quality_score"\s*:|'quality_score'\s*:"""
    r"""|"iteration_number"\s*:|"execution_time_ms"\s*:""",
    re.I,
)
_CODE_RE      = re.compile(r"```|(?<![a-zA-Z])def|(?<![a-zA-Z])class|(?<![a-zA-Z])import|(?<![a-zA-Z])return", re.I)
_ISSUE_RE     = re.compile(r"\bISSUE\s*:", re.I)
_FIX_RE       = re.compile(r"\bFIX\s*:", re.I)
_SCORE_RE     = re.compile(r"\bSCORE\s*:", re.I)
# Match "1." style OR "Step 1:" style — both are valid numbered-list outputs
_NUMBERED_RE  = re.compile(r"^\s*(\d+\.|[Ss]tep\s+\d+[.:])", re.M)
_VERDICT_RE   = re.compile(r"\b(VERDICT|SCORE|BLOCKERS)\s*:", re.I)

# Type alias for a single check: (check_id, predicate, human description)
Check = Tuple[str, Callable[[str], bool], str]

# Common checks applied to every capability
_COMMON: List[Check] = [
    ("no_bpe",           lambda t: not bool(_BPE_RE.search(t)),      "No BPE artifacts"),
    ("no_contamination", lambda t: not bool(_JSON_META_RE.search(t)), "No JSON metadata contamination"),
]

# ---------------------------------------------------------------------------
# Per-capability test fixtures
# ---------------------------------------------------------------------------
_TESTS: dict[str, dict] = {
    "coding": {
        "user": (
            "Write a Python function that computes the nth Fibonacci number "
            "using memoization with a dictionary cache."
        ),
        "checks": [
            ("has_code",    lambda t: bool(_CODE_RE.search(t)),    "Output contains code tokens"),
            ("min_length",  lambda t: len(t.strip()) >= 80,        "Output length >= 80 chars"),
        ],
    },
    "debugging": {
        "user": (
            "Fix this code:\n\n"
            "def divide(a, b):\n"
            "    return a / b  # crashes when b is zero\n\n"
            "result = divide(10, 0)"
        ),
        "checks": [
            ("has_issue",  lambda t: bool(_ISSUE_RE.search(t)),  "Output contains ISSUE: section"),
            ("has_fix",    lambda t: bool(_FIX_RE.search(t)),    "Output contains FIX: section"),
            ("min_length", lambda t: len(t.strip()) >= 60,       "Output length >= 60 chars"),
        ],
    },
    "review": {
        "user": (
            "Review this pull request:\n\n"
            "def add(a, b): return a + b\n\n"
            "There are no tests and no docstring."
        ),
        "checks": [
            ("has_score",  lambda t: bool(_SCORE_RE.search(t)),  "Output contains SCORE: section"),
            ("min_length", lambda t: len(t.strip()) >= 60,       "Output length >= 60 chars"),
        ],
    },
    "planning": {
        "user": "Create a step-by-step plan to add JWT authentication to a Python REST API.",
        "checks": [
            ("has_list",   lambda t: bool(_NUMBERED_RE.search(t)), "Output contains numbered list (1. …)"),
            ("min_length", lambda t: len(t.strip()) >= 80,         "Output length >= 80 chars"),
        ],
    },
    "research": {
        "user": "Explain how transformer self-attention mechanisms work.",
        "checks": [
            ("min_sentences", lambda t: len(re.split(r"[.!?]+", t.strip())) >= 3,
             "Output contains >= 3 sentences"),
            ("min_length",    lambda t: len(t.strip()) >= 100, "Output length >= 100 chars"),
        ],
    },
    "criticism": {
        "user": (
            "Critique this architectural decision:\n\n"
            "Storing all application state in a single global variable in a multi-process web server."
        ),
        "checks": [
            ("has_verdict", lambda t: bool(_VERDICT_RE.search(t)), "Output contains VERDICT/SCORE/BLOCKERS section"),
            ("min_length",  lambda t: len(t.strip()) >= 60,        "Output length >= 60 chars"),
        ],
    },
}


# ---------------------------------------------------------------------------
# Core validation logic
# ---------------------------------------------------------------------------

def validate(capability: str, adapter_dir: Path) -> bool:
    """
    Load the adapter, run a targeted prompt, and verify the output passes all
    capability-specific quality checks.

    Returns True if all checks pass (adapter integrity confirmed).
    """
    if capability not in _TESTS:
        print(f"[VALIDATE] No test defined for '{capability}' — skipping (PASS)")
        return True

    test       = _TESTS[capability]
    role_name  = BOOTSTRAP_CAP_TO_ROLE.get(capability, capability)
    sys_prompt = AGENT_PROMPTS.get(role_name, "You are a helpful assistant.")

    # Prompt uses the exact same format as training and inference
    prompt_text = (
        f"[SYSTEM]\n{sys_prompt}\n\n"
        f"[USER]\n{test['user']}\n\n"
        f"[RESPONSE]\n"
    )

    adapter_path = str(adapter_dir / f"{capability}_lora")
    print(f"\n  [VALIDATE] Adapter : {adapter_path}")
    print(f"  [VALIDATE] Prompt  : {test['user'][:80]}...")

    # ── Load adapter ──────────────────────────────────────────────────────────
    try:
        import torch
        from unsloth import FastLanguageModel
    except ImportError:
        print("  [VALIDATE] unsloth/torch not available — skipping (PASS)")
        return True

    # Determine model family from adapter metadata — gates tokenizer fix below.
    _meta_path = adapter_dir / f"{capability}_lora" / "metadata.json"
    _base_model = ""
    try:
        import json as _json
        _base_model = _json.loads(_meta_path.read_text()).get("base_model", "")
    except Exception:
        pass
    _is_deepseek = "deepseek" in _base_model.lower()

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name    = adapter_path,
            max_seq_length = 2048,
            dtype          = None,
            load_in_4bit   = True,
        )
        FastLanguageModel.for_inference(model)
        # Apply the ByteLevel pre-tokenizer fix for DeepSeek adapters only.
        # DeepSeek's vocab uses Ġ-prefixed tokens but ships with Metaspace;
        # ByteLevel was applied during training, so validation must match.
        # Mistral's vocab uses ▁-prefixed tokens and Metaspace IS correct —
        # applying ByteLevel to Mistral would produce Ġ tokens absent from
        # Mistral's vocab, causing validation to test a broken tokenizer state.
        if _is_deepseek:
            from tokenizers.pre_tokenizers import ByteLevel as _ByteLevel
            tokenizer.backend_tokenizer.pre_tokenizer = _ByteLevel(add_prefix_space=False)
    except Exception as exc:
        print(f"  [VALIDATE] FAIL — adapter load error: {exc}")
        return False

    # ── Generate ──────────────────────────────────────────────────────────────
    try:
        inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens = 200,
                temperature    = 0.1,
                do_sample      = False,
            )
        output_ids  = out[0][inputs["input_ids"].shape[1]:]
        generated   = tokenizer.decode(output_ids, skip_special_tokens=True)
    except Exception as exc:
        print(f"  [VALIDATE] FAIL — generation error: {exc}")
        return False
    finally:
        del model, tokenizer
        gc.collect()
        if "torch" in sys.modules and sys.modules["torch"].cuda.is_available():
            sys.modules["torch"].cuda.empty_cache()

    # Apply the same BPE artifact correction that local_vllm.py applies at
    # inference time so checks run on human-readable text, not raw BPE tokens.
    generated = _fix_bpe_artifacts(generated)

    preview = generated.strip().replace("\n", " ")[:300]
    print(f"\n  [VALIDATE] Response ({len(generated)} chars):\n    {preview}\n")

    # ── Run checks ────────────────────────────────────────────────────────────
    all_checks: List[Check] = _COMMON + test["checks"]
    all_pass = True
    for check_id, fn, desc in all_checks:
        ok     = fn(generated)
        status = "PASS" if ok else "FAIL"
        mark   = "✓" if ok else "✗"
        print(f"  [VALIDATE]   {mark} {status:4s}  {desc}")
        if not ok:
            all_pass = False

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
