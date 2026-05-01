#!/usr/bin/env python3
"""
Quick smoke test for coding_lora and debugging_lora.
Run inside the training container:
  python /scripts/smoke_test_adapters.py

Loads each adapter, runs one targeted prompt, and prints the output.
Pass/Fail heuristics:
  - PASS: output contains code tokens (def/class/import/return/```), not JSON metadata
  - FAIL: output contains 'agent_name', 'quality_score', 'iteration_number' (contamination signal)
"""
import gc
import re
import sys

import torch
from unsloth import FastLanguageModel

ADAPTER_BASE = "/data/lora_checkpoints"

CONTAMINATION_RE = re.compile(
    r"'agent_name'\s*:|\"agent_name\"\s*:|'quality_score'\s*:|\"quality_score\"\s*:"
    r"|'iteration_number'\s*:|\"iteration_number\"\s*:|'execution_time_ms'\s*:",
    re.I,
)
CODE_RE = re.compile(r"```|def |class |import |\breturn\b", re.I)
ISSUE_FIX_RE = re.compile(r"\bISSUE\s*:", re.I)

TESTS = [
    {
        "adapter": f"{ADAPTER_BASE}/coding_lora",
        "label": "coding_lora",
        "system": (
            "You are CoderAgent — an expert software engineer.\n"
            "Write clean, idiomatic, well-commented code.\n"
            "Include type hints (Python & Typescript).\n"
            "Output ONLY the code and any necessary inline comments."
        ),
        "user": (
            "Write a Python and Typescript function that computes the nth Fibonacci number "
            "using memoization with a dictionary cache."
        ),
        "checks": [
            ("has_code", lambda t: bool(CODE_RE.search(t)), "Output contains code tokens"),
            ("no_contamination", lambda t: not bool(CONTAMINATION_RE.search(t)), "No JSON metadata contamination"),
            ("min_length", lambda t: len(t.strip()) >= 80, "Output length >= 80 chars"),
        ],
    },
    {
        "adapter": f"{ADAPTER_BASE}/debugging_lora",
        "label": "debugging_lora",
        "system": (
            "You are DebuggerAgent — a senior debugging specialist.\n"
            "First state the root cause clearly. Then provide the corrected code.\n"
            "Format: ISSUE: <root cause>\nFIX: <corrected code or diff>"
        ),
        "user": (
            "Fix this code:\n\n"
            "def divide(a, b):\n"
            "    return a / b  # crashes when b is zero\n"
            "\n"
            "result = divide(10, 0)"
        ),
        "checks": [
            ("has_issue_label", lambda t: bool(ISSUE_FIX_RE.search(t)), "Output contains ISSUE: label"),
            ("no_contamination", lambda t: not bool(CONTAMINATION_RE.search(t)), "No JSON metadata contamination"),
            ("min_length", lambda t: len(t.strip()) >= 60, "Output length >= 60 chars"),
        ],
    },
]


def run_test(test: dict) -> dict:
    label = test["label"]
    print(f"\n{'=' * 62}")
    print(f"  ADAPTER: {label}")
    print(f"{'=' * 62}")
    print(f"  Prompt : {test['user'][:80]}...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=test["adapter"],
        max_seq_length=512,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    messages = [
        {"role": "system", "content": test["system"]},
        {"role": "user",   "content": test["user"]},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=False,
            repetition_penalty=1.1,
        )

    # Decode the full sequence then strip the prompt — decoding only new
    # tokens loses byte-level boundary context in BPE tokenizers causing
    # spaces to be dropped at the split point.
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    prompt_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    output_text = full_text[len(prompt_text):].lstrip()

    print(f"\n  --- OUTPUT ({len(output_text)} chars) ---")
    print(output_text[:1200])
    if len(output_text) > 1200:
        print("  ...[truncated]")
    print()

    results = {}
    all_pass = True
    for check_name, check_fn, description in test["checks"]:
        passed = check_fn(output_text)
        results[check_name] = passed
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {description}")
        if not passed:
            all_pass = False

    results["all_pass"] = all_pass
    results["output"] = output_text

    # Free GPU memory before next adapter load
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return results


def main():
    all_results = {}
    for test in TESTS:
        result = run_test(test)
        all_results[test["label"]] = result

    print(f"\n{'=' * 62}")
    print("  SUMMARY")
    print(f"{'=' * 62}")
    overall_pass = True
    for label, r in all_results.items():
        status = "PASS" if r["all_pass"] else "FAIL"
        print(f"  {label:20s}  {status}")
        if not r["all_pass"]:
            overall_pass = False

    print()
    if overall_pass:
        print("  Both adapters passed smoke test — cleared for eval_adapters.py run.")
    else:
        print("  One or more adapters FAILED — review output before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()
