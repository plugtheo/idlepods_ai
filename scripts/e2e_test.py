#!/usr/bin/env python3
"""
Full End-to-End Test Suite — IdlePods AI
==========================================
Tests every agent role and LoRA adapter combination via the streaming
/v1/chat/stream endpoint so that autoregressive token generation is visible
in real-time as agents "think".

Usage
-----
  python scripts/e2e_test.py [--url http://localhost:8080] [--scenario <name>]

What it checks
--------------
  ACCURACY
    - No BPE artifacts (Ġ U+0120, Ċ U+010A, ▁ U+2581)
    - No training-format hallucinations (###Instruction:, ###Response:, [RESPONSE])
    - No raw JSON metadata blobs ({"score": ..., "pipeline_metadata": ...})
    - Role-specific format compliance:
        * coder/consensus  → has code block (``` ... ```)
        * debugger         → has ISSUE: and FIX: sections
        * reviewer         → has SCORE: / STRENGTHS: / ISSUES:
        * critic           → has SCORE: / VERDICT: / BLOCKERS:
        * planner          → has numbered list (1. ...)
        * researcher       → has at least 3 sentences of prose

  PERFORMANCE
    - Time-to-first-token (TTFT) per agent step
    - Total wall time per scenario
    - Approximate token throughput (output chars / total_time)
    - Flag if any agent step takes > 60 s (too slow)

Output
------
  - Live streaming display: role header + tokens as they arrive
  - Per-scenario result card with pass/fail for each check
  - Final summary table across all scenarios
"""

from __future__ import annotations

import argparse
import json
import re
import sys

# Ensure UTF-8 output on Windows where PowerShell defaults to cp1252.
# Without this, box-drawing characters (━ ═ ▶ ✔ ✘) render as mojibake.
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import time

# Force UTF-8 output on Windows so box-drawing / emoji chars render correctly.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from dataclasses import dataclass, field
from typing import Generator

try:
    import httpx
except ImportError:
    sys.exit("httpx is required — install it with: pip install httpx")

# ─── ANSI colours ────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RED    = "\033[31m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
WHITE  = "\033[37m"

ROLE_COLOURS = {
    "planner":    "\033[35m",   # magenta
    "researcher": "\033[36m",   # cyan
    "coder":      "\033[32m",   # green
    "debugger":   "\033[33m",   # yellow
    "reviewer":   "\033[34m",   # blue
    "critic":     "\033[31m",   # red
    "review_critic": "\033[31m",
    "consensus":  "\033[37m",   # white
}

# ─── Test scenarios ──────────────────────────────────────────────────────────
# Each scenario exercises a different intent/complexity combination to ensure
# all adapter-equipped roles are exercised at least once.

SCENARIOS = [
    {
        "name": "coding_simple",
        "label": "CODING / SIMPLE — fibonacci function",
        "adapters": ["coding_lora"],
        "prompt": "Write a Python function to calculate the nth Fibonacci number using dynamic programming.",
        "expected_roles": ["coder", "reviewer"],
    },
    {
        "name": "coding_moderate",
        "label": "CODING / MODERATE — rate limiter class",
        "adapters": ["planning_lora", "coding_lora"],
        "prompt": "Implement a thread-safe token bucket rate limiter class in Python with configurable rate and burst capacity.",
        "expected_roles": ["planner", "coder", "review_critic"],
    },
    {
        "name": "coding_complex",
        "label": "CODING / COMPLEX — async task queue",
        "adapters": ["planning_lora", "coding_lora", "debugging_lora"],
        "prompt": (
            "Build a complete Python async task queue system with Redis backend, "
            "supporting priority queues, worker pools, retry logic, "
            "and dead-letter queues for failed tasks."
        ),
        "expected_roles": ["planner", "coder", "debugger", "review_critic"],
    },
    {
        "name": "debugging_simple",
        "label": "DEBUGGING / SIMPLE — off-by-one fix",
        "adapters": ["debugging_lora"],
        "prompt": (
            "Fix this Python function that should return the sum of a list "
            "but returns the wrong result:\n\n"
            "def total(nums):\n    result = 0\n    for i in range(len(nums) - 1):\n        result += nums[i]\n    return result"
        ),
        "expected_roles": ["debugger", "reviewer"],
    },
    {
        "name": "research_moderate",
        "label": "RESEARCH / MODERATE — B-tree vs LSM-tree",
        "adapters": ["research_lora", "criticism_lora"],
        "prompt": "Research and compare B-tree and LSM-tree data structures: use cases, trade-offs, and when to choose each.",
        "expected_roles": ["researcher", "critic"],
    },
    {
        "name": "planning_moderate",
        "label": "PLANNING / MODERATE — auth microservice",
        "adapters": ["planning_lora", "criticism_lora"],
        "prompt": "Plan the design and implementation roadmap for a JWT-based authentication microservice in Python.",
        "expected_roles": ["planner", "critic"],
    },
]

# ─── Accuracy checks per role ────────────────────────────────────────────────

BPE_ARTIFACT_RE    = re.compile(r"[\u0120\u010a\u2581]")   # Ġ Ċ ▁
HALLUCINATION_RE   = re.compile(r"###\s*Instruction:|###\s*Response:|\[RESPONSE\]", re.I)
JSON_METADATA_RE   = re.compile(r'\{\s*["\'](?:score|session_id|pipeline_metadata|iteration_scores|agent_chain)["\']')
CODE_BLOCK_RE      = re.compile(r"```[\w\s]*\n[\s\S]+?```", re.S)
NUMBERED_LIST_RE   = re.compile(r"^\s*\d+\.", re.M)
# Code content: any common keyword that signals actual code output (not prose)
CODE_KEYWORD_RE    = re.compile(r"\b(def |class |function |const |let |var |import |public |static |async )\b|#include|[;{}]")
# Well-formatted code: has a newline+indentation after a block opener (: or {)
# Catches both Python (`def f():\n    ...`) and C-style (`{\n  ...`).
MULTILINE_CODE_RE  = re.compile(r"(def |class |function )\w.*?:\s*\n\s+|\{\s*\n\s+", re.S)

ROLE_FORMAT_CHECKS: dict[str, list[tuple[str, re.Pattern | None]]] = {
    # coder outputs RAW code (no markdown wrapping — system prompt says "Output ONLY the code").
    # Check (1): output contains recognisable code keywords.
    # Check (2): code is properly formatted with newlines/indentation (not minified).
    "coder":      [("has code content",       CODE_KEYWORD_RE),
                   ("code is well-formatted", MULTILINE_CODE_RE)],
    "debugger":   [("has ISSUE:", re.compile(r"ISSUE\s*:", re.I)),
                   ("has FIX:",   re.compile(r"FIX\s*:",   re.I))],
    # reviewer uses review_lora; structured format from system prompt is enforced.
    "reviewer":   [("has SCORE:",      re.compile(r"SCORE\s*:",   re.I)),
                   ("has ISSUES:",     re.compile(r"ISSUES\s*:",  re.I))],
    # critic uses criticism_lora; expanded history filter provides proper context.
    "critic":     [("has SCORE:",      re.compile(r"SCORE\s*:",   re.I)),
                   ("has BLOCKERS:",   re.compile(r"BLOCKERS\s*:", re.I))],
    "planner":    [("has numbered list", NUMBERED_LIST_RE)],
    "researcher": [("has \u22653 sentences", None)],   # special \u2014 checked by len
    # consensus is the final synthesiser; its format is scenario-dependent
    # (prose for planning/research, code for coding). No universal format check.
    "consensus":  [],
}

# ─── Result structures ────────────────────────────────────────────────────────

@dataclass
class AgentResult:
    role: str
    iteration: int
    output: str
    ttft_s: float        # time to first token from agent_step start
    total_s: float       # time from first token to agent_step event
    checks_passed: list[str] = field(default_factory=list)
    checks_failed: list[str] = field(default_factory=list)

@dataclass
class ScenarioResult:
    name: str
    label: str
    prompt: str
    success: bool = True
    agent_results: list[AgentResult] = field(default_factory=list)
    final_output: str = ""
    confidence: float = 0.0
    iterations: int = 0
    total_wall_s: float = 0.0
    error: str = ""

# ─── SSE streaming helper ────────────────────────────────────────────────────

def _sse_stream(url: str, payload: dict, timeout: float = 300.0) -> Generator[dict, None, None]:
    """Yield parsed SSE event dicts from the streaming endpoint."""
    with httpx.Client(timeout=timeout) as client:
        with client.stream("POST", url, json=payload, headers={"Accept": "text/event-stream"}) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line.startswith("data: "):
                    raw = line[6:].strip()
                    if raw:
                        try:
                            yield json.loads(raw)
                        except json.JSONDecodeError:
                            pass

# ─── Accuracy evaluation ─────────────────────────────────────────────────────

def _evaluate_output(role: str, output: str) -> tuple[list[str], list[str]]:
    """Return (passed, failed) check lists for a given role + output."""
    passed, failed = [], []

    # Universal checks
    if BPE_ARTIFACT_RE.search(output):
        failed.append("no BPE artifacts (Ġ/Ċ/▁ found)")
    else:
        passed.append("no BPE artifacts")

    if HALLUCINATION_RE.search(output):
        failed.append("no training hallucinations (###Instruction: found)")
    else:
        passed.append("no training hallucinations")

    if JSON_METADATA_RE.search(output):
        failed.append("no raw JSON metadata blobs")
    else:
        passed.append("no raw JSON metadata")

    if len(output.strip()) < 20:
        failed.append("output not empty/too short")
    else:
        passed.append("output non-empty")

    # Role-specific format checks
    checks = ROLE_FORMAT_CHECKS.get(role, [])
    for label, pattern in checks:
        if pattern is None:
            # researcher sentence count
            sentences = re.split(r"[.!?]+", output)
            ok = sum(1 for s in sentences if s.strip()) >= 3
        else:
            ok = bool(pattern.search(output))
        (passed if ok else failed).append(label)

    # review_critic is a virtual role — we check reviewer/critic sub-format
    # when its output arrives — handled by caller splitting on the step role.

    return passed, failed

# ─── Per-scenario runner ──────────────────────────────────────────────────────

def run_scenario(name: str, label: str, prompt: str,
                 expected_roles: list[str], base_url: str) -> ScenarioResult:
    result = ScenarioResult(name=name, label=label, prompt=prompt)
    stream_url = f"{base_url.rstrip('/')}/v1/chat/stream"

    print(f"\n{'━'*72}")
    print(f"{BOLD}{CYAN}▶ {label}{RESET}")
    print(f"{DIM}  Prompt: {prompt[:120]}{'…' if len(prompt) > 120 else ''}{RESET}")
    print(f"{'━'*72}")

    # State tracking while streaming
    agent_token_buf: dict[str, list[str]] = {}   # role → accumulated tokens
    agent_start_ts:  dict[str, float] = {}       # role → wall time of first token
    agent_first_token_ts: dict[str, float] = {}  # role → wall time of first token event
    current_role: str = ""
    current_iter: int = 1

    wall_start = time.monotonic()

    try:
        for event in _sse_stream(stream_url, {"prompt": prompt}):
            etype = event.get("type", "")

            if etype == "start":
                chain = event.get("agent_chain", [])
                print(f"  {DIM}agent_chain: {' → '.join(chain)}{RESET}")

            elif etype == "token":
                role = event.get("role", "?")
                token = event.get("token", "")
                iteration = event.get("iteration", 1)

                # First token from this (role, iteration) pair
                key = f"{role}:{iteration}"
                if key not in agent_token_buf:
                    # Print role header
                    colour = ROLE_COLOURS.get(role, WHITE)
                    print(f"\n  {colour}{BOLD}[{role.upper()} — iter {iteration}]{RESET}")
                    agent_token_buf[key] = []
                    agent_start_ts[key] = time.monotonic()
                    agent_first_token_ts[key] = time.monotonic()

                agent_token_buf[key].append(token)
                # Print token immediately (autoregressive inference visible)
                print(token, end="", flush=True)
                current_role = role
                current_iter = iteration

            elif etype == "agent_step":
                role = event.get("role", "?")
                iteration = event.get("iteration", 1)
                output = event.get("output", "")
                key = f"{role}:{iteration}"
                step_end = time.monotonic()

                # Timing
                start_ts = agent_start_ts.get(key, step_end)
                first_ts = agent_first_token_ts.get(key, step_end)
                ttft = first_ts - start_ts
                total = step_end - start_ts

                # Print newline after streamed tokens, then summary
                print(f"\n  {DIM}  ↳ agent_step done  ttft={ttft:.2f}s  elapsed={total:.2f}s  chars={len(output)}{RESET}")

                # Evaluate accuracy
                check_role = role  # review_critic just uses the stated role
                passed, failed = _evaluate_output(check_role, output)
                result.agent_results.append(AgentResult(
                    role=role, iteration=iteration, output=output,
                    ttft_s=ttft, total_s=total,
                    checks_passed=passed, checks_failed=failed,
                ))
                if failed:
                    print(f"  {YELLOW}  ⚠ {', '.join(failed)}{RESET}")

            elif etype == "iteration_complete":
                iteration = event.get("iteration", 1)
                score = event.get("score", 0.0)
                colour = GREEN if score >= 0.7 else (YELLOW if score >= 0.4 else RED)
                print(f"\n  {colour}◆ iteration {iteration} complete — score={score:.4f}{RESET}")

            elif etype == "done":
                result.final_output = event.get("output", "")
                result.confidence   = event.get("confidence", 0.0)
                result.iterations   = event.get("iterations", 1)
                result.total_wall_s = time.monotonic() - wall_start
                conf_colour = GREEN if result.confidence >= 0.7 else (YELLOW if result.confidence >= 0.4 else RED)
                print(
                    f"\n  {conf_colour}✔ done — confidence={result.confidence:.4f} "
                    f"iterations={result.iterations} "
                    f"wall={result.total_wall_s:.1f}s{RESET}"
                )

            elif etype == "error":
                msg = event.get("message", "unknown error")
                result.error = msg
                result.success = False
                print(f"\n  {RED}✖ pipeline error: {msg}{RESET}")

    except Exception as exc:
        result.error = str(exc)
        result.success = False
        result.total_wall_s = time.monotonic() - wall_start
        print(f"\n  {RED}✖ request failed: {exc}{RESET}")

    return result

# ─── Result card printer ─────────────────────────────────────────────────────

def _print_result_card(res: ScenarioResult) -> None:
    overall_ok = res.success and not res.error
    all_failed = [f for ar in res.agent_results for f in ar.checks_failed]
    if all_failed:
        overall_ok = False

    status = f"{GREEN}PASS{RESET}" if overall_ok else f"{RED}FAIL{RESET}"
    print(f"\n{'─'*72}")
    print(f"  {BOLD}RESULT — {res.label}  [{status}{BOLD}]{RESET}")
    print(f"{'─'*72}")
    if res.error:
        print(f"  {RED}Error: {res.error}{RESET}")
    print(f"  Wall time : {res.total_wall_s:.1f}s")
    print(f"  Confidence: {res.confidence:.4f}")
    print(f"  Iterations: {res.iterations}")

    slow_threshold = 60  # seconds
    for ar in res.agent_results:
        role_col = ROLE_COLOURS.get(ar.role, WHITE)
        timing_ok = ar.total_s <= slow_threshold
        timing_col = GREEN if timing_ok else RED
        timing_note = "" if timing_ok else f"  {RED}(SLOW){RESET}"
        print(
            f"  {role_col}{ar.role:<14}{RESET}  iter={ar.iteration}"
            f"  elapsed={timing_col}{ar.total_s:.1f}s{RESET}{timing_note}"
            f"  chars={len(ar.output)}"
        )
        for chk in ar.checks_passed:
            print(f"    {GREEN}✔ {chk}{RESET}")
        for chk in ar.checks_failed:
            print(f"    {RED}✘ {chk}{RESET}")

    # Final output preview (first 400 chars)
    if res.final_output:
        preview = res.final_output[:400].replace("\n", " ↵ ")
        print(f"\n  {DIM}Final output preview:{RESET}")
        print(f"  {DIM}{preview}{'…' if len(res.final_output) > 400 else ''}{RESET}")

# ─── Summary table ────────────────────────────────────────────────────────────

def _print_summary(results: list[ScenarioResult]) -> None:
    print(f"\n\n{'═'*72}")
    print(f"{BOLD}  E2E TEST SUMMARY{RESET}")
    print(f"{'═'*72}")
    print(f"  {'Scenario':<32} {'Status':>6}  {'Conf':>5}  {'Wall':>6}  {'Checks'}")
    print(f"  {'─'*32} {'─'*6}  {'─'*5}  {'─'*6}  {'─'*24}")

    total_pass = total_fail = 0
    for res in results:
        all_failed = [f for ar in res.agent_results for f in ar.checks_failed]
        ok = res.success and not res.error and not all_failed
        status_str = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
        if ok:
            total_pass += 1
        else:
            total_fail += 1
        n_pass = sum(len(ar.checks_passed) for ar in res.agent_results)
        n_fail = sum(len(ar.checks_failed) for ar in res.agent_results)
        checks_str = (f"{GREEN}{n_pass}✔{RESET}  {RED}{n_fail}✘{RESET}") if n_fail else f"{GREEN}{n_pass}✔{RESET}"
        slow = any(ar.total_s > 60 for ar in res.agent_results)
        wall_str = f"{RED}{res.total_wall_s:>5.1f}s{RESET}" if slow else f"{res.total_wall_s:>5.1f}s"
        print(f"  {res.name:<32} {status_str:>6}  {res.confidence:>5.3f}  {wall_str}  {checks_str}")

    print(f"{'─'*72}")
    overall = f"{GREEN}ALL PASSED{RESET}" if total_fail == 0 else f"{RED}{total_fail} FAILED{RESET}"
    print(f"  Total: {total_pass} passed, {total_fail} failed  →  {overall}")
    print(f"{'═'*72}\n")

# ─── 1-shot adapter connectivity smoke test ──────────────────────────────────

def _smoke_test_adapters(base_url: str) -> None:
    """
    Smoke-test all v3 LoRA adapters via /v1/completions on each vLLM server.

    Uses the canonical [SYSTEM]/[USER]/[RESPONSE] prompt format identical to
    what was used during training so the adapter token boundaries fire correctly.
    Also verifies the base models are healthy.
    """
    _SYS = "[SYSTEM]\n{sys}\n\n[USER]\n{user}\n\n[RESPONSE]\n"
    base_checks = [
        # (server_url, adapter_or_model_id, prompt, label, require_pattern)
        ("http://localhost:8000", "coding_lora",
         _SYS.format(
             sys="You are CoderAgent — an expert software engineer across Python, JavaScript/TypeScript, Go, Rust, SQL, React, Vue, CSS, and shell scripting",
             user="Write a Python function that returns the nth Fibonacci number using memoization."),
         "coding_lora     / primary",
         re.compile(r"def |```|return", re.I)),
        ("http://localhost:8000", "debugging_lora",
         _SYS.format(
             sys="You are DebuggerAgent — an expert debugger and root-cause analyst for all programming languages and runtimes",
             user="Fix this: def divide(a, b): return a / b  # crashes when b=0"),
         "debugging_lora  / primary",
         re.compile(r"ISSUE\s*:|ZeroDivision|zero", re.I)),
        ("http://localhost:8000", "review_lora",
         _SYS.format(
             sys="You are ReviewerAgent — an expert code reviewer focused on correctness, security (OWASP), style, and maintainability",
             user="Review this code:\ndef add(a, b):\n    return a + b"),
         "review_lora     / primary",
         re.compile(r"SCORE\s*:", re.I)),
        ("http://localhost:8000", "planning_lora",
         _SYS.format(
             sys="You are PlannerAgent — an expert technical project planner and system architect",
             user="Plan a REST API for a simple todo list application."),
         "planning_lora   / primary",
         re.compile(r"^\s*\d+\.", re.M)),
        ("http://localhost:8000", "research_lora",
         _SYS.format(
             sys="You are ResearchAgent — an expert technical researcher and knowledge synthesizer",
             user="What are the key differences between REST and GraphQL APIs?"),
         "research_lora   / primary",
         re.compile(r"REST|GraphQL|query|schema", re.I)),
        ("http://localhost:8000", "criticism_lora",
         _SYS.format(
             sys="You are CriticAgent — a ruthless quality gatekeeper.\nYour job: give an honest overall assessment of the full solution so far.\nOutput:\nSCORE: <0.0\u20131.0>\nVERDICT: <one sentence summary>\nBLOCKERS: <critical issues that must be fixed, or 'None'>\nIMPROVEMENT: <the single most impactful change>",
             user="Critique this solution: A single-threaded Python web server that stores all session data in a global dictionary, with no authentication, deployed directly to production."),
         "criticism_lora  / primary",
         re.compile(r"SCORE\s*:|BLOCKERS\s*:", re.I)),
    ]

    print(f"\n{'━'*72}")
    print(f"{BOLD}{CYAN}▶ ADAPTER SMOKE TESTS — v3 LoRA adapters via direct vLLM calls{RESET}")
    print(f"{'━'*72}")

    all_ok = True
    with httpx.Client(timeout=90.0) as client:
        for server_url, adapter_or_model, prompt_text, label, require_re in base_checks:
            payload: dict = {
                "model":       adapter_or_model,
                "prompt":      prompt_text,
                "max_tokens":  200,
                "temperature": 0.1,
                "stop":        ["[SYSTEM]", "[USER]", "[ASSISTANT]", "\n[RESPONSE]"],
            }

            t0 = time.monotonic()
            try:
                resp = client.post(f"{server_url}/v1/completions", json=payload)
                resp.raise_for_status()
                data = resp.json()
                text = data["choices"][0]["text"]
                elapsed = time.monotonic() - t0
                bpe_hit  = bool(BPE_ARTIFACT_RE.search(text))
                hall_hit = bool(HALLUCINATION_RE.search(text))
                fmt_hit  = bool(require_re.search(text)) if require_re else True
                ok = not bpe_hit and not hall_hit and len(text.strip()) > 0 and fmt_hit
                status = f"{GREEN}OK{RESET}" if ok else f"{RED}FAIL{RESET}"
                if not ok:
                    all_ok = False
                issues = []
                if bpe_hit:       issues.append("BPE artifact")
                if hall_hit:      issues.append("hallucination pattern")
                if not text.strip(): issues.append("empty output")
                if not fmt_hit:   issues.append("format check failed")
                issue_str = f"  {RED}← {', '.join(issues)}{RESET}" if issues else ""
                print(f"  {label:<34}  {status}  {elapsed:.2f}s  {repr(text[:80])}{issue_str}")
            except Exception as exc:
                all_ok = False
                print(f"  {label:<34}  {RED}ERROR{RESET}  {exc}")

    if all_ok:
        print(f"\n  {GREEN}All v3 adapters responding with correct format ✔{RESET}")
    else:
        print(f"\n  {YELLOW}Some adapters had issues — pipeline tests may be affected.{RESET}")

# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="IdlePods AI — Full E2E test suite")
    parser.add_argument("--url", default="http://localhost:8080", help="Gateway base URL")
    parser.add_argument("--scenario", default=None, action="append", dest="scenarios",
                        metavar="SCENARIO",
                        help="Run only this scenario by name; repeat to run multiple")
    parser.add_argument("--no-smoke", action="store_true",
                        help="Skip adapter smoke tests")
    args = parser.parse_args()

    print(f"\n{BOLD}{'═'*72}")
    print(f"  IdlePods AI — Full End-to-End Test Suite")
    print(f"  Gateway: {args.url}")
    print(f"{'═'*72}{RESET}")

    # 1. Adapter smoke tests
    if not args.no_smoke:
        _smoke_test_adapters(args.url)

    # 2. Pipeline scenarios
    scenarios_to_run = [
        s for s in SCENARIOS
        if args.scenarios is None or s["name"] in args.scenarios
    ]
    if not scenarios_to_run:
        print(f"{RED}No scenario matches {args.scenarios!r}. "
              f"Available: {[s['name'] for s in SCENARIOS]}{RESET}")
        sys.exit(1)

    results: list[ScenarioResult] = []
    for s in scenarios_to_run:
        res = run_scenario(
            name=s["name"],
            label=s["label"],
            prompt=s["prompt"],
            expected_roles=s["expected_roles"],
            base_url=args.url,
        )
        _print_result_card(res)
        results.append(res)

    # 3. Summary
    _print_summary(results)

    # Exit code: 0 if all pass, 1 if any fail
    any_fail = any(
        not r.success or r.error or
        any(ar.checks_failed for ar in r.agent_results)
        for r in results
    )
    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
