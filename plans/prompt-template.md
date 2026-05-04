# Claude Template

## Discovery
    Core Rules:
    - Concise and precise. Short bullets preferred, but allow 1-2 sentence explanations when they prevent misdiagnosis of subtle logic, control flow, concurrency, or state issues.
    - Never restate CLAUDE.md.
    - Strongly prioritize token efficiency: ALWAYS start with and limit this response to lightweight searches (grep/symbol/tree) only.
    - Do NOT perform any full file reads automatically. If deeper analysis is required, list candidate files and ask the user for explicit permission.
    - Do not modify any files.
    Tasks:
    - 
    - Output exactly:
      - Bullet list of action plan + critical observations/risks (based only on lightweight searches)
      - Files identified so far (no edits)
      - Max one short clarifying question (use this to request permission on which files to read fully if needed)
    Begin.

## Planning
    Write @plans/current-task.md with exact structure (Do not output file diffs):
    Task Goal: [one sentence]
    Relevant Files:
    - file1.py
    - ...
    Steps:
    1. ...
    2. ...
    Constraints/Notes:
    - ...
    Keep entire file under 32 lines. Extremely concise and actionable.

## Execution
    HARD GATE: You may ONLY read or modify files explicitly listed in @plans/current-task.md. Never touch any other file. If you need anything outside the allowed list, stop immediately and ask.
    Every Read call MUST specify `offset` + `limit`. Target only the lines you need (function bodies, import blocks, the specific section under change).
    Never re-read a file or section already read in this run. Retain it in working memory.
    Use Grep (not Read) to locate symbols, call sites, and string literals.
    Use Glob for file discovery.
    Make file changes silently — no diffs, no code blocks, no before/after dumps in your output.

    Core Rules:
    - Ultra-concise. Short bullets only — one bullet per step. No prose, no headers, no commentary between bullets.
    - Never restate the plan or roadmap.
    - Use ONLY content from allowed files.
    - MAXIMUM TOKEN EFFICIENCY: On every read, ALWAYS use offset + limit to fetch ONLY the exact lines/sections needed. Never read an entire file unless the user explicitly approves it.
    - CRITICAL RULE: Never re-read any file (or any section of a file) that has already been read during this execution run. Retain all previously read information in context. Re-reading is strictly forbidden.
    - Tests written as part of a step must pass before the step is reported complete. If a test fails, fix the underlying step rather than weakening the assertion.

    Task:
        - Execute the exact plan in @plans/current-task.md. Follow its steps in strict sequential order with zero deviation or extra scope.
        - Plan E is now the stability + safety pass *and* the adapter-training-readiness gate. On Plan E completion the codebase MUST be in a state where rank-stabilised LoRA training can be started immediately with zero further code changes. The plan's steps 12–17 enforce this; treat them as load-bearing, not optional.
        - Adapter-training readiness — non-negotiable assertions to verify as you complete each related step:
            • rsLoRA / DoRA / QLoRA flexibility: `AdapterRecipe.peft_type` is the ONLY switch; selecting one is a `recipes.yaml` edit. The trainer logs `apply_recipe peft_type=... use_rslora=... ...` once per call. The startup gate (step 14) refuses `peft_type ∉ {lora,rslora,dora,qlora}` for any role except `consensus`, and refuses inconsistent flags (e.g. `peft_type=rslora` with `use_rslora=False`).
            • Tool-calling scope is config-driven via `settings.role_tools_enabled: Dict[str, List[str]]`. After Plan E, the literal `{"coder"}` MUST NOT appear in `nodes.py`, `pipeline.py`, `runner.py`, or `recorder.py` for routing/scope decisions — assert via a grep guard in the new test. Researcher / debugger / any future agent gains tool calling by adding tool names to `role_tools_enabled[<role>]` plus registering the tool in `runner.py:_TOOL_REGISTRY`. No graph or pipeline change required.
            • Per-role tool allowlist is honoured end-to-end: `build_tool_schemas(allowlist=...)` filters `_TOOL_REGISTRY` by name, called from `nodes.py` with `settings.role_tools_enabled.get(role)`.
            • Initial seed-data observability: `_load_curated_pairs` and the experience-pair load both emit a structured `seed_data_status` log with `role`, `curated_count` / `experience_count`, `path`, `exists`. No automatic data download from inside the trainer subprocess.
            • Recipe defaults: `recipes.yaml` keeps rsLoRA as default (r=32, alpha=64). Per-role and per-backend overrides remain a YAML edit.
            • Tool-call SFT is role-agnostic: `_load_sft_pairs` already iterates contributions by role and reads `tool_turns` without a coder-only branch. Verify and add the explanatory comment from step 16; do not introduce any role check.
            • Adapter-name decoupling: `_CAPABILITY_TO_ADAPTER` and `_CAPABILITY_TO_CURATED` are the only role↔filename maps. Onboarding a new role follows the four-step procedure documented in the new `recipes.yaml` comment block (step 17).

    Instructions:
    - Make all file changes silently (no diffs, no patches, no code blocks, no before/after).
    - For any read, always specify offset + limit.
    - Output exactly one bullet per completed step: "• Step N: [one-line description of what changed and why]".
    - For steps 14, 15, and the grep guard in step 12's test, also include the literal log-line / assertion shape in the bullet so it is obvious from the output that the readiness gate fired (still one line per step).
    - No reasoning, explanations, or commentary beyond those bullets.

    Begin.

## Compact
    /compact
    Full roadmap context is in @plans/current-roadmap.md.
    Summarize in 4-6 short bullets only:
    - Which plan just completed
    - Key files changed
    - Critical invariants now enforced (especially no hardcoded model names, registry usage)
    - Current state of the codebase
    - Any open items or next steps
    Stay extremely concise. Do not explain or add commentary.

## Verification
You are now in VERIFICATION MODE for the just-completed plan.
Full roadmap context is in @plans/current-roadmap.md (you have read it).
The plan you just executed is in @plans/current-task.md.

Task: Silently create or overwrite `scripts/healthcheck.py` with a complete, standalone, runnable verification script for this exact plan.

Rules:
- Read only files listed in @plans/current-task.md (use offset + limit).
- The generated `scripts/healthcheck.py` must be a self-contained Python script that can be run directly with `python scripts/healthcheck.py`.
- The script must perform every check from the plan's Steps, Constraints/Notes, and Tests section (pytest, grep for forbidden model names, registry validation, smoke tests, docker compose checks, etc.).
- Script output must be clear, structured, and Windows-friendly (PASS/FAIL per section with headers and summaries).
- Write the file silently (no diffs, no code blocks, no explanations).
- After writing the file, output exactly one bullet: "• Verification: Wrote scripts/healthcheck.py with full plan verification".

Begin.


