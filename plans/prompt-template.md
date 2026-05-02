# Claude Template

## Discovery
    Core Rules:
    - Concise and precise. Short bullets preferred, but allow 1-2 sentence explanations when they prevent misdiagnosis of subtle logic, control flow, concurrency, or state issues.
    - Never restate CLAUDE.md.
    - Strongly prioritize token efficiency: ALWAYS start with and limit this response to lightweight searches (grep/symbol/tree) only.
    - Do NOT perform any full file reads automatically. If deeper analysis is required, list candidate files and ask the user for explicit permission.
    - Do not modify any files.
    Tasks:
    - Replace static --lora-modules with runtime adapter lifecycle registry + dynamic load + smoke gate (after trainer writes checkpoint: run smoke test with canonical prompts per role, validate structure/no artifacts, only then dynamic load + registry update, keep previous adapter until verified)
    - Build adapter version registry with rollback (persistent manifest with checkpoint path, timestamp, smoke results, eval scores, active/retired status)
    Instructions:
    - Begin with lightweight searches ONLY (grep, symbol, tree).
    - Identify the smallest set of potentially relevant files from these searches.
    - If lightweight results are insufficient to fully understand control flow, mutex logic, queuing, or state management for accurate diagnosis, list the specific additional files (in priority order, max 5) you believe need full reads.
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

    CONTEXT ENFORCEMENT: Full roadmap context is in @plans/current-roadmap.md. You have read it. Always obey its cross-cutting principles: plug-and-play via models.yaml (no hardcoded "qwen", "deepseek", or "mistral" anywhere), native OpenAI tool calling is primary, explicit tool_executor node + sandboxed runner must be preserved, simple ReAct + human-in-the-loop core is mandatory.

    Core Rules:
    - Ultra-concise. Short bullets only.
    - Never restate the plan or roadmap.
    - Use ONLY content from allowed files.
    - MAXIMUM TOKEN EFFICIENCY: On every read, ALWAYS use offset + limit to fetch ONLY the exact lines/sections needed. Never read an entire file unless user explicitly approves it.
    - CRITICAL RULE: Never re-read any file (or any section of a file) that has already been read during this execution run. Retain all previously read information in context. Re-reading is strictly forbidden.

    Task: Execute the exact plan in @plans/current-task.md. Follow its steps in strict sequential order with zero deviation or extra scope.

    Instructions:
    - Make all file changes silently (no diffs, no patches, no code blocks, no before/after).
    - For any read, always specify offset + limit.
    - Output exactly one bullet per completed step: "• Step N: [one-line description of what changed and why]".
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


