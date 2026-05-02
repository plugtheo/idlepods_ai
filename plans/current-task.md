=== PLAN D (revised): Plan B Residual Fix-Up + Plan-File Workflow ===


Task Goal: First close the Plan B gaps so adapters actually train and serve as OpenAI tool-call native (Phase 1 — load-bearing). Then wire `plans/current-task.md` into the orchestration layer so the planner ingests it on the first turn, coder/debugger update step status as they work, and the consensus/finalize path archives completed plans (Phase 2). Plan state is persisted in Redis under `task_state:{task_id}` for cross-session continuity. Phase 1 must land before Phase 2 because Phase 2's planner prompt extensions ride the same chat-completions + tools wire format that Phase 1 fixes for adapters.


Relevant Files:
Phase 1 (Plan B residual fix-up):
- `training/app/trainer_entry.py`
- `training/training/lora_trainer.py`
- `training/training/train_gpu_simple.py`
- `training/training/smoke_gate.py`
- `inference/app/backends/local_vllm.py`
- `orchestration/app/experience/sft_builder.py` (read-only — already correct, used as the canonical builder)
- `shared/contracts/training.py` (read-only — `AdapterRecipe`, `lookup_recipe` already correct)
- `shared/contracts/manifest_schema.py` (read-only — `HistoryEntry.recipe` already carries `sft_format`)
- `docker/compose.yml`
- `recipes.yaml`
- `scripts/healthcheck.py`
- `training/tests/test_trainer_entry_sft_format.py` (new)
- `training/tests/test_lora_trainer_messages_path.py` (new)
- `inference/tests/test_local_vllm_adapter_chat.py` (new)

Phase 2 (Plan-File workflow):
- `orchestration/app/plans/__init__.py` (new)
- `orchestration/app/plans/{schema.py, reader.py, writer.py}` (new)
- `orchestration/app/db/redis.py`
- `orchestration/app/graph/{nodes.py, state.py, pipeline.py, edges.py}`
- `orchestration/app/routes/run.py`
- `orchestration/app/tools/runner.py` (read-only — `_safe_path` reused)
- `shared/contracts/{orchestration.py, agent_prompts.py}`
- `plans/current-task.md` (canonical bootstrap source)
- `plans/archive/` (new dir, `.gitkeep`)
- `orchestration/tests/test_plans_reader_writer.py` (new)
- `orchestration/tests/test_planner_with_plan.py` (new)
- `orchestration/tests/test_plan_redis_persist.py` (new)


Steps — PHASE 1 (Plan B residual fix-up):

1. `training/app/trainer_entry.py`:
   - Delete `_format_messages_as_prompt` and the ChatML-string SFT-pair construction in `_load_sft_pairs`.
   - Replace the per-contribution emit with a call to `orchestration.app.experience.sft_builder.build_sft_pair(contribution, recipe, role, system_prompt, user_prompt)` so the on-disk record shape is `{"messages": [...]}` for `openai_messages` and `{"prompt","completion"}` for `legacy_response_marker`.
   - The `recipe` is already resolved at line ~381 via `lookup_recipe(backend, role)`; thread it down to `_load_sft_pairs` and `_load_curated_pairs`.
   - Curated-pair wrap (currently `_load_curated_pairs` lines 308-313) must also branch on `recipe.sft_format`: emit `{"messages":[{"role":"system",...},{"role":"user",...},{"role":"assistant","content":response}]}` for `openai_messages`; keep the legacy `[SYSTEM]/[USER]/[RESPONSE]` wrap only for `legacy_response_marker`.
   - Fallback-prompt path (`_make_fallback_prompt`) likewise: emit messages-list shape under openai_messages.

2. `training/training/lora_trainer.py`:
   - In `LoRATrainer.train` (lines 619-639), remove the unconditional wrap with `[SYSTEM]/[USER]/[RESPONSE]`. New control flow: if `record` has `messages` → leave it untouched (TRL ≥ 0.24 + `processing_class=tokenizer` accepts conversational records and applies the model's chat template); elif `prompt`+`completion` already present → leave untouched; else fall back to the legacy marker wrap (only when `recipe.sft_format == "legacy_response_marker"`).
   - In `_train_unsloth`, when records carry `messages`, do NOT inject `record["completion"]`; SFTTrainer handles message records natively. Pass through `dataset` either as `HFDataset.from_list(dataset)` of message records or of prompt/completion records — TRL detects shape per-row.
   - In `_promote_to_active` (line 263), replace the hardcoded `"sft_format": "chatml"` literal with the actual `recipe.sft_format` value. Thread `recipe` (or just `recipe.sft_format`) into `_promote_to_active` from the trainer_entry caller (which already has `recipe` and writes `new_meta["history"][-1]["recipe"] = recipe.model_dump()` at trainer_entry.py:506 — extend the same pattern to the persisted history `recipe` dict and read it here).
   - Confirm `apply_recipe` is called for every `recipe.peft_type` in {`lora`,`rslora`,`dora`,`qlora`} and that `qlora` implies `load_in_4bit=True` (already enforced at validate_adapter.py:220 — mirror in trainer).

3. `training/training/train_gpu_simple.py` (bootstrap path):
   - Either align with the new flow (load `recipe = lookup_recipe(backend, role)`, build records via `build_sft_pair`, save shape per `recipe.sft_format`) — OR retire the script and add a `--bootstrap` flag to `trainer_entry.py` that bypasses the experience JSONL and reads `training_data_curated/*.jsonl` only.
   - If retiring, update any caller (compose, scripts, README) and remove `train_gpu_simple.py`. Default: align in place to keep the bootstrap entrypoint stable.

4. `inference/app/backends/local_vllm.py`:
   - Tag the adapter call path on the manifest: read `Manifest` → adapter history's most-recent `recipe.tool_call_style` (default `"openai_native"` if absent). Cache the value next to `_adapter_fallback_counts`.
   - When `tool_call_style != "none"` (the default for new adapters), route adapter calls through `/v1/chat/completions` (the same path used today for base-model calls) with `tools=` propagated from `request.tools`, instead of the legacy `/v1/completions` + `_build_adapter_prompt` path. The `effective_model` (e.g. `"coding_lora"`) goes into the `model` field; vLLM activates the LoRA. Tool-call output is parsed by the existing Hermes parser configured in compose.yml.
   - Keep `_build_adapter_prompt` and the `/v1/completions` branch only as a fallback for adapters whose manifest entry has `tool_call_style == "none"` OR `sft_format == "legacy_response_marker"`. Tag the branch decision in the log line.

5. `training/training/smoke_gate.py`:
   - Verify the smoke-gate path uses `/v1/chat/completions` with `tools=` when the recipe is openai_native (existing `test_smoke_gate_tool_call.py` already covers the contract — make the implementation conform if it does not).
   - The non-tool-role smoke (planner/researcher/critic/consensus) keeps using whichever endpoint matches their adapter's recorded `sft_format`.

6. `recipes.yaml`:
   - Tighten defaults for the new MVP base (Qwen3-8B-AWQ): `peft_type: rslora`, `use_rslora: true`, `r: 32`, `alpha: 64`, `dropout: 0.05`. Keep coder/debugger overrides; consensus stays `peft_type: none`.
   - Add a comment block describing the per-backend override seam (`by_backend_role: "primary:coder": {...}`).

7. `docker/compose.yml`:
   - Mount `../recipes.yaml:/config/recipes.yaml:ro` on **all three** services (orchestration, inference, training) — same pattern as the existing models.yaml mounts at lines 96, 170, 225.
   - Add `RECIPES_YAML_PATH=/config/recipes.yaml` to each service's `environment` block alongside the existing `MODELS_YAML_PATH` lines (85, 181, 236). Without this, `_find_recipes_yaml()` falls back to repo-root and silently uses defaults inside containers.

8. `scripts/healthcheck.py`:
   - Replace the assertion at line 415 (`'"chatml"' in content` for `lora_trainer.py`) with: assert that the trainer reads `recipe.sft_format` and writes it to the manifest history entry verbatim. Also assert that no `[RESPONSE]\n` literal is built unconditionally in `lora_trainer.py` (legacy wrap must be guarded by `recipe.sft_format == "legacy_response_marker"`).
   - Add a section: assert `recipes.yaml` is mounted in compose.yml for orchestration, inference, training, and that each has `RECIPES_YAML_PATH` env.


Steps — PHASE 2 (Plan-File workflow):

9. `orchestration/app/plans/schema.py` (~80 LOC):
   ```python
   class PlanStep(BaseModel):
       id: str                      # auto-assigned "step-N" or LLM-supplied
       description: str
       status: Literal["pending","in_progress","done","blocked"] = "pending"
       owner_role: Optional[str] = None
       evidence: str = ""
       files_touched: List[str] = []
       tools_used: List[str] = []
       depends_on: List[str] = []   # reserved for DAG-aware routing (enhancement c)

   class Plan(BaseModel):
       goal: str
       steps: List[PlanStep]
       created_at: datetime
       updated_at: datetime
   ```
   - Validation: `id` unique within `steps`; status transitions forward-only except `pending→blocked` and `blocked→in_progress` (enforced in writer).

10. `orchestration/app/plans/reader.py` (~120 LOC):
    - Markdown-section parser:
      - `## Task Goal` → `Plan.goal`
      - `## Steps` → numbered list → `Plan.steps[*].description`
      - `## Constraints/Notes` → kept in raw markdown for human readers
    - Auto-assigns `id="step-1"`, `"step-2"`, ... and `status="pending"`.
    - Path resolution uses `_safe_path` from `orchestration/app/tools/runner.py`. Additionally clamp the resolved path under `plans/` so the planner cannot accidentally archive over user files (`raise ToolError` if the resolved relative path is not rooted at `plans/`).

11. `orchestration/app/plans/writer.py` (~100 LOC):
    - `write_plan_atomic(path: Path, plan: Plan) -> None`: render plan back to markdown (preserve human formatting), write to `<path>.tmp`, then `os.replace`. Uses `filelock.FileLock(str(path) + ".lock", timeout=30)` — same library/pattern as `shared/manifest.write_manifest_locked`. Do NOT import the manifest helper itself; the per-task lock is independent.
    - `validate_transition(old: Plan, new: Plan)`: reject illegal status transitions; called by writer.

12. `orchestration/app/db/redis.py`:
    - Add `set_plan(task_id: str, plan: Plan, ttl_s: int)` and `get_plan(task_id: str) -> Optional[Plan]`. Key: `task_state:{task_id}` (does not collide with existing `session:v2:`, `fps:v2:`, `snippets:v2:` namespaces).
    - TTL default `7 * 86400` via `ORCHESTRATION__TASK_STATE_TTL_S` (default 604800).
    - On Redis miss, log `"task_state degraded"` once per task and fall back to file read.

13. `orchestration/app/graph/state.py`:
    - Append to `AgentState`:
      ```python
      plan: Optional[Plan]
      plan_changed: bool          # set True when any node mutates plan; finalize writes back
      current_step_id: Optional[str]  # planner sets, coder/debugger respect
      ```

14. `orchestration/app/routes/run.py`:
    - On request entry: keep the existing `task_id = getattr(request, "task_id", None) or session_id` fallback at line 82 unchanged (do NOT make `task_id` mandatory — that would break existing callers). When `request.task_id` was not supplied (i.e. it equals `session_id`), the plan is treated as **ephemeral**: read it for the planner's bootstrap turn, but do NOT call `set_plan` and do NOT archive on convergence. Log `"plan ephemeral — task_id not supplied"` once.
    - When `request.task_id` IS supplied: try `get_plan(task_id)`; on miss, read `request.plan_path` (default `plans/current-task.md`).
    - On finalize (`quality_converged=True` OR explicit checkpoint event): call `set_plan(task_id, plan, ttl)` AND `write_plan_atomic(plan_path, plan)`. Also archive a snapshot to `plans/archive/{task_id}-{YYYYMMDD-HHMMSS}.md`.
    - SSE: emit `event: task_state` with `{task_id, completed_steps, total_steps}` whenever `plan_changed` flips.

15. `orchestration/app/graph/nodes.py`:
    - `planner_node`:
      - If `state["plan"]` is None or empty → produce a fresh plan, return delta with `plan` and `plan_changed=True`.
      - Else → mark the next `pending` step as `in_progress`, set `state["current_step_id"]`, and emit guidance referencing it. Do NOT regenerate the plan when one already exists.
      - The planner runs in chat-completions mode. Inject the plan-step context as an **additional `system` message** (not by string-concatenation into the existing system prompt) so it remains compatible with Phase 1's tool-call wire format and the Hermes parser.
    - `coder_node` / `debugger_node`:
      - Receive `current_step_id` via the system-message append (see step 16).
      - When the tool round completes successfully (no errors in last 3 tool entries), set the step's `status="done"`, append `tools_used` and `files_touched` (extracted from successful `write_file` / `read_file` calls), set `plan_changed=True`.
      - When the LLM emits `[BLOCKED:<reason>]` in its output, set `status="blocked"`, record `evidence=<reason>`.
    - `consensus_node`: ensure all `pending` steps are either `done` or explicitly `blocked` before emitting the final answer; if any remain `pending`, append a "remaining work" section to the consensus output.

16. `shared/contracts/agent_prompts.py`:
    - Append to the planner prompt: a strict JSON schema for plan steps that the planner emits when no plan exists. Runtime parses this into `PlanStep` objects.
    - Provide a constant `PLAN_STEP_SYSTEM_TEMPLATE` (single line — `"You are working on plan step {current_step_id}: {current_step_description}. Use the provided tools to complete this step, then signal done."`) that nodes inject as an **extra system message** in the request, only when `current_step_id` is set. The base coder/debugger prompts remain byte-for-byte unchanged.
    - Byte-for-byte invariant for non-tool roles (reviewer/critic/consensus) holds — those prompts are unchanged.

17. `shared/contracts/orchestration.py`:
    - `OrchestrationRequest` gains `plan_path: Optional[str] = "plans/current-task.md"` and the existing `task_id: Optional[str] = None` is preserved. When `task_id is None`, the plan is ephemeral (in-memory only — not persisted to Redis or markdown), per step 14.

18. `orchestration/app/graph/pipeline.py` and `edges.py`:
    - No structural change. Plan reading happens in `routes/run.py` before the graph is invoked; writing happens after the graph returns.


Tests:

Phase 1:
- `training/tests/test_trainer_entry_sft_format.py`: with `recipe.sft_format="openai_messages"`, asserts `_load_sft_pairs` emits records with a `messages` key (list of role/content dicts) and **no** `instruction` key. With `legacy_response_marker`, asserts `prompt`/`completion` shape.
- `training/tests/test_lora_trainer_messages_path.py`: feeds a record with `messages` to `LoRATrainer.train` (mock `SFTTrainer`); asserts no `[SYSTEM]`/`[USER]`/`[RESPONSE]` wrap is added and that `recipe.sft_format` is propagated to the manifest history entry built by `_promote_to_active`.
- `inference/tests/test_local_vllm_adapter_chat.py`: with a manifest where the adapter's history has `recipe.tool_call_style="openai_native"`, asserts adapter calls go to `/v1/chat/completions` with `tools=` and that `_build_adapter_prompt` is NOT invoked. With `tool_call_style="none"`, asserts the legacy `/v1/completions` + adapter-prompt path IS used.

Phase 2:
- `orchestration/tests/test_plans_reader_writer.py`: round-trip a fixture markdown file → `Plan` → markdown; assert byte equality after a no-op edit.
- `orchestration/tests/test_planner_with_plan.py`: fake inference returns a plan JSON; assert `state["plan"]` populated correctly and the second invocation does NOT regenerate. Also assert the plan-step injection is an extra system message, not a concatenation into the base planner prompt.
- `orchestration/tests/test_plan_redis_persist.py`: fakeredis fixture; round-trip and TTL eviction. Also covers the ephemeral path (no `task_id` → no Redis write).


Constraints/Notes:

Phase 1:
- The fix-up is sequenced so that step 1 (trainer_entry) and step 2 (lora_trainer) land together — the trainer would fail on `messages`-shaped records if step 2 has not landed. Land step 4 (inference adapter chat path) only after a smoke-tested adapter trained under the new flow exists; until then, set the new adapter's manifest history `recipe.tool_call_style="none"` so the legacy inference branch keeps it serving (zero-downtime cutover).
- Do NOT delete `_build_adapter_prompt` or the legacy `/v1/completions` branch — they must remain reachable for any historic adapter whose manifest history entry has `sft_format="legacy_response_marker"` (e.g. anything migrated by `scripts/migrate_manifest.py`, which seeds `sft_format="legacy"`).
- The `INFERENCE__ACCEPT_LEGACY_BACKEND_NAMES` shim and the `"qwen" → "primary"` alias map remain untouched in this plan (deferred to a later cleanup).

Phase 2:
- Redis is source of truth during a live task; markdown is bootstrap + checkpoint only — do not write back to markdown on every step (would churn the working tree).
- Writeback to markdown happens only at convergence or on explicit `checkpoint` SSE input — once per task lifecycle, typically.
- `task_id` MUST be supplied for plan persistence. Existing callers without `task_id` continue to work (ephemeral plan, no persistence) — preserves the current `routes/run.py:82` fallback to `session_id`.
- `plan_path` resolution uses `_safe_path` AND additionally clamps to `plans/` (see step 10) — same denylist (`.git`, `.env`, `secrets/`).
- Reader → writer round-trip MUST be idempotent. Fixture-test this explicitly; a non-idempotent round-trip will silently rewrite the file every step.
- Plan-update logic is *opportunistic* — if the LLM doesn't follow the format, the step status stays `in_progress` and the human can fix it manually. Do NOT add hard validation that crashes the run.
- Plan-step prompt injection is an **extra system message** (never concatenated into the base prompt) so the byte-for-byte invariant for tool-role base prompts holds and Phase 1's Hermes tool-call parser keeps working.
- ENHANCEMENT HOOK (a): a `/v1/plan/checkpoint` REST endpoint can force a markdown writeback mid-run for long tasks; implement when self-paced /loop runs become routine.
- ENHANCEMENT HOOK (b): a planner-budget cap (max steps per plan, default 20) prevents the LLM from emitting 200-step manifests; default off.
- ENHANCEMENT HOOK (c): `plan.steps[*].depends_on` is reserved in `PlanStep` (step 9) for future DAG-aware step routing.
---
