# FULL PLAN OVERVIEW

After Plan 1 (Tool-Use MVP), the codebase has evolved further than the original Plans 2–7 anticipated. The DeepSeek + Mistral split is gone; the entire stack now runs exclusively against a single Qwen/Qwen3-14B vLLM server on a 3090, with `INFERENCE__QWEN_*` settings, a single `vllm-qwen` compose service, and `role_model_family` set to `"qwen"` for every role. Native OpenAI tool-calling is wired end-to-end: `GenerateRequest.tools`, `Message.tool_calls`/`tool_call_id`, `tool_executor_node`, and the OpenAI-format prompt assembly in `nodes._build_messages` are already live. The plain-text `<<TOOL>>{...}<<END>>` markers from Plan 1 are no longer the primary path — they survive only as a graceful fallback for non-tool-trained adapters. `run_command` is in the registry. Qwen3 thinking mode is disabled by default.

The original Plans 2–7 had three big themes: (a) **plug-and-play backends** (Plan 2), (b) **flexible adapter recipes + extended manifest** (Plans 3 + 4) and (c) **operational hardening + plan-file workflow + nice-to-haves** (Plans 5 + 6 + 7). Those goals are still all valid, but the framing must change. We no longer need a multi-backend enum — we need a *registry of one (today), N (tomorrow)* that is read from `models.yaml` so swapping Qwen for any future model never touches Python. We no longer need DeepSeek/Mistral conditionals (Metaspace vs ByteLevel pre-tokenizer hacks, capability→family lookup) — those must be deleted as dead code that contradicts the "plug-and-play, no hardcoded model names" invariant. Adapter retraining must produce LoRAs that are themselves OpenAI-tool-call native, so SFT pairs must capture `tool_calls`/`tool_results` rounds in the OpenAI message format, not the legacy `[SYSTEM]/[USER]/[RESPONSE]` text wrapper.

The new sequence is therefore restructured around the current reality. **Plan A** rips out every remaining hardcoded `"deepseek"`/`"mistral"`/`"qwen"` literal and the `CAPABILITY_TO_FAMILY` map, replacing them with a single `models.yaml` registry mounted into all three services — this is the load-bearing prerequisite for everything else and is mostly a deletion exercise now that only Qwen runs. **Plan B** delivers `AdapterRecipe` + tool-call-native SFT — *one* unified plan that combines the old Plan 3 (recipe flexibility) with adapter retraining for OpenAI tool calls, since they share the SFT data path; the recipe abstraction is the natural seam to keep tool-call training format swappable for future models. **Plan C** extends the manifest with full provenance and adds the cross-platform shared read/write helper (filelock) plus a one-shot migration. **Plan D** adds the plan-file workflow (orchestration ↔ `plans/current-task.md`), keeping the ReAct/human-in-loop core simple. **Plan E** is the stability + safety pass: cross-platform locking on JSONL, rotation+cursor, scheduler timeout/heartbeat, and auto-rollback on adapter fallback. **Plan F** is the nice-to-haves backlog (embedding router, distillation, ChromaDB tiers, prefix caching, async subprocess, cancellation propagation).

Two cross-cutting principles govern all six plans:

1. **No model-name literals in Python.** Every reference to `"qwen"`, `"deepseek"`, or `"mistral"` is removed; identity flows through `models.yaml` keyed by an opaque `backend` name. Tests assert the absence of such literals via a grep guard.
2. **Tool-calling lives at two layers, both swappable.** The wire format (OpenAI `tool_calls` in `Message`/`GenerateResponse`) is invariant across models. The *training* format is captured per-recipe so retraining a future model with a different tool-call convention (e.g. Hermes, Llama-3, or a custom format) is a recipe edit, not a code edit. The explicit `tool_executor` node and its sandboxed runner stay exactly as-is — they are the simple, mandatory ReAct + human-in-loop core.

The six plans below are ordered so each unblocks the next: A (plug-and-play) → B (recipes + tool-call SFT) → C (manifest + migration) → D (plan-file workflow) → E (stability/safety) → F (nice-to-haves).

────────────────────────────────────────────────────────────

=== PLAN A: Backend Registry — Eliminate Hardcoded Model Names ===

Task Goal: Replace every hardcoded `"qwen"`/`"deepseek"`/`"mistral"` literal and the `CAPABILITY_TO_FAMILY` map with a `models.yaml`-driven backend registry that all three services load at startup, so adding or replacing a base model is a YAML edit only.

Relevant Files:
- `models.yaml` (new, repo root, mounted into all three containers)
- `shared/contracts/models.py` (new)
- `shared/contracts/inference.py`
- `shared/grpc_stubs/inference.proto` (regen via `scripts/generate_protos.py`) and `shared/grpc_stubs/inference_pb2*.py` (regenerated artefacts)
- `inference/app/{main.py, config/settings.py, backends/factory.py, backends/local_vllm.py, backends/remote_vllm.py, routes/generate.py, routes/adapters.py, grpc/server.py}`
- `orchestration/app/{config/settings.py, clients/inference.py, clients/inference_grpc.py, graph/nodes.py}`
- `training/app/{config/settings.py, trainer_entry.py, trainer_wrapper.py, utils/trainer_launcher.py}`
- `training/training/{lora_trainer.py, train_gpu_simple.py, validate_adapter.py}`
- `scripts/{seed_adapter_metadata.py, patch_tokenizer.py, show_adapters.py, e2e_test.py, eval_adapters.py, smoke_test_adapters.py, test_critic.py}`
- `docker/compose.yml`
- `inference/tests/{test_factory.py, test_generate_route.py, test_local_vllm.py, test_settings.py, test_streaming_backends.py, test_streaming_route.py, test_grpc_server.py, test_adapter_registry.py}`
- `orchestration/tests/{test_nodes.py, test_inference_grpc_client.py, test_nodes_token_queue.py}`
- `shared/tests/test_contracts.py` and a new `shared/tests/test_no_model_literals.py`

Steps:
1. `models.yaml` (~25 LOC):
   ```yaml
   default_backend: primary
   backends:
     primary:
       served_url: http://vllm-qwen:8000
       model_id: Qwen/Qwen3-14B
       max_model_len: 4096
       quantization: fp8
       chat_template_supports_tools: true
       thinking_default: false
       backend_type: local_vllm  # local_vllm | remote_vllm
       auth_token: ""            # only used by remote_vllm
       ssl_verify: true
   ```
   The opaque key `primary` replaces the model-name string across the codebase. `default_backend` is the fallback when a per-role mapping is absent.

2. `shared/contracts/models.py` (~80 LOC, new):
   - Pydantic `BackendEntry { served_url, model_id, max_model_len, quantization, chat_template_supports_tools, thinking_default, backend_type, auth_token, ssl_verify }`.
   - Pydantic `ModelsRegistry { default_backend: str, backends: Dict[str, BackendEntry] }`.
   - `load_registry(path: str = os.environ.get("MODELS_YAML_PATH", "/config/models.yaml")) -> ModelsRegistry` with `lru_cache(maxsize=1)`; raises `RuntimeError` when the file is missing or the `default_backend` key is not in `backends`.
   - `get_backend_entry(name: str | None) -> BackendEntry` — returns `default_backend` entry when `name is None`.
   - Public `__all__` exporting `BackendEntry`, `ModelsRegistry`, `load_registry`, `get_backend_entry`.

3. `shared/contracts/inference.py`:
   - Rename `model_family: str` → `backend: str` on `GenerateRequest` and `GenerateResponse`. Add a Pydantic `field_validator("backend")` that calls `load_registry()` and raises `ValueError` if the name is unknown.
   - Keep an internal `_LEGACY_BACKEND_ALIASES = {"qwen": <default_backend>, "deepseek": <default_backend>, "mistral": <default_backend>}` ONLY for one minor release, gated behind `INFERENCE__ACCEPT_LEGACY_BACKEND_NAMES=true` (default false). When an alias is hit, log `logger.warning("Legacy backend name %r → %r", raw, mapped)` exactly once per process. Do NOT use this shim from new code.

4. Proto bump (`shared/grpc_stubs/inference.proto`):
   - Rename `string model_family = 1;` → `string backend = 1;` on `GenerateRequest` (keep field tag 1 for wire compatibility — it is a string, the field name is metadata only). Bump the `PROTO_SCHEMA_HASH` constant. Re-run `python scripts/generate_protos.py` and commit the regenerated `inference_pb2.py` / `inference_pb2_grpc.py`.

5. `inference/app/config/settings.py`:
   - Delete `qwen_backend`, `qwen_url`, `qwen_model_id`, `qwen_auth_token`, `qwen_ssl_verify`. Replace with `models_yaml_path: str = Field(default="/config/models.yaml")` and `accept_legacy_backend_names: bool = Field(default=False)`.
   - Keep `request_timeout_seconds`, `http_max_*`, `grpc_default_*`, `port`, `grpc_port`, `grpc_shutdown_grace_seconds`, `mode` (delete `mode` if unused after sweep — verify with grep first).

6. `inference/app/backends/factory.py`:
   - Delete `CAPABILITY_TO_FAMILY` and `get_backend_for_capability`.
   - Rewrite `get_backend(backend_name: str)` to look up `models.yaml` via `load_registry()` and pick `LocalVLLMBackend` vs `RemoteVLLMBackend` from `entry.backend_type`. The factory still caches singletons in `_backends: dict[str, InferenceBackend]`.
   - `bootstrap_adapters(manifest_path)` now reads `entry["backend"]` (NOT `capability`) from each manifest record; for v1 entries with no `backend` field, fall back to `registry.default_backend`. Log a deprecation warning per legacy entry.

7. `inference/app/backends/local_vllm.py`:
   - Delete every `if "deepseek" in self.base_model:` / `if "mistral" in …` branch — there is exactly one (the Metaspace/ByteLevel pre-tokenizer fix). Since Qwen needs neither hack, the entire branch is dead code under the new invariant. Remove it; if a future model needs a tokenizer hack, surface it via `BackendEntry.tokenizer_pre_tokenizer: Optional[str]` and apply it in `_load_tokenizer` — do NOT inline a string match.
   - Constructor signature becomes `__init__(self, backend_name: str, entry: BackendEntry)`.

8. `inference/app/backends/remote_vllm.py`: same constructor refactor; remove all model-name literals.

9. `inference/app/routes/generate.py` + `inference/app/routes/adapters.py` + `inference/app/grpc/server.py`:
   - Replace `("deepseek", "mistral")` / `"qwen"` loops with `for name in load_registry().backends:`.
   - In gRPC server, map the proto `backend` field directly without translation.

10. `orchestration/app/config/settings.py`:
    - Rename `role_model_family` → `role_backend`. Keep type `Dict[str, str]`. Default factory returns `{role: "primary" for role in (...)}` *only if* `role` is explicitly listed; otherwise `_run_agent_node` falls back to `load_registry().default_backend`.
    - Delete the hardcoded `"qwen"` literals from the default factory. Use the registry's default backend symbolically: read `os.environ.get("MODELS_YAML_PATH")` lazily inside the factory.
    - `model_context_len` stays as-is (caller of the inference service still needs to budget tokens). Add a comment that this MUST equal the `max_model_len` from the matching `models.yaml` entry; add a startup assertion in `orchestration/app/main.py` that compares them and warns on mismatch.

11. `orchestration/app/clients/inference.py` + `inference_grpc.py`:
    - Rename `model_family` keyword arg → `backend` everywhere. Update both HTTP and gRPC paths.

12. `orchestration/app/graph/nodes.py`:
    - Replace `model_family=settings.role_model_family.get(role, "qwen")` with `backend=settings.role_backend.get(role) or load_registry().default_backend` (line 340).
    - Remove the `"qwen"` string literal entirely.

13. `training/app/config/settings.py` + `trainer_entry.py` + `trainer_wrapper.py`:
    - `_base_model_for(capability)` is replaced by `_base_model_for(role)` which reads `role_backend` (mirrored from orchestration via the same `models.yaml`) and returns `entry.model_id`.
    - Delete every `"deepseek-coder-6.7b"` / `"mistralai/Mistral-…"` literal. Default `LoRATrainer.__init__` parameter `base_model` becomes `Optional[str] = None` and is resolved from the registry when None.

14. `training/training/lora_trainer.py`:
    - Lines 658–661 + 774–782: delete the `_is_deepseek = "deepseek" in self.base_model.lower()` branch and all conditional pre-tokenizer / tokenizer.json writeback logic. (Future per-model tokenizer hacks will be expressed via `recipe.tokenizer_pre_tokenizer` once Plan B lands.)
    - Lines 172, 521, 864, 1030: delete every `"deepseek"` / `"mistral"` / `"deepseek-coder-6.7b"` literal. `model_family` field in `new_meta` becomes `backend` (string from registry) — coordinate the rename with Plan C's manifest schema; for this plan, write `backend=registry.default_backend` for now.

15. `training/training/train_gpu_simple.py`:
    - Strip module-level base-model constants. Resolve base model per role from `_base_model_for(role)`. Delete the DeepSeek/Mistral capability lookup table at the top of the file.

16. `scripts/*.py`:
    - `seed_adapter_metadata.py`, `patch_tokenizer.py`, `show_adapters.py`, `e2e_test.py`, `eval_adapters.py`, `smoke_test_adapters.py`, `test_critic.py`: replace any `("deepseek","mistral")` lists with `list(load_registry().backends)`. `patch_tokenizer.py` becomes a no-op for backends that don't declare `tokenizer_pre_tokenizer`.

17. `docker/compose.yml`:
    - Hoist hardcoded `--model Qwen/Qwen3-14B` and `--max-model-len 4096` out of the service's `command:` block via `${VLLM_MODEL_ID}` / `${VLLM_MAX_MODEL_LEN}` env interpolation, sourced from a generated `.env.vllm` file produced by a tiny `scripts/render_compose_env.py` that reads `models.yaml` and writes one env-var per backend. The compose service name `vllm-qwen` becomes `vllm-primary`. Mount `models.yaml` read-only into all three Python services at `/config/models.yaml`.

18. Tests:
    - `shared/tests/test_no_model_literals.py` (new): walk every `*.py` under `inference/`, `orchestration/`, `training/`, `shared/`, `scripts/` (exclude `tests/`, `.claude/`, `__pycache__`) and assert that no file contains the substrings `"qwen"`, `"deepseek"`, `"mistral"` (case-insensitive). Allow-list: this test file itself, `shared/contracts/models.py` if it contains the alias map's *values* (it shouldn't — alias map keys are model names, values are backend names like `"primary"`). Failing this test = the plan is not done.
    - Update every existing test that constructs a `GenerateRequest(model_family=…)` to use `backend=…`.
    - `inference/tests/test_factory.py`: parameterise over `models.yaml` fixture instead of hardcoded `("deepseek","mistral")`.

Constraints/Notes:
- Removing the DeepSeek Metaspace→ByteLevel hack is safe ONLY because Qwen3 ships a correct ByteLevel tokenizer out of the box. Verify this once with a smoke run of `validate_adapter.py` against the current Qwen adapter directory before deleting the branch — do NOT take this on faith.
- Compose interpolation via `.env.vllm` is preferred over Jinja-rendered compose.yml because Docker Compose natively supports `${VAR}` substitution; rendering compose.yml itself adds a build step.
- The proto field rename keeps tag `1` — wire format is unchanged. Old clients sending `model_family` over JSON HTTP will break (intentionally) unless `accept_legacy_backend_names=true`. Document this in the commit message.
- The `_LEGACY_BACKEND_ALIASES` shim is a one-release deprecation bridge — schedule its deletion via `/schedule` after this plan ships.
- DO NOT add Qwen-specific tokenizer hacks back. If they are needed, they go in `BackendEntry`, never in source code.
- ENHANCEMENT HOOK (a): once a second backend is added (e.g. `secondary: { model_id: "meta-llama/Llama-3-…" }`), the `role_backend` map can route `coder→primary, planner→secondary` with zero Python edits.
- ENHANCEMENT HOOK (b): `BackendEntry.tokenizer_pre_tokenizer: Optional[Literal["bytelevel","metaspace"]]` field is reserved for per-backend pre-tokenizer override; not implemented now because Qwen needs neither.
- ENHANCEMENT HOOK (c): `BackendEntry.chat_template_supports_tools` is read by Plan B to decide whether to use native OpenAI tool_calls or fall back to the marker format during SFT pair construction.
---

=== PLAN B: AdapterRecipe + Native OpenAI Tool-Call SFT ===

Task Goal: Introduce an `AdapterRecipe` abstraction that captures both the PEFT knobs (LoRA / RSLoRA / QLoRA / DoRA, rank, alpha, target_modules) AND the SFT message format, so retraining each agent's adapter against OpenAI native `tool_calls` (today) or any future tool-call convention (tomorrow) is a `recipes.yaml` edit, not a code edit. Coder + debugger get retrained adapters that emit native `tool_calls` correctly; non-tool roles keep their existing format.

Relevant Files:
- `recipes.yaml` (new, repo root, mounted at `/config/recipes.yaml`)
- `shared/contracts/training.py`
- `shared/contracts/messages.py` (new — shared OpenAI message construction helpers)
- `training/app/config/settings.py`
- `training/app/trainer_entry.py`
- `training/training/{lora_trainer.py, train_gpu_simple.py, validate_adapter.py, smoke_gate.py}`
- `orchestration/app/experience/recorder.py`
- `orchestration/app/experience/sft_builder.py` (new)
- `shared/contracts/experience.py`
- `scripts/seed_adapter_metadata.py`
- `training/tests/{test_recipe_loader.py (new), test_lora_trainer_recipe.py (new), test_sft_builder.py (new)}`
- `orchestration/tests/test_recorder_tool_pairs.py` (new)

Steps:
1. `shared/contracts/training.py` (~120 LOC additions):
   - Pydantic `AdapterRecipe`:
     ```python
     class AdapterRecipe(BaseModel):
         peft_type: Literal["lora","rslora","dora","qlora"] = "lora"
         r: int = 16
         alpha: int = 32
         dropout: float = 0.0
         target_modules: List[str]                 # required, no default
         use_rslora: bool = False
         use_dora: bool = False
         loftq_config: Optional[Dict[str, Any]] = None
         load_in_4bit: bool = False                 # set True for qlora
         learning_rate: float = 2e-4
         num_epochs: int = 3
         max_seq_length: int = 2048
         # SFT message format — the load-bearing flexibility hook
         sft_format: Literal["openai_messages","legacy_response_marker"] = "openai_messages"
         tool_call_style: Literal["openai_native","hermes","none"] = "openai_native"
         # Optional per-recipe tokenizer override (rare)
         tokenizer_pre_tokenizer: Optional[Literal["bytelevel","metaspace"]] = None
     ```
   - `RecipeRegistry { default: AdapterRecipe, by_role: Dict[str, AdapterRecipe], by_backend_role: Dict[Tuple[str,str], AdapterRecipe] }`.
   - `load_recipes(path: str | None = None) -> RecipeRegistry` with `lru_cache`. Lookup precedence: `(backend, role)` > `role` > `default`.
   - Export `lookup_recipe(backend: str, role: str) -> AdapterRecipe`.

2. `recipes.yaml` (~40 LOC):
   ```yaml
   default:
     peft_type: lora
     r: 16
     alpha: 32
     dropout: 0.0
     target_modules: [q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]
     learning_rate: 2e-4
     num_epochs: 3
     max_seq_length: 2048
     sft_format: openai_messages
     tool_call_style: openai_native

   by_role:
     coder:    { r: 32, alpha: 64, num_epochs: 4 }   # tool-using → larger capacity
     debugger: { r: 32, alpha: 64, num_epochs: 4 }
     consensus: { peft_type: none }                   # consensus runs base, no recipe

   # Reserved seam: future per-backend overrides go here.
   # by_backend_role:
   #   [secondary, coder]: { peft_type: rslora, use_rslora: true }
   ```

3. `shared/contracts/messages.py` (~150 LOC, new):
   - `build_openai_messages(role, system_prompt, user_prompt, tool_rounds) -> List[dict]` — returns OpenAI-format messages list including any `assistant{tool_calls=[…]}` + `tool{tool_call_id=…, content=…}` rounds.
   - `build_legacy_marker_prompt(role, system_prompt, user_prompt, completion) -> Tuple[str,str]` — returns `(prompt, completion)` for the legacy `[SYSTEM]/[USER]/[RESPONSE]` format. ONE source of truth for both training and inference fallback paths.
   - Both helpers preserve the byte-for-byte invariant declared in `agent_prompts.py`.

4. `shared/contracts/experience.py`:
   - Extend `AgentContribution` with:
     ```python
     tool_calls: Optional[List[Dict[str, Any]]] = None        # assistant→tool_calls round
     tool_results: Optional[List[Dict[str, Any]]] = None      # tool→content round (paired with tool_calls)
     used_base_fallback: bool = False                          # populated by inference response (Plan E)
     ```
   - These fields are populated by `recorder.py` from the `iteration_history` entries that already carry `tool_calls` / role=`tool` rows — no extra plumbing.

5. `orchestration/app/experience/recorder.py`:
   - In `_build_contributions`, walk `iteration_history` and group consecutive `(assistant-with-tool_calls, tool-result*)` rows into a single `AgentContribution` whose `tool_calls` and `tool_results` fields capture the round. The final assistant turn (no tool_calls) becomes the `output` field.
   - Skip pure `role="tool"` rows that aren't paired (defensive).
   - Explicitly skip `role=="tool_result"` rows for backwards-compat with Plan 1 history (the new code emits `role="tool"`, but a manifest may still carry old shards).

6. `orchestration/app/experience/sft_builder.py` (~120 LOC, new):
   - `build_sft_pair(contribution: AgentContribution, recipe: AdapterRecipe, role: str) -> dict` returns:
     - For `recipe.sft_format == "openai_messages"`:
       ```json
       {"messages": [
         {"role": "system",    "content": "<system_prompt>"},
         {"role": "user",      "content": "<user_prompt>"},
         {"role": "assistant", "tool_calls": [...], "content": null},
         {"role": "tool",      "tool_call_id": "…", "content": "…"},
         {"role": "assistant", "content": "<final_output>"}
       ]}
       ```
     - For `recipe.sft_format == "legacy_response_marker"`:
       ```json
       {"prompt": "[SYSTEM]\n…[USER]\n…\n[RESPONSE]\n", "completion": "<output>"}
       ```
   - SFTTrainer (TRL ≥ 0.24) accepts the `messages` format natively via `apply_chat_template` — no custom collator needed.
   - This is the seam that lets a future model with a different tool-call format (Hermes, Llama-3 `<|python_tag|>`, …) be trained without code changes — only `recipe.sft_format` + `recipe.tool_call_style` change.

7. `training/training/lora_trainer.py`:
   - Add `apply_recipe(model, recipe: AdapterRecipe)`:
     ```python
     return FastLanguageModel.get_peft_model(
         model,
         r=recipe.r,
         lora_alpha=recipe.alpha,
         target_modules=recipe.target_modules,
         lora_dropout=recipe.dropout,
         bias="none",
         use_gradient_checkpointing="unsloth",
         use_rslora=recipe.use_rslora,
         use_dora=recipe.use_dora,
         loftq_config=recipe.loftq_config,
     )
     ```
     For `peft_type == "qlora"`, set `load_in_4bit=True` in the upstream `FastLanguageModel.from_pretrained` call.
   - Replace the hardcoded `target_modules=[…]` block (currently lines 670–673) with `apply_recipe(model, recipe)`.
   - Before training: `assert set(recipe.target_modules).issubset({n.split('.')[-2] for n,_ in model.named_modules()})` — fail closed with a clear error if any target module name does not exist on the loaded base model (catches silent identity-LoRA bugs when porting recipes between architectures).
   - The `_train_unsloth` / `_train_mock` signatures gain `recipe: AdapterRecipe` and forward it through.
   - Decide SFT format per-record: when `recipe.sft_format == "openai_messages"` and the dataset record has a `messages` field (produced by sft_builder), call `SFTTrainer(formatting_func=…)` that applies the chat template. Otherwise fall back to today's prompt/completion path.

8. `training/training/train_gpu_simple.py`:
   - Drop module-level `LORA_R` / `LORA_ALPHA`. Per capability, resolve `recipe = lookup_recipe(backend=registry.default_backend, role=cap_to_role[cap])` and pass through. The bootstrap path still uses `sft_format="legacy_response_marker"` UNLESS `recipe.sft_format == "openai_messages"` — if openai, the bootstrap dataset is converted via `build_sft_pair`.
   - Remove the DeepSeek/Mistral conditional pre-tokenizer block (already deleted in Plan A) and replace with `if recipe.tokenizer_pre_tokenizer: …`.

9. `training/app/trainer_entry.py`:
   - Add CLI flag `--recipe-name <str>` (overrides registry lookup; useful for one-off experiments).
   - Resolve the recipe at startup; pass through `LoRATrainer.train(recipe=recipe, …)`.
   - Build SFT pairs from `experiences.jsonl` using `sft_builder.build_sft_pair(contribution, recipe, role)`. The coder/debugger SFT JSONL now contains full `messages` arrays with `assistant.tool_calls` and `tool.tool_call_id` rounds.
   - Persist the *full recipe dict* into `metadata.json["history"][-1]["recipe"]` (Plan C consumes this).
   - Hash the dataset (sorted SHA256) and tokenizer config; pass through to manifest writer (Plan C).

10. `training/training/{validate_adapter.py, smoke_gate.py}`:
    - Adapter loader switches on `recipe.peft_type`. QLoRA path: load the base model in 4-bit via `bitsandbytes` before applying the adapter.
    - Smoke gate verifies tool-using adapters by feeding a fixed prompt that requires a `read_file` call, calling the loaded adapter via vLLM's OpenAI endpoint, and asserting the response contains a structurally-valid `tool_calls` array (not just text). This catches "adapter forgot how to emit tool calls" regressions.

11. `orchestration/app/graph/nodes.py`:
    - No semantic change. Tool-using roles already include `tools=build_tool_schemas()` in the request. Verify after retraining that `response.tool_calls` is non-empty for at least one coder turn in `e2e_test.py`.

12. `scripts/seed_adapter_metadata.py`:
    - Seed entries now include `recipe` payload from `recipes.yaml`. Skip stale entries that pre-date the recipe field.

13. Tests:
    - `training/tests/test_recipe_loader.py`: assert YAML round-trip, precedence rule `(backend,role) > role > default`, and recipe override behaviour.
    - `training/tests/test_lora_trainer_recipe.py`: parametrise over `peft_type ∈ {lora, rslora, dora}` (skip qlora when no GPU) and assert `apply_recipe` calls Unsloth with the right kwargs (mock `FastLanguageModel`).
    - `training/tests/test_sft_builder.py`: feed a 3-turn `(coder→tool_calls, tool, coder→final)` history and assert the produced `messages` array matches the OpenAI canonical shape exactly.
    - `orchestration/tests/test_recorder_tool_pairs.py`: assert `_build_contributions` correctly groups `tool_calls` and `tool` rows.
    - `training/tests/test_smoke_gate_tool_call.py`: gate fails when adapter outputs raw text instead of a `tool_calls` array (mock vLLM response).

Constraints/Notes:
- The OpenAI message format is what *vLLM* serves for tool-using models when the chat template supports tools (see `BackendEntry.chat_template_supports_tools` from Plan A). Qwen3-14B's chat template DOES support tool calling — verify by `tokenizer.apply_chat_template(messages, tools=[…])` and inspect the rendered string contains the `<tool_call>…</tool_call>` markers Qwen expects.
- target_modules MUST match the architecture or the LoRA is an identity transform — the runtime `issubset` assertion is non-negotiable.
- QLoRA requires 4-bit base load — gate by `torch.cuda.is_available()` and a free-memory check; fail closed if not enough VRAM. On a single 3090 with Qwen-14B already in fp8, QLoRA is unlikely to fit alongside — flag as a future-work hook, not a default.
- The "byte-for-byte identical training/inference prompt" invariant from `agent_prompts.py` STILL holds — but it now applies to whichever format the recipe declares. The legacy marker path remains the byte-for-byte spec; the openai_messages path defers byte-equality to the chat template, which both sides apply identically.
- Coder/debugger MUST be retrained with `sft_format=openai_messages` before the legacy marker fallback in `inference/backends/local_vllm.py` is removed (do NOT delete the fallback in this plan — Plan E removes it after metrics confirm parity).
- Non-tool roles (planner, researcher, reviewer, critic, consensus) keep `sft_format=legacy_response_marker` for now — switching them is risk for no clear win.
- ENHANCEMENT HOOK (a): a future model that demands Hermes-style `<tool_call>{…}</tool_call>` SFT only changes `recipe.tool_call_style="hermes"` and a small branch inside `sft_builder.build_sft_pair`.
- ENHANCEMENT HOOK (b): execution-grounded scoring (tool failure rate as a training signal) requires `tool_results` to land in `AgentContribution` first — that field is added here; the scorer is Plan F.
- ENHANCEMENT HOOK (c): if the chat template's tool block is too token-heavy, set `recipe.compress_tool_schemas=True` in a future patch and pre-hash schemas in `sft_builder` — not done now.
- ENHANCEMENT HOOK (d): when a model's tokenizer needs a pre-tokenizer override, set `recipe.tokenizer_pre_tokenizer="bytelevel"`; the conditional in `lora_trainer` is the only place that reads it.
---

=== PLAN C: Extended Manifest + Cross-Platform Locking + One-Shot Migration ===

Task Goal: Replace the per-call inline JSON read-modify-write of `manifest.json` with a single shared `shared/manifest.py` helper that uses `filelock` (cross-platform, replaces fcntl-only path on Linux), encodes the v2 schema with full provenance (backend, peft_type, target_modules, recipe, dataset_hash, tokenizer_hash, trainer_version, eval_metrics, smoke_results, used_base_fallback aggregate), and ship a one-shot `migrate_manifest.py` that converts legacy v1 entries to v2.

Relevant Files:
- `shared/manifest.py` (new)
- `shared/contracts/manifest_schema.py` (new — Pydantic schema)
- `pyproject.toml` / `requirements.txt` for all three services (add `filelock`)
- `training/training/lora_trainer.py`
- `training/app/trainer_entry.py`
- `inference/app/backends/factory.py`
- `inference/app/routes/adapters.py`
- `scripts/{show_manifest.py, seed_adapter_metadata.py, rollback_adapter.py, migrate_manifest.py (new)}`
- `shared/tests/test_manifest_schema.py` (new)
- `shared/tests/test_manifest_locking.py` (new)
- `scripts/tests/test_migrate_manifest.py` (new)

Steps:
1. `shared/contracts/manifest_schema.py` (~140 LOC):
   ```python
   class HistoryEntry(BaseModel):
       version: str
       status: Literal["staging","active","retired","failed"]
       trained_at: datetime
       backend: str                       # opaque key from models.yaml
       base_model: str                    # entry.model_id at training time
       peft_type: str
       target_modules: List[str]
       r: int
       alpha: int
       dropout: float
       quantization: Optional[str] = None
       recipe: Dict[str, Any]             # full recipe dict for replay
       dataset_hash: str                  # sha256 of sorted training pairs
       tokenizer_hash: str                # sha256 of tokenizer.json bytes
       trainer_version: str               # e.g. "trl==0.24.0+unsloth==2025.10"
       n_samples: int
       final_loss: float
       size_mb: float
       eval_metrics: Dict[str, float] = Field(default_factory=dict)
       smoke: Dict[str, Any] = Field(default_factory=dict)
       used_base_fallback_aggregate: float = 0.0   # reserved for Plan E

   class AdapterEntry(BaseModel):
       schema_version: int = 2
       active_version: str
       active_path: str
       previous_version: str = ""
       previous_path: str = ""
       backend: str
       updated_at: datetime
       history: List[HistoryEntry]

   class Manifest(BaseModel):
       schema_version: int = 2
       updated_at: datetime
       adapters: Dict[str, AdapterEntry]
   ```
   - Strict validation via Pydantic; readers fail closed on `schema_version > 2`.

2. `shared/manifest.py` (~120 LOC):
   ```python
   def read_manifest(path: Path) -> Manifest: ...
   def write_manifest_locked(path: Path, mutator: Callable[[Manifest], None]) -> None:
       with FileLock(str(path) + ".lock", timeout=30):
           m = read_manifest(path) if path.exists() else Manifest(...)
           mutator(m)
           m.updated_at = datetime.now(timezone.utc)
           tmp = path.with_suffix(".tmp")
           tmp.write_text(m.model_dump_json(indent=2))
           os.replace(tmp, path)   # atomic on POSIX + Windows
   ```
   - On read, if `schema_version == 1`, raise `LegacyManifestError("run scripts/migrate_manifest.py")`. Do NOT silently migrate at read time — migration is an explicit, auditable step.

3. `requirements.txt` for orchestration / inference / training: add `filelock>=3.13`. Update Dockerfiles' pip install layer accordingly.

4. `training/training/lora_trainer.py`:
   - Delete the `_manifest_file_lock` context manager (currently fcntl-only, no-op on Windows). It is replaced by `write_manifest_locked` from the shared helper.
   - `_post_train_stage`, `_promote_to_active`, `_mark_failed` are rewritten to call `write_manifest_locked(manifest_path, mutator)`. The mutator closure receives the parsed `Manifest` Pydantic object and mutates it directly — no JSON parsing inside.
   - Compute `dataset_hash = sha256(sorted_jsonl_lines).hexdigest()` and `tokenizer_hash = sha256(tokenizer.json bytes).hexdigest()` BEFORE training; pass them to the manifest mutator.
   - Capture `trainer_version` via `importlib.metadata.version("trl")` + `version("unsloth")`; format as `"trl==X.Y.Z|unsloth==A.B.C"`.

5. `training/app/trainer_entry.py`:
   - After `validate_adapter.py` finishes, capture `eval_metrics` (loss, perplexity, tool_call_validity_rate from Plan B's smoke gate) and pass into the `_promote_to_active` call.

6. `inference/app/backends/factory.py`:
   - `bootstrap_adapters(manifest_path)` is rewritten to use `read_manifest`. Loop over `m.adapters.items()`; resolve backend via `entry.backend` (always present in v2). On `LegacyManifestError`, log a fatal warning and skip — the operator must run `migrate_manifest.py`.

7. `inference/app/routes/adapters.py`:
   - `/adapters/rollback` now calls `write_manifest_locked` to swap `active_version`/`previous_version`, then issues the vLLM `unload_lora_adapter` + `load_lora_adapter` HTTP calls. Tolerates v2 only.

8. `scripts/migrate_manifest.py` (new, ~150 LOC):
   - CLI: `python scripts/migrate_manifest.py [--check] [--manifest PATH]`.
   - Reads the v1 manifest. For each adapter entry:
     - Sniff `adapter_config.json` in `active_path` → fill `peft_type`, `r`, `alpha`, `target_modules`, `dropout`.
     - Sniff `tokenizer_config.json` → compute `tokenizer_hash`.
     - Use `models.yaml` `default_backend` for the `backend` field (since v1 had `model_family` which is now obsolete).
     - Set `dataset_hash="legacy"`, `trainer_version="legacy"`, `eval_metrics={}`, `smoke={}`, `recipe={"peft_type":"lora", ...}` reconstructed from the sniffed values.
     - Bump `schema_version` 1 → 2.
   - `--check`: prints a unified diff of v1→v2 without writing.
   - Backs up the original to `manifest.json.v1.bak.<ISO8601>` before overwriting.
   - Idempotent: re-running on a v2 manifest is a no-op (returns code 0 with `"already at v2"`).

9. `scripts/show_manifest.py`, `seed_adapter_metadata.py`, `rollback_adapter.py`:
   - Read via `shared/manifest.read_manifest`. Render new fields (`recipe.peft_type`, `dataset_hash[:8]`, `eval_metrics`).
   - `seed_adapter_metadata.py` writes v2 entries directly; no v1 emission paths remain.

10. Tests:
    - `shared/tests/test_manifest_schema.py`: round-trip validation; reject `schema_version=3`.
    - `shared/tests/test_manifest_locking.py`: spawn 4 processes that each call `write_manifest_locked` with a counter-increment mutator; assert the final counter equals 4 (no lost writes). Use `multiprocessing.Pool`.
    - `scripts/tests/test_migrate_manifest.py`: feed a fixture v1 manifest; assert the migrated v2 manifest matches an expected snapshot; assert `--check` produces a diff and does not write; assert idempotency.

Constraints/Notes:
- `filelock` MUST be added to all three services' requirements; the pip install must be in the Docker build layer to keep cold-start fast.
- `os.replace` is atomic on Windows only when source and dest are on the same volume — the `.tmp` file uses `path.with_suffix(".tmp")` precisely for this reason.
- `schema_version` is the gate; readers fail closed on unknown versions — don't add a "best-effort" tolerant path, it will mask migration bugs.
- `dataset_hash` enables Plan E's idempotent scheduler replay (skip identical dataset) — do not omit even if it feels redundant today.
- `tokenizer_hash` mismatch at adapter swap time is a future auto-rollback signal (Plan E); record it now.
- The migration is one-shot but MUST be re-runnable safely. Idempotency is verified by the test.
- ENHANCEMENT HOOK (a): `eval_metrics["tool_call_validity_rate"]` (from Plan B smoke gate) is the auto-rollback metric in Plan E.
- ENHANCEMENT HOOK (b): once latency / tokens-per-sec / output-entropy are recorded by smoke_gate (Plan F item 4), they slot into `HistoryEntry.smoke` without schema bump.
- ENHANCEMENT HOOK (c): the `previous_version`/`previous_path` fields stay shallow (one slot); a deeper rollback chain requires a list[str] field — defer to a future schema_version=3.
---

=== PLAN D: Plan-File Workflow Integration (plans/current-task.md ↔ orchestration) ===

Task Goal: Wire `plans/current-task.md` into the orchestration layer so the planner agent ingests it on the first turn, the coder/debugger update step status as they work (using the existing tool runner), and the consensus/finalize path archives the completed plan to `plans/archive/`. Plan state is persisted in Redis under `task_state:{task_id}` for cross-session continuity.

Relevant Files:
- `orchestration/app/plans/__init__.py` (new)
- `orchestration/app/plans/{schema.py, reader.py, writer.py}` (new)
- `orchestration/app/db/redis.py`
- `orchestration/app/graph/{nodes.py, state.py, pipeline.py, edges.py}`
- `orchestration/app/routes/run.py`
- `shared/contracts/{orchestration.py, agent_prompts.py}`
- `plans/current-task.md` (canonical bootstrap source — already exists)
- `plans/archive/` (new dir, .gitkeep)
- `orchestration/tests/test_plans_reader_writer.py` (new)
- `orchestration/tests/test_planner_with_plan.py` (new)
- `orchestration/tests/test_plan_redis_persist.py` (new)

Steps:
1. `orchestration/app/plans/schema.py` (~80 LOC):
   ```python
   class PlanStep(BaseModel):
       id: str                      # auto-assigned step-N or LLM-supplied
       description: str
       status: Literal["pending","in_progress","done","blocked"] = "pending"
       owner_role: Optional[str] = None
       evidence: str = ""
       files_touched: List[str] = []
       tools_used: List[str] = []

   class Plan(BaseModel):
       goal: str
       steps: List[PlanStep]
       created_at: datetime
       updated_at: datetime
   ```
   - Validation: `id` is unique within `steps`; status transitions are forward-only except `pending→blocked` and `blocked→in_progress` (enforced in writer).

2. `orchestration/app/plans/reader.py` (~120 LOC):
   - Parse `plans/current-task.md` via a small markdown-section parser:
     - `## Task Goal` → `Plan.goal`
     - `## Steps` → numbered list → `Plan.steps[*].description`
     - `## Constraints/Notes` → discarded for now (kept in raw markdown for human readers)
   - Return `Plan` populated with auto-generated `id="step-1"`, `step-2"`, … and `status="pending"`.
   - Path resolution uses Plan 1's `_safe_path` to reject `..` and absolute paths.

3. `orchestration/app/plans/writer.py` (~100 LOC):
   - `write_plan_atomic(path: Path, plan: Plan) -> None`: render plan back to markdown (preserve human formatting), write to `<path>.tmp`, then `os.replace`. Uses the same `filelock` helper from Plan C with a per-task `.lock`.
   - `validate_transition(old: Plan, new: Plan)`: reject illegal status transitions; called by writer.

4. `orchestration/app/db/redis.py`:
   - Add `set_plan(task_id, plan: Plan, ttl_s: int)` / `get_plan(task_id) -> Optional[Plan]`.
   - Key namespace: `task_state:{task_id}`. TTL default `7 * 86400` (configurable via `ORCHESTRATION__TASK_STATE_TTL_S`, default 604800).
   - On Redis miss, log `"task_state degraded"` once per task and fall back to file read.

5. `orchestration/app/graph/state.py`:
   - Add to `AgentState`:
     ```python
     plan: Optional[Plan]
     plan_changed: bool          # set True when any node mutates plan; finalize writes back
     current_step_id: Optional[str]  # planner sets, coder/debugger respect
     ```

6. `orchestration/app/routes/run.py`:
   - On request entry: if `request.task_id` is present, try `get_plan(task_id)`; on miss, read `request.plan_path` (default `plans/current-task.md`).
   - On finalize (when `quality_converged=True` OR explicit checkpoint event): call `set_plan(task_id, plan, ttl)` AND `write_plan_atomic(plan_path, plan)`. Also archive a snapshot to `plans/archive/{task_id}-{YYYYMMDD-HHMMSS}.md` so completed plans are not overwritten by the next task.
   - SSE: emit `event: task_state` with `{task_id, completed_steps, total_steps}` whenever `plan_changed`.

7. `orchestration/app/graph/nodes.py`:
   - `planner_node`:
     - If `state["plan"]` is None or empty → produce a fresh plan, return delta with `plan` and `plan_changed=True`.
     - Else → mark the next `pending` step as `in_progress`, set `state["current_step_id"]`, and emit guidance referencing it. Do NOT regenerate the plan when one already exists.
   - `coder_node` / `debugger_node`:
     - Receive `current_step_id` via the system prompt (see step 9 below).
     - When their tool round completes successfully (no errors in last 3 tool entries), set the step's `status="done"`, append `tools_used` and `files_touched` (extracted from successful `write_file` / `read_file` calls), set `plan_changed=True`.
     - When the LLM emits `[BLOCKED:<reason>]` in its output, set `status="blocked"`, record `evidence=<reason>`.
   - `consensus_node`: ensures all `pending` steps are either `done` or explicitly `blocked` before emitting the final answer; if any remain `pending`, append a "remaining work" section to the consensus output.

8. `orchestration/app/graph/pipeline.py`:
   - No structural change. Plan reading happens in `routes/run.py` before the graph is invoked; writing happens after the graph returns.

9. `shared/contracts/agent_prompts.py`:
   - Append to the planner prompt: a strict JSON schema for plan steps that the planner emits when no plan exists. The runtime parses this into `PlanStep` objects.
   - Append to coder + debugger prompts: a single line — `"You are working on plan step {current_step_id}: {current_step_description}. Use the provided tools to complete this step, then signal done."` — *only* injected when `current_step_id` is set; otherwise unchanged.
   - The byte-for-byte invariant for non-tool roles (reviewer/critic/consensus) holds — those prompts are unchanged.

10. `shared/contracts/orchestration.py`:
    - `OrchestrationRequest` gains `plan_path: Optional[str] = "plans/current-task.md"` and `task_id: Optional[str] = None`. When `task_id is None`, the plan is ephemeral (in-memory only — not persisted to Redis or markdown).

11. Tests:
    - `orchestration/tests/test_plans_reader_writer.py`: round-trip a fixture markdown file → `Plan` → markdown; assert byte equality after a no-op edit.
    - `orchestration/tests/test_planner_with_plan.py`: fake inference returns a plan JSON; assert `state["plan"]` is populated correctly and the second invocation does NOT regenerate.
    - `orchestration/tests/test_plan_redis_persist.py`: fakeredis fixture; assert round-trip and TTL eviction.

Constraints/Notes:
- Redis is source of truth during a live task; markdown is bootstrap + checkpoint only — do not write back to markdown on every step (would churn the working tree).
- Writeback to markdown happens only at convergence or on explicit `checkpoint` SSE input — once per task lifecycle, typically.
- `task_id` MUST be supplied for plan persistence; falling back to `session_id` would conflate tasks across user requests.
- `plan_path` resolution uses `_safe_path` from Plan 1's tool runner — same denylist (`.git`, `.env`, `secrets/`).
- Reader → writer round-trip MUST be idempotent. Fixture-test this explicitly; a non-idempotent round-trip will silently rewrite the file every step.
- The plan-update logic is *opportunistic* — if the LLM doesn't follow the format, the step status stays `in_progress` and the human can fix it manually. Do NOT add hard validation that crashes the run.
- ENHANCEMENT HOOK (a): a `/v1/plan/checkpoint` REST endpoint can force a markdown writeback mid-run for long tasks; implement when self-paced /loop runs become routine.
- ENHANCEMENT HOOK (b): a planner-budget cap (max steps per plan, default 20) prevents the LLM from emitting 200-step manifests; default off.
- ENHANCEMENT HOOK (c): `plan.steps[*].depends_on: List[str]` enables DAG-aware step routing later — schema field reserved.
---

=== PLAN E: Stability & Safety — Locking, Rotation, Cursor, Auto-Rollback ===

Task Goal: Harden the self-training loop against crashes, races, and adapter regressions: cross-platform JSONL locking via `filelock`, daily JSONL rotation with a Redis cursor for the trainer, scheduler timeout/heartbeat, and auto-rollback on adapter fallback threshold breach. Removes the legacy stop/start vLLM training path.

Relevant Files:
- `training/app/trainer_wrapper.py`
- `training/app/utils/experience_reader.py`
- `training/scheduler/scheduler.py`
- `training/app/config/settings.py`
- `inference/app/backends/local_vllm.py`
- `inference/app/config/settings.py`
- `orchestration/app/experience/{jsonl_store.py, inflight.py, recorder.py}`
- `orchestration/app/config/settings.py`
- `shared/contracts/experience.py`
- `scripts/rollback_adapter.py`
- `training/tests/{test_scheduler_timeout.py, test_experience_cursor.py}` (new)
- `inference/tests/test_auto_rollback.py` (new)
- `orchestration/tests/test_jsonl_rotation.py` (new)

Steps:
1. `training/app/trainer_wrapper.py`:
   - Delete `stop_vllm()` and `start_vllm()` and the `BLOCK` mode / `training_exclusive_mode` setting. The trainer NEVER touches the vLLM container — adapters land via the hot-swap `/adapters/load` route.
   - `_base_model_for(role)` already reads the registry (Plan A).
   - Subprocess invocation gains `timeout=settings.training_timeout_seconds` (default 3600) and `creationflags=subprocess.CREATE_NEW_PROCESS_GROUP` on Windows / `start_new_session=True` on POSIX. On `TimeoutExpired`, terminate the group and log `training_timed_out` with `role`, `dataset_hash` (Plan C).

2. `orchestration/app/experience/jsonl_store.py`:
   - Replace the existing `fcntl.flock` block with `filelock.FileLock(<jsonl>.lock, timeout=10)`. Same lock is used by `inflight.spool_pending` so spool-flush + append never race.
   - Add daily rotation: `_resolve_jsonl_path()` returns `f"experiences-{utcnow:%Y%m%d}.jsonl"` under `settings.jsonl_dir`. The legacy single-file `experiences.jsonl` continues to be readable by the cursor (step 4) until pruned.
   - Reads (`read_all_for_role(role)`) walk every file matching `experiences-*.jsonl` in chronological order.

3. `training/app/utils/experience_reader.py`:
   - Iterate over dated shards in chronological order, each with a `filelock`-shared lock for the duration of the read.
   - Yield records with `(shard_path, line_offset)` so the cursor can record exact resumption point.

4. `training/scheduler/scheduler.py`:
   - On each tick, read cursor from Redis: `cursor:{role}` → `{shard, offset}`. Iterate `experience_reader.iter_after(cursor)`. After successful training (and ONLY after `_promote_to_active` succeeds), commit the new cursor to Redis. Failed runs leave the cursor untouched so the next tick replays — Plan C's `dataset_hash` makes replay idempotent (skip if last-trained dataset_hash matches current).
   - Subprocess wrapping (per step 1) gets `timeout=settings.training_timeout_seconds` plus a heartbeat:
     ```python
     def _heartbeat_loop():
         while not done:
             Path(settings.heartbeat_path).write_text(datetime.utcnow().isoformat())
             time.sleep(15)
     ```
     A separate watchdog (out of process — operator's concern, not the scheduler's) reads the heartbeat. The scheduler does not auto-restart from inside its own thread.

5. `training/app/config/settings.py`:
   - Add `training_timeout_seconds: int = 3600`, `heartbeat_path: str = "/data/scheduler.heartbeat"`, `jsonl_retention_days: int = 30` (deletion is manual; setting is informational only).

6. `inference/app/config/settings.py` + `inference/app/backends/local_vllm.py`:
   - Add `adapter_fallback_rollback_threshold: int = 5` and `adapter_fallback_window_seconds: int = 60`.
   - In `local_vllm.LocalVLLMBackend.generate`, when the response indicates an adapter was unavailable (`response.used_base_fallback=True`), increment `_adapter_fallback_counts[adapter_name]` and timestamp; if N events occur within the window, fire `POST /adapters/rollback` for that adapter, then reset the counter.
   - `_adapter_fallback_counts` is a `dict[str, deque[float]]` (timestamps); deque is size-bounded.

7. `shared/contracts/experience.py`:
   - `AgentContribution.used_base_fallback: bool = False` (already added in Plan B; ensure it is wired through).
   - `GenerateResponse.used_base_fallback: bool = False` field added so the inference layer reports the truth.

8. `inference/app/backends/local_vllm.py` (continued):
   - When the vLLM `chat/completions` response carries an internal flag indicating LoRA fallback (vLLM emits a header `x-vllm-lora-fallback` when the requested adapter is missing, or the request omits the LoRA name), set `GenerateResponse.used_base_fallback=True`. This is the ground truth used by `recorder.py` (Plan B) and by step 6's threshold logic.

9. `orchestration/app/experience/recorder.py`:
   - Read `response.used_base_fallback` and propagate to the corresponding `AgentContribution`.

10. `scripts/rollback_adapter.py`:
    - Add `--auto` (used by step 6's auto-rollback path) and `--reason` (free-text, recorded into `Manifest.adapters[name].history[-1].smoke["rollback_reason"]` via Plan C's helper).
    - `--auto` mode skips the interactive confirmation prompt.

11. Tests:
    - `training/tests/test_scheduler_timeout.py`: mock subprocess that sleeps > timeout; assert termination + `training_timed_out` log line; assert cursor is NOT advanced.
    - `training/tests/test_experience_cursor.py`: write 3 dated shards; cursor at shard-2/offset-5; assert reader yields exactly the unread tail.
    - `inference/tests/test_auto_rollback.py`: simulate 5 fallback events within 60s; assert `/adapters/rollback` is fired once.
    - `orchestration/tests/test_jsonl_rotation.py`: monkeypatch `utcnow` to advance one day mid-write; assert two shards exist with correct contents.

Constraints/Notes:
- `filelock` is already required by Plan C — no new dependency here.
- Rotation: keep 30 days, configurable. **Deletion is never automatic** — flagged for manual review in `scripts/prune_experiences.py` (out of scope; document the file's intent only).
- Cursor commit happens AFTER successful training; failed runs replay (idempotent via Plan C `dataset_hash`).
- Fallback rollback threshold default 5 within 60s; tune via env if false-positive rate is high in production.
- Heartbeat watchdog is **read-only** on the scheduler thread; an external systemd / docker healthcheck is the restart authority. Do NOT auto-restart from inside the scheduler — that loops cleanly into self-DoS.
- The legacy `stop_vllm`/`start_vllm` deletion is the load-bearing simplification — do not preserve the BLOCK code path "just in case".
- The vLLM `x-vllm-lora-fallback` header is implementation-specific to recent vLLM (≥0.6); the local_vllm backend already reads response headers — verify with a smoke test before relying on it. If absent in your vLLM version, infer fallback by comparing requested `adapter_name` vs response's `model` field.
- ENHANCEMENT HOOK (a): per-shard compression (gzip) when older than N days — stub function but no caller.
- ENHANCEMENT HOOK (b): cross-region cursor (when training runs on a different host than orchestration) — Redis already supports this; just document.
- ENHANCEMENT HOOK (c): rollback metric → SLO dashboard. Track `rollback_per_24h` as a gauge.
---

=== PLAN F: Backlog — Embedding Router, Distillation, Tiered ChromaDB, Prefix Caching, Async Subprocess, Cancellation ===

Task Goal: Track lower-priority quality-of-life and performance improvements that are net-new value but not blockers, so they are not lost as Plans A–E land.

Relevant Files:
- `shared/routing/query_router.py`
- `orchestration/app/{experience/vector_store.py, routes/run.py, clients/inference_grpc.py}`
- `shared/contracts/inference_defaults.py` (new)
- `inference/app/config/settings.py`
- `training/{training/lora_trainer.py, scheduler/scheduler.py, app/trainer_wrapper.py}`
- `training/distillation/__init__.py`, `training/distillation/distiller.py` (new)
- `docker/compose.yml`
- `scripts/{eval_adapters.py, e2e_test.py}`
- `.claude/worktrees/eloquent-rhodes-1a8504/` (cleanup — confirm stale first)
- `orchestration/tests/test_embedding_router.py` (new)
- `training/tests/test_distillation.py` (new)
- `orchestration/tests/test_cancellation_propagation.py` (new)

Steps:
1. `shared/routing/query_router.py` (~150 LOC):
   - Replace regex-based capability classification with embedding similarity. At startup, load a labelled prompt set (`shared/routing/router_prompts.yaml` — ~6 prompts × 7 roles = 42 anchors), embed via the same `sentence-transformers` model orchestration uses for context retrieval (`settings.embedding_model`), and cache the matrix. At query time, embed the user prompt and pick the role whose anchor centroid has highest cosine similarity.
   - Fallback: if `numpy`/`sentence-transformers` is unavailable at import time, degrade gracefully to the existing regex tables (so dev environments without the embedder still work).

2. `training/distillation/distiller.py` (~200 LOC):
   - Pipeline: read `experiences.jsonl` shards, filter for `critic.score < threshold` rows, regenerate the assistant turn via the consensus role with a `improve this` system prompt, save the corrected pair to `experiences-distilled.jsonl` flagged `provenance="distilled"`.
   - The trainer reads both shards. SFT pairs from distilled rows get a lower sample weight via `recipe.distilled_sample_weight: float = 0.5` (new optional field on `AdapterRecipe`).

3. `orchestration/app/experience/vector_store.py`:
   - Split the single `experiences` ChromaDB collection into per-role collections (`exp_coder`, `exp_debugger`, …) and per-outcome (`_passed`, `_failed`). Retrieval can then weight `exp_coder_failed` higher than `exp_coder_passed` so harder cases surface preferentially.
   - Migration: new collections only — leave `experiences` in place until backfilled (one-time job).

4. `training/training/lora_trainer.py`:
   - Smoke gate also records latency_ms (single-token decode), tokens_per_sec (256-token decode), and output_entropy (softmax entropy of 32-token sample) into `HistoryEntry.smoke` (extends Plan C schema; no schema bump — `smoke` is open-shape).

5. `docker/compose.yml`:
   - Add `--enable-prefix-caching` to the vLLM service args. Bump `--max-model-len` if VRAM allows (target 8192 once fp8 KV-cache headroom is verified). Per-backend overrides flow through Plan A's `models.yaml` `max_model_len`.

6. `training/app/trainer_wrapper.py` + `training/scheduler/scheduler.py`:
   - Convert `subprocess.run` to `asyncio.create_subprocess_exec` for parity with the rest of the async stack. The timeout/heartbeat logic from Plan E is preserved (use `asyncio.wait_for`).

7. `orchestration/app/routes/run.py` + `clients/inference_grpc.py`:
   - On client disconnect (SSE), `task.cancel()` propagates through the LangGraph `astream` loop; in `inference_grpc.py`, the active gRPC `aio.UnaryStreamCall.cancel()` is called so vLLM stops generation. Half-measures (cancel orchestration but leave gRPC streaming) leak generations and waste GPU time.

8. `shared/contracts/inference_defaults.py` (new, ~30 LOC):
   - Consolidate `grpc_default_max_tokens` / `_temperature` / `_top_p` into a single Pydantic model imported by both inference (server-side defaults) and orchestration (client-side fallback when role doesn't override).

9. `scripts/eval_adapters.py` + `scripts/e2e_test.py`:
   - Refactor to consume `models.yaml` registry (Plan A).

10. `.claude/worktrees/eloquent-rhodes-1a8504/`:
    - Confirm staleness via `git -C .claude/worktrees/eloquent-rhodes-1a8504 status` (it ships old static `--lora-modules` compose flags, pre-Plan A naming). If confirmed unmodified, delete the directory. **Do not delete without verifying — it may contain in-progress work.**

11. Tests:
    - `orchestration/tests/test_embedding_router.py`: parametrise over 20 fixture prompts; assert top-1 role accuracy ≥ 90 % on the labelled set.
    - `training/tests/test_distillation.py`: feed 5 low-score experiences; assert distilled output is non-empty and `provenance="distilled"` is set.
    - `orchestration/tests/test_cancellation_propagation.py`: simulate SSE client disconnect mid-stream; assert the gRPC call is cancelled and no further tokens are generated.

Constraints/Notes:
- All items here are net-new value; none is a blocker for Plans A–E.
- Items 1 and 3 share the embedder; build the embedder cache once at orchestration startup and re-use across both call sites.
- Item 4 depends on Plan C's manifest schema (open-shape `smoke` field).
- Item 5 depends on Plan A's `models.yaml`.
- Item 7 cancellation must traverse all three layers (SSE handler → LangGraph task → gRPC call) — half-measures leak generations.
- Distillation MUST flag provenance so SFT replay does not loop on its own outputs (model collapse risk).
- ChromaDB tier split increases storage modestly (~2×); accept it.
- ENHANCEMENT HOOK (a): RAG-style retrieval over the per-role tiers can plug in here once the embedder is cached.
- ENHANCEMENT HOOK (b): a streaming gRPC `cancel()` wired to client-side abort means the user can hit ESC and free the GPU instantly — quality-of-life win.
- ENHANCEMENT HOOK (c): output_entropy (item 4) is a leading indicator of adapter degradation; long-term, it can feed Plan E's auto-rollback alongside `tool_call_validity_rate`.
---
