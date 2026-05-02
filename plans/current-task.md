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
