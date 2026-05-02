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
