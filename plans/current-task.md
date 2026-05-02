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
