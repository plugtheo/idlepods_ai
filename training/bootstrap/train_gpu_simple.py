#!/usr/bin/env python3
"""
Agent-specific LoRA training — one adapter per agent capability.

Pipeline:
  1. generate_data.py downloads 5k-10k real samples per capability from
     publicly licensed HuggingFace datasets (Apache-2.0 / CC-BY-4.0 / MIT).
  2. This script formats the data and delegates all GPU training to LoRATrainer
     (lora_trainer.py), which uses Unsloth + SFTTrainer with completion-only
     masking.  Having a single training implementation eliminates format/config
     drift between the bootstrap path and the online experience-driven path.
  3. After training, versioned metadata and a root manifest are written so
     vLLM can discover and hot-load newly produced adapters.

Model assignment: all capabilities use the default backend from models.yaml.

Output (auto-discovered by ModelRegistry._scan_adapters):
  data/lora_checkpoints/{capability}_lora/
    adapter_model.safetensors
    adapter_config.json
    metadata.json
"""

import asyncio
import json
import os
import shutil
import sys
import tempfile
import torch
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Repo root must be on sys.path before lora_trainer is imported because
# lora_trainer.py imports shared.* at module level.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# lora_trainer.py and generate_data.py live in the same directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_data import generate_datasets, TARGET_DIR as DATA_DIR
from lora_trainer import LoRATrainer, _pre_train_backup, _post_train_version, _post_train_stage, _promote_to_active, _mark_failed
from shared.contracts.agent_prompts import AGENT_PROMPTS, BOOTSTRAP_CAP_TO_ROLE
from shared.contracts.training import lookup_recipe
from shared.contracts.experience import AgentContribution
from shared.contracts.sft_builder import build_sft_pair

# ---------------------------------------------------------------------------
# Agent capability → (base model id, system description)
# Mirrors ModelRegistry._CAPABILITY_MODEL_MAP and local_models.json
# ---------------------------------------------------------------------------
def _resolve_base_model_id() -> str:
    from shared.contracts.models import load_registry
    backend = load_registry().backends[load_registry().default_backend]
    return backend.resolve_training_model_id()


def _resolve_local_model_path(model_id: str) -> str:
    """Return the local HF hub snapshot path for model_id if it exists.

    When HF_HUB_OFFLINE=1 is set (container env), Unsloth's from_pretrained()
    fails to resolve model IDs through the hub cache even though the weights
    are present locally.  This helper walks the hub cache directory and returns
    the absolute snapshot path, falling back to the original model_id string
    so callers can still try.
    """
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(model_id, local_files_only=True)
    except Exception:
        pass
    # Manual fallback: locate the snapshot dir in the HF hub cache.
    try:
        import huggingface_hub.constants as _hfc
        hub_cache = Path(getattr(_hfc, "HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))
        slug = "models--" + model_id.replace("/", "--")
        snap_root = hub_cache / slug / "snapshots"
        if snap_root.is_dir():
            snaps = sorted(snap_root.iterdir())
            for snap in reversed(snaps):
                if (snap / "config.json").exists():
                    return str(snap)
    except Exception:
        pass
    return model_id  # last resort — let Unsloth try the model ID directly

# Include both attention AND FFN (MLP) layers for higher-quality fine-tuning.
# Attention-only LoRA (v2.0) was insufficient — the feed-forward layers carry
# the bulk of the knowledge that maps from instruction tokens to response tokens.
# Omitting them caused garbled token distributions in early adapter versions.
LORA_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",   # attention
    "gate_proj", "up_proj", "down_proj",        # FFN / MLP
]

# AGENT_SPECS: bootstrap capability → (base_model_id, system_prompt)
#
# Built from shared/contracts/agent_prompts.py (AGENT_PROMPTS + BOOTSTRAP_CAP_TO_ROLE)
# so system prompts are defined exactly once and always byte-for-byte identical to
# what orchestration/app/graph/nodes.py sends at inference time.
#
# Bootstrap key → AGENT_PROMPTS key:
#   coding → coder  |  debugging → debugger  |  review → reviewer
#   planning → planner  |  research → researcher  |  criticism → critic
_BASE_MODEL_ID = _resolve_base_model_id()
AGENT_SPECS: Dict[str, Tuple[str, str]] = {
    cap: (_resolve_local_model_path(_BASE_MODEL_ID), AGENT_PROMPTS[BOOTSTRAP_CAP_TO_ROLE[cap]])
    for cap in BOOTSTRAP_CAP_TO_ROLE
}

# Training hyper-parameters — defaults only; per-capability values come from recipes.yaml.
NUM_EPOCHS    = 3
LEARNING_RATE = 2e-4


def _resolve_recipe(capability: str):
    """Resolve the AdapterRecipe for a bootstrap capability label."""
    try:
        from shared.contracts.models import load_registry
        backend = load_registry().default_backend
    except Exception:
        backend = "primary"
    role = BOOTSTRAP_CAP_TO_ROLE.get(capability, capability)
    return lookup_recipe(backend, role)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_sft_pairs(capability: str, system_prompt: str, recipe) -> List[Dict]:
    """
    Load instruction/response pairs from the pre-generated JSONL and build
    SFT records shaped by recipe.sft_format via build_sft_pair.

    Falls back to triggering generate_datasets() if the JSONL is absent.
    """
    jsonl_path = DATA_DIR / f"{capability}_dataset.jsonl"

    if not jsonl_path.exists():
        print(f"  [DATA] {jsonl_path.name} not found — generating now...")
        generate_datasets(capabilities=[capability])

    if not jsonl_path.exists():
        raise FileNotFoundError(f"Dataset not found even after generation: {jsonl_path}")

    role_name = BOOTSTRAP_CAP_TO_ROLE.get(capability, capability)
    pairs: List[Dict] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            messages = rec.get("messages") or []
            response = next(
                (m.get("content", "") for m in reversed(messages) if m.get("role") == "assistant"),
                "",
            )
            if messages and response:
                # generate_data.save_jsonl now bakes the system prompt in as the
                # first message.  Only prepend if it's missing (e.g. old-format JSONL).
                if messages[0].get("role") != "system":
                    full_messages = [{"role": "system", "content": system_prompt}] + messages
                else:
                    full_messages = messages
                contrib = AgentContribution(
                    role=role_name,
                    output=response,
                    quality_score=1.0,
                    iteration=1,
                    messages=full_messages,
                )
                pairs.extend(build_sft_pair(contrib, recipe, role_name))
    return pairs


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def setup_hf_auth() -> None:
    token_file = Path.home() / ".huggingface" / "token"
    if token_file.exists():
        token = token_file.read_text().strip()
        if token:
            os.environ["HF_TOKEN"] = token
            print("[HF] Token loaded")


def print_device_info() -> None:
    if torch.cuda.is_available():
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        free  = (torch.cuda.get_device_properties(0).total_memory
                 - torch.cuda.memory_allocated()) / 1e9
        print(f"      VRAM total={total:.1f}GB  free={free:.1f}GB")
    else:
        print("[CPU] No GPU — training will be slow")


# ---------------------------------------------------------------------------
# Per-capability training (delegates GPU work to LoRATrainer)
# ---------------------------------------------------------------------------

async def train_capability(
    capability: str,
    checkpoint_dir: Path,
    fresh: bool = False,
    validate: bool = False,
) -> Dict[str, Any]:
    """
    Prepare the dataset JSONL, delegate all GPU training to LoRATrainer
    (Unsloth + SFTTrainer + DataCollatorForCompletionOnlyLM), then apply
    versioned save and manifest update.

    This is the single training implementation shared between the bootstrap
    path (this script) and the online experience-driven path (trainer_entry.py →
    lora_trainer.py).  All training logic lives in LoRATrainer so there is
    only one place to fix if something breaks.
    """
    print(f"\n{'='*70}")
    print(f"  Training  : {capability.upper()} adapter")
    print(f"  Trainer   : LoRATrainer (Unsloth + SFTTrainer + DCFCOL masking)")
    print(f"{'='*70}")

    model_id, system_prompt = AGENT_SPECS[capability]
    save_path = checkpoint_dir / f"{capability}_lora"

    _WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin")
    has_existing  = save_path.exists() and any((save_path / f).exists() for f in _WEIGHT_FILES)
    if fresh and has_existing:
        print(f"  [--fresh] Discarding existing adapter — training from scratch")
        shutil.rmtree(save_path)
        has_existing = False

    # ── Resolve recipe ───────────────────────────────────────────────────────
    recipe = _resolve_recipe(capability)

    # ── Prepare dataset JSONL ────────────────────────────────────────────────
    pairs = load_sft_pairs(capability, system_prompt, recipe)
    if not pairs:
        raise ValueError(f"No training pairs loaded for capability '{capability}'")
    print(f"  Pairs loaded : {len(pairs):,}")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as tmp:
        for pair in pairs:
            tmp.write(json.dumps(pair) + "\n")
        dataset_path = Path(tmp.name)

    # ── Pre-training backup (MUST happen before weights are overwritten) ───────
    # _pre_train_backup copies the current adapter weights to a versioned dir
    # so we have a real rollback target.  Calling this AFTER training would
    # back up the NEW weights, rendering rollback useless.
    old_version, backup_path = _pre_train_backup(save_path, capability)
    if backup_path:
        print(f"  [BACKUP] v{old_version} → {backup_path.name}")

    # Warm-start: re-load prior adapter weights as the LoRA starting point so
    # bootstrap re-runs (without --fresh) refine the existing adapter instead
    # of training from scratch.  Opt-in via recipes.yaml: resume_from_prev_adapter: true
    resume_from_checkpoint = (
        str(backup_path)
        if (getattr(recipe, "resume_from_prev_adapter", False) and backup_path)
        else None
    )
    if resume_from_checkpoint:
        print(f"  [WARM-START] Resuming from {backup_path.name}")

    # ── Train via LoRATrainer ────────────────────────────────────────────────
    try:
        trainer = LoRATrainer(base_model=model_id, output_dir=str(save_path))
        result  = await trainer.train(
            dataset_path=dataset_path,
            num_epochs=recipe.num_epochs,
            learning_rate=recipe.learning_rate,
            lora_rank=recipe.r,
            lora_alpha=recipe.alpha,
            recipe=recipe,
            resume_from_checkpoint=resume_from_checkpoint,
        )
    finally:
        dataset_path.unlink(missing_ok=True)

    # ── Post-training: stage → validate (if --validate) → promote / fail ─────
    final_loss = result.get("final_loss", 0.0)
    note       = "incremental" if has_existing else "fresh"

    new_meta = _post_train_stage(
        save_path=save_path,
        capability=capability,
        old_version=old_version,
        base_model_id=model_id,
        n_samples=len(pairs),
        num_epochs=recipe.num_epochs,
        learning_rate=recipe.learning_rate,
        lora_r=recipe.r,
        lora_alpha=recipe.alpha,
        final_loss=final_loss,
        note=note,
    )
    new_version = new_meta["version"]

    # The validate flag is passed down from the caller (main → train_capability).
    # If validation runs and fails, mark failed instead of promoting.
    validation_status = "SKIP"
    if validate:
        val = await validate_adapter(capability, checkpoint_dir)
        validation_status = val["status"]
        if validation_status == "FAIL":
            _mark_failed(
                checkpoint_dir, capability, new_version,
                smoke_results={"pass": False, "reason": val.get("detail", "validate_adapter failed")},
            )
            size_mb = result.get("size_mb", 0.0)
            print(f"  [FAIL] v{new_version}  validate_adapter FAILED — adapter NOT promoted  ({size_mb:.1f} MB)")
            return {
                "capability":   capability,
                "adapter_path": str(save_path),
                "version":      new_version,
                "samples":      len(pairs),
                "epochs":       recipe.num_epochs,
                "final_loss":   final_loss,
                "status":       "FAILED",
                "validation":   "FAIL",
            }

    old_slug = old_version.replace(".", "_")
    guessed_backup = checkpoint_dir / f"{capability}_lora_v{old_slug}"
    previous_backup = guessed_backup if guessed_backup.exists() else None
    _promote_to_active(
        checkpoint_dir, capability, new_meta,
        previous_backup_path=previous_backup,
        perplexity=result.get("perplexity"),
        assistant_masking_enabled=result.get("assistant_masking_enabled"),
    )

    size_mb = result.get("size_mb", 0.0)
    print(f"  [OK] v{new_version}  ({size_mb:.1f} MB)  → {save_path}")
    return {
        "capability":   capability,
        "adapter_path": str(save_path),
        "version":      new_version,
        "samples":      len(pairs),
        "epochs":       recipe.num_epochs,
        "final_loss":   final_loss,
        "status":       "SUCCESS",
        "validation":   validation_status,
    }


# ---------------------------------------------------------------------------
# Post-training validation
# ---------------------------------------------------------------------------

async def validate_adapter(capability: str, checkpoint_dir: Path) -> Dict[str, Any]:
    """
    Spawn validate_adapter.py as a subprocess so it starts a fresh process with
    no residual GPU allocations from training, then load the adapter and run
    capability-specific prompts + quality checks.

    Returns a dict with keys: status ("PASS"/"FAIL"/"SKIP"), and optional detail.
    """
    import subprocess
    validator = Path(__file__).resolve().parent / "validate_adapter.py"
    if not validator.exists():
        return {"status": "SKIP", "detail": "validate_adapter.py not found"}

    print(f"\n  [VALIDATE] Starting post-train smoke test for '{capability}'...")
    try:
        proc = subprocess.run(
            [
                sys.executable, str(validator),
                "--capability", capability,
                "--adapter-dir", str(checkpoint_dir),
            ],
            # Stream stdout/stderr directly to this terminal so the user sees
            # the generated text and per-check results in real time.
            timeout=300,
        )
        if proc.returncode == 0:
            return {"status": "PASS"}
        else:
            return {"status": "FAIL", "detail": f"exit code {proc.returncode}"}
    except subprocess.TimeoutExpired:
        return {"status": "FAIL", "detail": "validation timed out (> 5 min)"}
    except Exception as exc:
        return {"status": "SKIP", "detail": str(exc)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    """
    Step 1 — Generate/verify datasets (downloads from HF if not cached).
    Step 2 — Train one LoRA adapter per capability via LoRATrainer.
    Step 3 — Optionally run a post-training smoke test per adapter (--validate).
    """
    import argparse
    parser = argparse.ArgumentParser(
        description="Agent LoRA training with real HF datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  Train a single adapter:     python train_gpu_simple.py --capability coding\n"
            "  Train + validate:           python train_gpu_simple.py --capability coding --validate\n"
            "  Train all + validate all:   python train_gpu_simple.py --validate\n"
            "  Re-train fresh + validate:  python train_gpu_simple.py --capability coding --fresh --validate"
        ),
    )
    parser.add_argument("--capability", metavar="CAP", default=None,
                        help="Train a single capability (shorthand for --capabilities <CAP>). "
                             f"Choices: {', '.join(AGENT_SPECS.keys())}")
    parser.add_argument("--capabilities", nargs="*", default=None,
                        help="Subset of capabilities to train (default: all 6). "
                             f"Choices: {', '.join(AGENT_SPECS.keys())}")
    parser.add_argument("--skip-data-gen", action="store_true",
                        help="Skip dataset generation (use existing JSONL files)")
    parser.add_argument("--target-samples", type=int, default=10_000,
                        help="Max samples per capability for data generation")
    parser.add_argument("--fresh", nargs="*", metavar="CAP", default=None,
                        help="Re-train from scratch, discarding existing checkpoint. "
                             "Omit args to apply to all. Example: --fresh coding debugging")
    parser.add_argument("--regen-data", nargs="*", metavar="CAP", default=None,
                        dest="regen_data",
                        help="Force dataset regeneration even if JSONL exists. "
                             "Omit args to apply to all. Example: --regen-data coding")
    parser.add_argument("--validate", action="store_true",
                        help="After each adapter is trained, spawn validate_adapter.py to "
                             "load the adapter, run a capability-specific prompt, and verify "
                             "output quality. Catches corrupted saves before they reach production.")
    parser.add_argument("--probe", action="store_true",
                        help="Preflight check: train 50 samples × 1 epoch for the first "
                             "capability in a temp dir, validate weight files exist, run "
                             "validate_adapter.py on the probe weights, then exit 0 (pass) "
                             "or 3 (fail). Run before a long bootstrap to catch broken "
                             "recipes, OOM conditions, or data issues early.")
    parser.add_argument("--validate-only",
                        metavar="CAP",
                        help="Run validate_adapter.py on an existing adapter without training")
    args = parser.parse_args()
    
    if args.validate_only:
        cap = args.validate_only
        if cap not in AGENT_SPECS:
            parser.error(f"Unknown capability '{cap}'")

        checkpoint_dir = Path(os.environ.get("TRAINING__OUTPUT_DIR", "/data/lora_checkpoints"))
        result = await validate_adapter(cap, checkpoint_dir)

        print(f"\n[VALIDATE-ONLY] {cap}: {result}")
        sys.exit(0 if result["status"] == "PASS" else 3)


    # --capability (singular) is a shorthand for --capabilities with one value.
    # If both are given, --capability takes precedence with a warning.
    if args.capability is not None:
        if args.capabilities is not None:
            print(f"  [WARN] --capability and --capabilities both set; using --capability '{args.capability}'")
        capabilities = [args.capability]
    else:
        capabilities = args.capabilities or list(AGENT_SPECS.keys())

    # Validate capability names early
    unknown = [c for c in capabilities if c not in AGENT_SPECS]
    if unknown:
        parser.error(f"Unknown capabilities: {unknown}. Valid: {list(AGENT_SPECS.keys())}")

    fresh_caps: set = (
        set() if args.fresh is None
        else set(capabilities) if args.fresh == []
        else set(args.fresh)
    )
    regen_caps: set = (
        set() if args.regen_data is None
        else set(capabilities) if args.regen_data == []
        else set(args.regen_data)
    )

    print("\n" + "="*70)
    print("  AGENT-SPECIFIC LORA TRAINING  (via LoRATrainer)")
    print(f"  Capabilities : {capabilities}")
    from shared.contracts.training import load_recipes as _load_recipes
    _dr = _load_recipes().default
    print(f"  LoRA rank    : {_dr.r}  alpha={_dr.alpha}  epochs={_dr.num_epochs}  (default recipe; per-capability may differ)")
    print("="*70)

    setup_hf_auth()
    print_device_info()

    # ── Step 1: dataset generation ───────────────────────────────────────────
    if not args.skip_data_gen:
        print("\n" + "="*70)
        print("  STEP 1: DATASET GENERATION")
        print("="*70)
        force_regen = [c for c in capabilities if c in regen_caps]
        missing     = [c for c in capabilities
                       if c not in regen_caps
                       and not (DATA_DIR / f"{c}_dataset.jsonl").exists()]
        to_generate = list(dict.fromkeys(force_regen + missing))
        if to_generate:
            if force_regen:
                print(f"  Force-regenerating : {force_regen}")
            if missing:
                print(f"  Generating missing : {missing}")
            generate_datasets(capabilities=to_generate, target=args.target_samples)
        else:
            print("  All datasets already exist — skipping generation.")
            for cap in capabilities:
                p = DATA_DIR / f"{cap}_dataset.jsonl"
                n = sum(1 for _ in open(p, encoding="utf-8"))
                print(f"    {cap:12s}  {n:>7,} samples  {p}")
    else:
        print("\n  [--skip-data-gen] Using existing JSONL files.")

    # ── Step 1b: preflight probe ─────────────────────────────────────────────
    # Trains 50 samples × 1 epoch in an isolated temp dir using the FIRST
    # capability, validates weight files exist, then runs validate_adapter.py
    # on the probe adapter.  Exits 0 on pass, 3 on fail.
    # Use this before committing a multi-hour full bootstrap run.
    if args.probe:
        probe_cap = capabilities[0]
        probe_recipe = _resolve_recipe(probe_cap)
        probe_model_id, probe_sys_prompt = AGENT_SPECS[probe_cap]

        print("\n" + "="*70)
        print(f"  PREFLIGHT PROBE  (capability={probe_cap}, 50 samples × 1 epoch)")
        print("="*70)

        # 9a: Use 200 samples × 2 epochs so there are enough optimiser steps to
        # produce a measurable loss decrease.  50 × 1 (old default) gave ~6 steps —
        # too few to distinguish a working training loop from random noise.
        _PROBE_SAMPLES = 200
        _PROBE_EPOCHS  = 2
        probe_pairs = load_sft_pairs(probe_cap, probe_sys_prompt, probe_recipe)[:_PROBE_SAMPLES]
        if not probe_pairs:
            print(f"  [PROBE] FAIL: no training pairs found for '{probe_cap}'")
            sys.exit(3)
        print(f"  [PROBE] {len(probe_pairs)} samples loaded (target={_PROBE_SAMPLES})")

        import time as _ptime
        _pt0 = _ptime.monotonic()

        with tempfile.TemporaryDirectory(prefix="bootstrap_probe_") as _tdir:
            _probe_save = Path(_tdir) / f"{probe_cap}_lora"
            _probe_save.mkdir(parents=True, exist_ok=True)

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
            ) as _tmp:
                for p in probe_pairs:
                    _tmp.write(json.dumps(p) + "\n")
                _probe_dataset = Path(_tmp.name)

            try:
                _probe_trainer = LoRATrainer(
                    base_model=probe_model_id, output_dir=str(_probe_save)
                )
                _probe_result = await _probe_trainer.train(
                    dataset_path=_probe_dataset,
                    num_epochs=_PROBE_EPOCHS,
                    learning_rate=probe_recipe.learning_rate,
                    lora_rank=probe_recipe.r,
                    lora_alpha=probe_recipe.alpha,
                    recipe=probe_recipe,
                )
            except Exception as exc:
                print(f"  [PROBE] FAIL: training error: {exc}")
                sys.exit(3)
            finally:
                _probe_dataset.unlink(missing_ok=True)

            _PROBE_WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin")
            _pw = next(
                (_probe_save / f for f in _PROBE_WEIGHT_FILES if (_probe_save / f).exists()),
                None,
            )
            if _pw is None or _pw.stat().st_size == 0:
                print("  [PROBE] FAIL: no adapter weight file written after probe training")
                sys.exit(3)
            print(
                f"  [PROBE] Weights OK: {_pw.name}  ({_pw.stat().st_size / 1e6:.1f} MB)"
                f"  loss={_probe_result.get('final_loss', 'n/a')}"
            )

            # 9a: Loss-drop assertion — confirm training signal is real.
            # Reads the per-step metrics file written by _MetricsStreamCallback.
            _probe_metrics = _probe_save / "train_metrics.jsonl"
            if _probe_metrics.exists():
                import json as _json
                _steps = []
                with open(_probe_metrics, encoding="utf-8") as _mf:
                    for _ml in _mf:
                        try:
                            _e = _json.loads(_ml.strip())
                            if "loss" in _e:
                                _steps.append(_e["loss"])
                        except Exception:
                            pass
                if len(_steps) >= 4:
                    _init_loss = _steps[0]
                    _final_loss = _steps[-1]
                    _drop_ratio = _final_loss / _init_loss if _init_loss else 1.0
                    print(
                        f"  [PROBE] Loss trajectory: {_init_loss:.4f} → {_final_loss:.4f}"
                        f"  (ratio={_drop_ratio:.3f}, threshold<0.85)"
                    )
                    if _drop_ratio >= 0.85:
                        print(
                            f"  [PROBE] FAIL: loss did not decrease enough ({_drop_ratio:.3f} ≥ 0.85) — "
                            "possible broken recipe, mask bug, or corrupt data."
                        )
                        sys.exit(3)
                else:
                    print(f"  [PROBE] Loss-drop check skipped: only {len(_steps)} steps logged.")
            else:
                print("  [PROBE] Loss-drop check skipped: train_metrics.jsonl not found.")

            val = await validate_adapter(probe_cap, Path(_tdir))
            _probe_dur = round(_ptime.monotonic() - _pt0, 1)
            print(f"  [PROBE] validate_adapter: {val['status']}"
                  + (f"  ({val.get('detail', '')})" if val.get("detail") else ""))

            if val["status"] == "FAIL":
                print(
                    f"  [PROBE] FAIL — environment not ready for full bootstrap "
                    f"(duration {_probe_dur}s).  Fix the issue then re-run."
                )
                sys.exit(3)
            # Temp dir cleaned up on context exit.

        print(
            f"  [PROBE] PASS ({_probe_dur}s) — GPU, recipe, and data pipeline verified.\n"
            "  Re-run without --probe to start full training."
        )
        print("="*70 + "\n")
        sys.exit(0)

    # ── Step 2: training ─────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  STEP 2: LORA TRAINING")
    print("="*70)

    checkpoint_dir = Path(os.environ.get("TRAINING__OUTPUT_DIR", "/data/lora_checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict[str, Any]] = []
    for cap in capabilities:
        try:
            result = await train_capability(
                capability=cap,
                checkpoint_dir=checkpoint_dir,
                fresh=(cap in fresh_caps),
                validate=args.validate,
            )
            all_results.append(result)
        except Exception as exc:
            import traceback
            print(f"  [ERROR] {cap}: {exc}")
            traceback.print_exc()
            all_results.append({"capability": cap, "status": "FAILED", "error": str(exc)})

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  TRAINING COMPLETE — SUMMARY")
    print("="*70)
    for r in all_results:
        cap    = r["capability"]
        status = r["status"]
        if status == "SUCCESS":
            val_tag = ""
            if "validation" in r:
                v = r["validation"]
                val_tag = f"  validate={v}"
            print(f"  [OK]   {cap:12s}  v{r['version']}  samples={r['samples']:,}  "
                  f"loss={r['final_loss']:.4f}{val_tag}  → {r['adapter_path']}")
        else:
            print(f"  [FAIL] {cap:12s}  {r.get('error', '')}")

    successes   = sum(1 for r in all_results if r["status"] == "SUCCESS")
    val_passes  = sum(1 for r in all_results if r.get("validation") == "PASS")
    val_fails   = sum(1 for r in all_results if r.get("validation") == "FAIL")
    print(f"\n  {successes}/{len(all_results)} adapters trained successfully")
    if args.validate:
        print(f"  {val_passes}/{successes} validation smoke tests PASSED"
              + (f"  ({val_fails} FAILED — check adapter before deploying)" if val_fails else ""))
    print(f"  Output: {checkpoint_dir.resolve()}")
    print("="*70 + "\n")

    return all_results


if __name__ == "__main__":
    asyncio.run(main())
