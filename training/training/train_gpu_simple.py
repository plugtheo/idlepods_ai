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

# lora_trainer.py and generate_data.py live in the same directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_data import generate_datasets, TARGET_DIR as DATA_DIR
from lora_trainer import LoRATrainer, _pre_train_backup, _post_train_version

# Add repo root to sys.path so shared/ is importable (mirrors the Docker PYTHONPATH=/app).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from shared.contracts.agent_prompts import AGENT_PROMPTS, BOOTSTRAP_CAP_TO_ROLE
from shared.contracts.training import lookup_recipe
from shared.contracts.experience import AgentContribution
from orchestration.app.experience.sft_builder import build_sft_pair

# ---------------------------------------------------------------------------
# Agent capability → (base model id, system description)
# Mirrors ModelRegistry._CAPABILITY_MODEL_MAP and local_models.json
# ---------------------------------------------------------------------------
def _resolve_base_model_id() -> str:
    from shared.contracts.models import load_registry
    registry = load_registry()
    return registry.backends[registry.default_backend].model_id


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
            instruction = rec.get("instruction", "").strip()
            response    = rec.get("response", "").strip()
            if instruction and response:
                contrib = AgentContribution(
                    role=role_name,
                    output=response,
                    quality_score=1.0,
                    iteration=1,
                )
                sft_pair = build_sft_pair(
                    contrib, recipe, role_name,
                    system_prompt=system_prompt, user_prompt=instruction,
                )
                pairs.append(sft_pair)
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

    # ── Resolve recipe ───────────────────────────────────────────────────────
    recipe = _resolve_recipe(capability)

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
        )
    finally:
        dataset_path.unlink(missing_ok=True)

    # ── Post-training: version bump, weight validation, manifest update ───────
    final_loss  = result.get("final_loss", 0.0)
    note        = "incremental" if has_existing else "fresh"
    new_version = _post_train_version(
        save_path=save_path,
        checkpoint_dir=checkpoint_dir,
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

    size_mb = result.get("size_mb", 0.0)
    print(f"  [OK] v{new_version}  ({size_mb:.1f} MB)  → {save_path}")
    return {
        "capability":   capability,
        "adapter_path": str(save_path),
        "version":      new_version,
        "samples":      len(pairs),
        "epochs":       NUM_EPOCHS,
        "final_loss":   final_loss,
        "status":       "SUCCESS",
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
    args = parser.parse_args()

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
            )
            # ── Post-train validation ─────────────────────────────────────────
            if args.validate:
                val = await validate_adapter(cap, checkpoint_dir)
                result["validation"] = val["status"]
                if val["status"] == "FAIL":
                    detail = val.get("detail", "")
                    print(f"  [VALIDATE] FAIL for '{cap}': {detail}")
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
