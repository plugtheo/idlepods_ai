#!/usr/bin/env python3
"""
LoRA Trainer for Agent-Generated Solutions
Tailored pipeline for agent outputs with:
  - problem, context, solution, evaluation, improvement, lessons_learned

Pipeline:
  1. Collect agent runs (JSONL format)
  2. Filter successful runs (eval score >= 0.65)
  3. Build training dataset (problem+context+lessons → solution)
  4. Prepare instructions (instruction-response pairs)
  5. Train LoRA adapter
"""

import asyncio
import hashlib
import importlib.metadata
import json
import os
import shutil
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import random
import re

from shared.manifest import write_manifest_locked, LegacyManifestError  # noqa: F401
from shared.contracts.manifest_schema import AdapterEntry, HistoryEntry, Manifest

from shared.contracts.agent_prompts import AGENT_PROMPTS as _AGENT_PROMPTS, BOOTSTRAP_CAP_TO_ROLE as _CAP_TO_ROLE
from shared.contracts.training import AdapterRecipe


def _resolve_registry_default() -> str:
    try:
        from shared.contracts.models import load_registry
        return load_registry().default_backend
    except Exception:
        return "primary"


# ---------------------------------------------------------------------------
# Versioning helpers — imported by train_gpu_simple.py and trainer_entry.py
# so both the one-time bootstrap path and the online self-training path use
# identical versioning, backup, and manifest-update logic.
# ---------------------------------------------------------------------------

_WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin")
_ADAPTER_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",   # attention
    "gate_proj", "up_proj", "down_proj",        # FFN / MLP
]



def _compute_trainer_version() -> str:
    parts = []
    for pkg in ("trl", "unsloth"):
        try:
            parts.append(f"{pkg}=={importlib.metadata.version(pkg)}")
        except importlib.metadata.PackageNotFoundError:
            pass
    return "|".join(parts) or "unknown"


def _pre_train_backup(save_path: Path, capability: str) -> tuple:
    """
    Back up the current adapter BEFORE training overwrites it.

    Must be called BEFORE LoRATrainer.train() so the backup contains the
    previous working adapter weights, not the newly trained ones.  A backup
    that is taken after training is useless for rollback.

    Returns:
        (old_version: str, backup_path: Optional[Path])
        old_version defaults to "1.0.0" when no existing metadata is found.
        backup_path is None when no existing adapter weights are present.
    """
    meta_path = save_path / "metadata.json"
    old_version = "1.0.0"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            old_version = meta.get("version", "1.0.0")
        except (json.JSONDecodeError, OSError):
            pass

    # Only backup when real weight files exist (skip empty/initialised-but-empty dirs).
    has_weights = save_path.exists() and any(
        (save_path / f).exists() for f in _WEIGHT_FILES
    )
    if not has_weights:
        return old_version, None

    old_slug    = old_version.replace(".", "_")
    backup_path = save_path.parent / f"{capability}_lora_v{old_slug}"
    if backup_path.exists():
        shutil.rmtree(backup_path)
    shutil.copytree(save_path, backup_path)
    return old_version, backup_path


def _post_train_stage(
    save_path: Path,
    capability: str,
    old_version: str,
    base_model_id: str,
    n_samples: int,
    num_epochs: int,
    learning_rate: float,
    lora_r: int,
    lora_alpha: int,
    final_loss: float,
    note: str = "bootstrap",
    dataset_hash: str = "unknown",
    tokenizer_hash: str = "unknown",
    trainer_version: str = "",
    experience_record_count: int = 0,
    synthetic_record_count: int = 0,
) -> Dict[str, Any]:
    """
    Validate adapter weights and write metadata.json with status='staging'.
    Does NOT touch the root manifest — call _promote_to_active after smoke passes.

    Raises RuntimeError if no adapter weight file is found.
    Returns the new metadata dict (includes bumped version).
    """
    weight_file = save_path / "adapter_model.safetensors"
    if not weight_file.exists():
        weight_file = save_path / "adapter_model.bin"
    if not weight_file.exists():
        raise RuntimeError(
            f"No adapter weight file found at {save_path} after training completed. "
            "Possible causes: out-of-disk-space, Unsloth save error, or mock mode. "
            "The previous adapter backup (if any) is intact and can be restored."
        )

    size_mb = round(weight_file.stat().st_size / 1e6, 2)

    old_history: List[Any] = []
    old_created_at: str = ""
    meta_path = save_path / "metadata.json"
    if meta_path.exists():
        try:
            old_meta = json.loads(meta_path.read_text())
            old_history    = old_meta.get("history", [])
            old_created_at = old_meta.get("created_at", "")
        except (json.JSONDecodeError, OSError):
            pass

    try:
        major, minor, patch = [int(x) for x in old_version.split(".")]
    except (ValueError, AttributeError):
        major, minor, patch = 1, 0, 0
    new_version = f"{major}.{minor + 1}.0"

    now_iso    = datetime.now(timezone.utc).isoformat()
    created_at = old_created_at or now_iso

    _tv = trainer_version or _compute_trainer_version()
    _total = experience_record_count + synthetic_record_count
    _exp_pct = round(experience_record_count / _total * 100, 1) if _total > 0 else 0.0

    new_history_entry: Dict[str, Any] = {
        "version":                new_version,
        "status":                 "staging",
        "created_at":             now_iso,
        "n_samples":              n_samples,
        "epochs":                 num_epochs,
        "learning_rate":          learning_rate,
        "lora_r":                 lora_r,
        "lora_alpha":             lora_alpha,
        "final_loss":             round(final_loss, 6),
        "size_mb":                size_mb,
        "note":                   note,
        "dataset_hash":           dataset_hash,
        "tokenizer_hash":         tokenizer_hash,
        "trainer_version":        _tv,
        "experience_record_count": experience_record_count,
        "synthetic_record_count":  synthetic_record_count,
        "experience_data_pct":     _exp_pct,
    }

    new_meta: Dict[str, Any] = {
        "name":                   f"{capability}_lora",
        "version":                new_version,
        "capability":             capability,
        "backend":                _resolve_registry_default(),
        "base_model":             base_model_id,
        "status":                 "staging",
        "created_at":             created_at,
        "updated_at":             now_iso,
        "lora_r":                 lora_r,
        "lora_alpha":             lora_alpha,
        "target_modules":         _ADAPTER_TARGET_MODULES,
        "size_mb":                size_mb,
        "dataset_hash":           dataset_hash,
        "tokenizer_hash":         tokenizer_hash,
        "trainer_version":        _tv,
        "experience_record_count": experience_record_count,
        "synthetic_record_count":  synthetic_record_count,
        "experience_data_pct":     _exp_pct,
        "history":                old_history + [new_history_entry],
    }
    meta_path.write_text(json.dumps(new_meta, indent=2))

    print(
        f"  [STAGE] {capability}_lora  {old_version} → {new_version}"
        f"  ({size_mb:.1f} MB)  status=staging"
    )
    return new_meta


def _promote_to_active(
    checkpoint_dir: Path,
    capability: str,
    new_meta: Dict[str, Any],
    previous_backup_path: Optional[Path] = None,
    smoke_results: Optional[Dict[str, Any]] = None,
    eval_results: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Under manifest file-lock: flip active/previous pointers and write v2 HistoryEntry.
    Also updates metadata.json status from 'staging' to 'active'.
    """
    adapter_name = f"{capability}_lora"
    new_version  = new_meta["version"]
    now_iso      = datetime.now(timezone.utc).isoformat()

    meta_path = checkpoint_dir / adapter_name / "metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            meta["status"] = "active"
            meta["updated_at"] = now_iso
            for entry in meta.get("history", []):
                if entry.get("version") == new_version:
                    entry["status"] = "active"
                    if smoke_results:
                        entry["smoke"] = smoke_results
                    if eval_results:
                        entry["eval"] = eval_results
                    break
            meta_path.write_text(json.dumps(meta, indent=2))
        except (json.JSONDecodeError, OSError):
            pass

    last_history = (new_meta.get("history") or [{}])[-1]
    _recipe_from_meta = last_history.get("recipe", {})
    _sft_format = _recipe_from_meta.get("sft_format", "openai_messages")
    trained_at_dt = datetime.fromisoformat(
        new_meta.get("updated_at", now_iso).replace("Z", "+00:00")
    )
    new_history_entry = HistoryEntry(
        version=new_version,
        status="active",
        trained_at=trained_at_dt,
        backend=new_meta.get("backend", _resolve_registry_default()),
        base_model=new_meta.get("base_model", "unknown"),
        peft_type=_recipe_from_meta.get("peft_type", "lora"),
        target_modules=new_meta.get("target_modules", _ADAPTER_TARGET_MODULES),
        r=new_meta.get("lora_r", 8),
        alpha=new_meta.get("lora_alpha", 16),
        dropout=0.0,
        recipe={
            "peft_type": _recipe_from_meta.get("peft_type", "lora"),
            "r": new_meta.get("lora_r", 8),
            "alpha": new_meta.get("lora_alpha", 16),
            "target_modules": new_meta.get("target_modules", _ADAPTER_TARGET_MODULES),
            "sft_format": _sft_format,
        },
        dataset_hash=new_meta.get("dataset_hash", "unknown"),
        tokenizer_hash=new_meta.get("tokenizer_hash", "unknown"),
        trainer_version=new_meta.get("trainer_version", _compute_trainer_version()),
        n_samples=last_history.get("n_samples", 0),
        final_loss=last_history.get("final_loss", 0.0),
        size_mb=last_history.get("size_mb", 0.0),
        eval_metrics={k: float(v) for k, v in (eval_results or {}).items() if isinstance(v, (int, float))},
        smoke=smoke_results or {},
    )

    manifest_path = checkpoint_dir / "manifest.json"
    _captured: Dict[str, str] = {}

    def _mutator(m: Manifest) -> None:
        existing = m.adapters.get(adapter_name)
        old_active_version = existing.active_version if existing else ""
        old_active_path    = existing.active_path if existing else str(checkpoint_dir / adapter_name)
        _captured["old_active_version"] = old_active_version

        if existing:
            updated_history = []
            for h in existing.history:
                if h.status == "active":
                    updated_history.append(h.model_copy(update={"status": "retired"}))
                else:
                    updated_history.append(h)
            updated_history.append(new_history_entry)
        else:
            updated_history = [new_history_entry]

        prev_path = (
            str(previous_backup_path)
            if previous_backup_path and Path(previous_backup_path).exists()
            else old_active_path
        )
        m.adapters[adapter_name] = AdapterEntry(
            schema_version=2,
            active_version=new_version,
            active_path=str(checkpoint_dir / adapter_name),
            previous_version=old_active_version,
            previous_path=prev_path,
            backend=new_meta.get("backend", _resolve_registry_default()),
            updated_at=datetime.now(timezone.utc),
            history=updated_history,
        )

    write_manifest_locked(manifest_path, _mutator)
    print(f"  [VERSION] {adapter_name}  {_captured.get('old_active_version','')} → {new_version}  status=active")


def _mark_failed(
    checkpoint_dir: Path,
    capability: str,
    version: str,
    smoke_results: Optional[Dict[str, Any]] = None,
) -> None:
    """Under manifest file-lock: mark the staged version as failed."""
    adapter_name = f"{capability}_lora"
    now_iso      = datetime.now(timezone.utc).isoformat()

    meta_path = checkpoint_dir / adapter_name / "metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            meta["status"] = "failed"
            meta["updated_at"] = now_iso
            for entry in meta.get("history", []):
                if entry.get("version") == version:
                    entry["status"] = "failed"
                    if smoke_results:
                        entry["smoke"] = smoke_results
                    break
            meta_path.write_text(json.dumps(meta, indent=2))
        except (json.JSONDecodeError, OSError):
            pass

    manifest_path = checkpoint_dir / "manifest.json"

    def _mutator(m: Manifest) -> None:
        entry = m.adapters.get(adapter_name)
        if entry is None:
            return
        updated = []
        for h in entry.history:
            if h.version == version and h.status == "staging":
                updated.append(h.model_copy(update={
                    "status": "failed",
                    "smoke": smoke_results or {},
                }))
            else:
                updated.append(h)
        m.adapters[adapter_name] = entry.model_copy(update={"history": updated})

    if manifest_path.exists():
        write_manifest_locked(manifest_path, _mutator)

    print(f"  [FAILED] {adapter_name}  v{version}  smoke_pass=False")


def _post_train_version(
    save_path: Path,
    checkpoint_dir: Path,
    capability: str,
    old_version: str,
    base_model_id: str,
    n_samples: int,
    num_epochs: int,
    learning_rate: float,
    lora_r: int,
    lora_alpha: int,
    final_loss: float,
    note: str = "bootstrap",
) -> str:
    """
    Legacy wrapper used by train_gpu_simple.py (no smoke gate).
    Calls _post_train_stage then _promote_to_active immediately.
    Returns the new version string.
    """
    new_meta = _post_train_stage(
        save_path=save_path,
        capability=capability,
        old_version=old_version,
        base_model_id=base_model_id,
        n_samples=n_samples,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        final_loss=final_loss,
        note=note,
    )
    # Try to find the pre-existing backup so previous_path is populated.
    old_slug = old_version.replace(".", "_")
    guessed_backup = checkpoint_dir / f"{capability}_lora_v{old_slug}"
    previous_backup = guessed_backup if guessed_backup.exists() else None
    _promote_to_active(checkpoint_dir, capability, new_meta, previous_backup_path=previous_backup)
    return new_meta["version"]


def apply_recipe(model, recipe: "AdapterRecipe"):
    """Apply LoRA/RS-LoRA/DoRA/QLoRA PEFT config to model using Unsloth."""
    print(
        f"apply_recipe peft_type={recipe.peft_type} use_rslora={getattr(recipe, 'use_rslora', False)} "
        f"use_dora={getattr(recipe, 'use_dora', False)} r={recipe.r} alpha={recipe.alpha} "
        f"target_modules={recipe.target_modules}",
        flush=True,
    )
    from unsloth import FastLanguageModel
    available = {name.split(".")[-1] for name, _ in model.named_modules()}
    missing = [m for m in recipe.target_modules if m not in available]
    if missing:
        raise ValueError(
            f"target_modules {missing} not found in model. Available: {sorted(available)}"
        )
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
    )


class LoRATrainer:
    """Train LoRA adapter on prepared dataset"""
    
    def __init__(self,
                 base_model: str,
                 output_dir: str = "./data/adapters"):
        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.adapter_id = None
        self.HAS_UNSLOTH = self._check_unsloth()
    
    def _check_unsloth(self) -> bool:
        try:
            import unsloth  # noqa: F401
            print("[TRAINER] [OK] Unsloth available (2-5x faster LoRA training)")
            return True
        except (ImportError, NotImplementedError, Exception):
            print("[TRAINER] [INFO] Unsloth not available — will use HF PEFT fallback")
            return False

    def _check_peft_gpu(self) -> bool:
        try:
            import torch
            if not torch.cuda.is_available():
                return False
            import transformers  # noqa: F401
            import peft  # noqa: F401
            import trl  # noqa: F401
            print("[TRAINER] [OK] HF PEFT+TRL available — GPU training without Unsloth")
            return True
        except ImportError:
            return False
    
    async def train(self,
                   dataset_path: Path,
                   num_epochs: int = 3,
                   learning_rate: float = 2e-4,
                   lora_rank: int = 16,
                   lora_alpha: int = 32,
                   recipe: Optional[AdapterRecipe] = None,
                   resume_from_checkpoint: Optional[str] = None) -> Dict[str, Any]:
        """
        Train LoRA adapter on dataset.
        """
        
        # TODO: Consolidate and move to helpers.py
        if recipe is None or not hasattr(recipe, "max_seq_length") or recipe.max_seq_length is None:
            raise RuntimeError("LoRATrainer.train() requires a recipe with max_seq_length override")
        
        print(f"\n{'='*70}")
        print("LORA TRAINING")
        print(f"{'='*70}")
        print(f"[TRAINER] Base model: {self.base_model}")
        print(f"[TRAINER] Dataset: {dataset_path}")
        print(f"[TRAINER] Epochs: {num_epochs}")
        print(f"[TRAINER] LoRA rank: {lora_rank}, alpha: {lora_alpha}")
        print(f"[TRAINER] Learning rate: {learning_rate}")
        print(f"[TRAINER] Max sequence length: {recipe.max_seq_length}")
        
        # Load dataset
        dataset = []
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                
                record = json.loads(line)
                # Only keep records that are in a supported format
                if isinstance(record, list):
                    for r in record:
                        if "messages" in r:
                            dataset.append(r)
                        elif "prompt" in r and "completion" in r:
                            dataset.append(r)
                    continue
                if "messages" in record:
                    dataset.append(record)
                elif "prompt" in record and "completion" in record:
                    dataset.append(record)

        print(f"[TRAINER] Loaded {len(dataset)} training samples")
        
        if self.HAS_UNSLOTH:
            return await self._train_unsloth(dataset, num_epochs, learning_rate, lora_rank, lora_alpha, recipe=recipe, resume_from_checkpoint=resume_from_checkpoint)
        else:
            return await self._train_mock(dataset, num_epochs)

    async def _train_unsloth(self,
                             dataset: List[Dict],
                             num_epochs: int,
                             learning_rate: float,
                             lora_rank: int,
                             lora_alpha: int,
                             recipe: Optional[AdapterRecipe] = None,
                             resume_from_checkpoint: Optional[str] = None) -> Dict[str, Any]:
        """Train with actual Unsloth (requires GPU)"""
        import time as _time
        _t_start = _time.monotonic()
        try:
            from unsloth import FastLanguageModel
            from trl import SFTTrainer, SFTConfig

            # TODO: Consolidate and move to helpers.py
            if recipe is None or not hasattr(recipe, "max_seq_length") or recipe.max_seq_length is None:
                raise RuntimeError("LoRATrainer.train() requires a recipe with max_seq_length override")

            print(
                f"\n[TRAINER] batch_size={recipe.per_device_train_batch_size} "
                f"grad_accum={recipe.gradient_accumulation_steps} "
                f"effective_batch={recipe.per_device_train_batch_size * recipe.gradient_accumulation_steps} "
                f"packing={recipe.packing} "
                f"max_seq_length={recipe.max_seq_length} "
                f"load_in_4bit={recipe.load_in_4bit}",
                flush=True,
            )
            print("\n[TRAINER] Loading Unsloth model...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.base_model,
                max_seq_length=recipe.max_seq_length,
                dtype=None,
                load_in_4bit=recipe.load_in_4bit,
            )

            # Ensure pad token is set — required for batched training.
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # One-shot validation: render a sample tool-call record through tokenizer.apply_chat_template and inspect for the markers - hard fail
            self._validate_qwen3_toolcall_wrapping(tokenizer)

            # Apply LoRA via recipe (preferred) or explicit rank/alpha fallback.
            if recipe is not None:
                model = apply_recipe(model, recipe)
            else:
                model = FastLanguageModel.get_peft_model(
                    model,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules=_ADAPTER_TARGET_MODULES,
                    lora_dropout=0.0,
                    bias="none",
                    use_gradient_checkpointing="unsloth",
                )
            
            print("[TRAINER] Preparing training data...")

            # Pre-render every record to a single "text" column so SFTTrainer
            # has an unambiguous shape and Unsloth's compiled wrapper does not
            # raise "must specify a formatting_func".
            # - openai_messages records → tokenizer.apply_chat_template(messages)
            # - legacy prompt+completion records → simple concat
            # completion_only_loss CANNOT be used with text-only datasets.
            _has_messages_records = any("messages" in r for r in dataset)

            sft_config_kwargs = dict(
                per_device_train_batch_size=recipe.per_device_train_batch_size,
                gradient_accumulation_steps=recipe.gradient_accumulation_steps,
                warmup_ratio=0.05,
                num_train_epochs=num_epochs,
                learning_rate=learning_rate,
                fp16=False,
                bf16=True,
                logging_steps=10,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                output_dir=str(self.output_dir / "checkpoint"),
                max_seq_length=recipe.max_seq_length,
                # Never write HF checkpoints during training — we do a single
                # explicit save after training completes.  Without this,
                # SFTTrainer dumps ~200 MB checkpoint dirs every 500 steps
                # (7+ times per capability), consuming ~1.4 GB of wasted disk.
                save_strategy="no",
                packing=recipe.packing,
                # Suppress wandb / tensorboard init — not needed in training
                # containers and causes hangs if wandb credentials are absent.
                report_to="none",
                dataset_num_proc=4,
                dataloader_num_workers=2,
                dataset_text_field=None,
                dataset_kwargs={"formatting_func": None}
            )
            # assistant_only_loss: only backprop on assistant tokens (using the
            # chat template's {% generation %} markers).  Best for chat-format
            # SFT.  Falls back to full-text loss if the TRL/template version
            # doesn't accept it.
            if _has_messages_records:
                try:
                    sft_config = SFTConfig(**sft_config_kwargs, assistant_only_loss=True)
                except TypeError:
                    sft_config = SFTConfig(**sft_config_kwargs)
            else:
                sft_config = SFTConfig(**sft_config_kwargs)

            from datasets import Dataset as HFDataset
            train_dataset = HFDataset.from_list(dataset)

            def _formatting_func(example):
                # SFTTrainer may call this with either a single example (dict of
                # scalars) or a batch (dict of lists).  Handle both.
                t = example["text"]
                return t if isinstance(t, list) else [t]

            _metrics_path = self.output_dir / "train_metrics.jsonl"
            _callbacks = []
            try:
                from transformers import TrainerCallback

                class _MetricsStreamCallback(TrainerCallback):
                    def on_log(self, args, state, control, logs=None, **kwargs):
                        if not logs:
                            return
                        entry = {"step": state.global_step, "epoch": state.epoch, **logs}
                        try:
                            with open(_metrics_path, "a", encoding="utf-8") as _fh:
                                _fh.write(json.dumps(entry) + "\n")
                        except OSError:
                            pass

                _callbacks = [_MetricsStreamCallback()]
            except ImportError:
                pass

            trainer = SFTTrainer(
                model=model,
                processing_class=tokenizer,
                train_dataset=train_dataset,
                args=sft_config,
                formatting_func=_formatting_func,
                callbacks=_callbacks if _callbacks else None,
            )

            print("[TRAINER] Starting Unsloth training...")
            if resume_from_checkpoint:
                print(f"[TRAINER] Warm-starting from prior adapter: {resume_from_checkpoint}")
                trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            else:
                trainer.train()

            self.adapter_id = self.output_dir.name
            model.save_pretrained(str(self.output_dir))

            # Guard against silent save failures (out-of-disk-space, Unsloth
            # version mismatch, etc.) where save_pretrained() returns without
            # error but writes no weight files.  Raise here so the subprocess
            # exits non-zero and callers can treat the run as FAILED rather than
            # promoting an empty adapter directory.
            weight_file = self.output_dir / "adapter_model.safetensors"
            if not weight_file.exists():
                weight_file = self.output_dir / "adapter_model.bin"
            if not weight_file.exists():
                raise RuntimeError(
                    f"save_pretrained() completed but no adapter weight file was "
                    f"written to {self.output_dir}. Check disk space and Unsloth version."
                )
            size_mb = round(weight_file.stat().st_size / 1e6, 2)
            print(f"[TRAINER] [OK] Adapter weights saved: {self.output_dir}  ({size_mb:.1f} MB)")

            # Extract final training loss from SFTTrainer log history so callers
            # (train_gpu_simple.py, trainer_entry.py) can record it in metadata.
            final_loss = 0.0
            if hasattr(trainer, "state") and trainer.state.log_history:
                for entry in reversed(trainer.state.log_history):
                    if "loss" in entry:
                        final_loss = float(entry["loss"])
                        break

            duration_s = round(_time.monotonic() - _t_start, 1)
            print(
                f"training_complete method=unsloth role={self.output_dir.name} "
                f"epochs={num_epochs} samples={len(dataset)} "
                f"final_loss={final_loss:.6f} duration_s={duration_s}",
                flush=True,
            )

            return {
                'status':     'success',
                'adapter_id': self.adapter_id,
                'method':     'unsloth',
                'epochs':     num_epochs,
                'samples':    len(dataset),
                'lora_rank':  lora_rank,
                'size_mb':    size_mb,
                'final_loss': round(final_loss, 6),
                'duration_s': duration_s,
            }

        except Exception as e:
            duration_s = round(_time.monotonic() - _t_start, 1)
            print(
                f"training_failed method=unsloth duration_s={duration_s} error={e}",
                flush=True,
            )
            # Re-raise so the subprocess exits non-zero and the Training
            # Service can record the failure accurately.
            raise
    
    async def _train_mock(self,
                         dataset: List[Dict],
                         num_epochs: int) -> Dict[str, Any]:
        """
        Dry-run path used only in unit tests (no GPU, no real model).

        WARNING: This path must NEVER write files to the LoRA checkpoint
        directory.  Writing fake adapter files would corrupt the directory
        and cause vLLM to fail on next startup in local mode.

        In production the trigger route checks torch.cuda.is_available()
        and refuses to launch the subprocess when no GPU is present, so
        this path is only reachable in test environments.
        """
        print(f"\n[TRAINER] Mock training (test mode — no files written) ...")
        
        for epoch in range(1, num_epochs + 1):
            loss = 2.0 - (epoch * 0.15) + random.random() * 0.1
            print(f"[TRAINER] Epoch {epoch}/{num_epochs} - Loss: {loss:.4f} (simulated)")
            await asyncio.sleep(0.1)
        
        self.adapter_id = f"mock_{self.output_dir.name}"
        print(f"[TRAINER] Mock run complete (no GPU, no adapter saved)")
        
        return {
            'status': 'success',
            'adapter_id': self.adapter_id,
            'method': 'mock',
            'epochs': num_epochs,
            'samples': len(dataset),
        }

    def _validate_qwen3_toolcall_wrapping(self, tokenizer):
        """
        Hard validation: Qwen3 chat template must wrap assistant tool-call turns
        inside {% generation %} ... {% endgeneration %}.

        If missing, training with assistant_only_loss=True will silently corrupt
        tool-call learning. This is a blocker for Phase 3.
        """

        sample_messages = [
            {"role": "user", "content": "test"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "t1",
                        "type": "python",
                        "function": {"name": "f", "arguments": "{}"},
                    }
                ],
                "content": None,
            },
        ]

        rendered = tokenizer.apply_chat_template(
            sample_messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        has_start = "{% generation %}" in rendered
        has_end = "{% endgeneration %}" in rendered

        if not (has_start and has_end):
            raise RuntimeError(
                "Qwen3 chat template is missing {% generation %} wrapping for "
                "assistant tool-call turns. Tool-call masking will be incorrect. "
                "Patch the chat template or install a custom assistant-mask collator."
            )

