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


def _resolve_base_model() -> str:
    from shared.contracts.models import load_registry
    registry = load_registry()
    return registry.backends[registry.default_backend].model_id


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
    new_history_entry: Dict[str, Any] = {
        "version":         new_version,
        "status":          "staging",
        "created_at":      now_iso,
        "n_samples":       n_samples,
        "epochs":          num_epochs,
        "learning_rate":   learning_rate,
        "lora_r":          lora_r,
        "lora_alpha":      lora_alpha,
        "final_loss":      round(final_loss, 6),
        "size_mb":         size_mb,
        "note":            note,
        "dataset_hash":    dataset_hash,
        "tokenizer_hash":  tokenizer_hash,
        "trainer_version": _tv,
    }

    new_meta: Dict[str, Any] = {
        "name":            f"{capability}_lora",
        "version":         new_version,
        "capability":      capability,
        "backend":         _resolve_registry_default(),
        "base_model":      base_model_id,
        "status":          "staging",
        "created_at":      created_at,
        "updated_at":      now_iso,
        "lora_r":          lora_r,
        "lora_alpha":      lora_alpha,
        "target_modules":  _ADAPTER_TARGET_MODULES,
        "size_mb":         size_mb,
        "dataset_hash":    dataset_hash,
        "tokenizer_hash":  tokenizer_hash,
        "trainer_version": _tv,
        "history":         old_history + [new_history_entry],
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


class AgentOutputDataset:
    """Manages agent output collection and filtering"""
    
    def __init__(self, results_dir: str = "./data/agent_runs"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.runs: List[Dict[str, Any]] = []
        self.run_file = self.results_dir / f"runs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    
    def add_run(self, run: Dict[str, Any]) -> None:
        """Add a single agent run"""
        required_fields = {"problem", "context", "solution", "evaluation", "improvement", "lessons_learned"}
        if not required_fields.issubset(run.keys()):
            missing = required_fields - set(run.keys())
            raise ValueError(f"Missing fields: {missing}")
        
        run["timestamp"] = datetime.now().isoformat()
        self.runs.append(run)
        self._save_run(run)
    
    def _save_run(self, run: Dict[str, Any]) -> None:
        """Persist run to JSONL"""
        with open(self.run_file, 'a') as f:
            f.write(json.dumps(run) + '\n')
    
    def load_runs_from_file(self, filepath: str) -> None:
        """Load runs from existing JSONL file"""
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    self.runs.append(json.loads(line))
    
    def filter_by_score(self, min_score: float = 0.65) -> List[Dict[str, Any]]:
        """Filter runs by evaluation score threshold"""
        filtered = []
        for run in self.runs:
            score = float(run.get('evaluation', 0))
            if score >= min_score:
                filtered.append(run)
        
        print(f"[FILTER] Total runs: {len(self.runs)}")
        print(f"[FILTER] Min score threshold: {min_score}")
        print(f"[FILTER] High-quality runs: {len(filtered)}")
        print(f"[FILTER] Filtered out: {len(self.runs) - len(filtered)}")
        print(f"[FILTER] Avg score: {sum(float(r.get('evaluation', 0)) for r in self.runs) / len(self.runs):.2f}")
        
        return filtered


class TrainingDatasetBuilder:
    """Builds LoRA training dataset from filtered agent runs"""
    
    def __init__(self, output_dir: str = "./data/training_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset: List[Dict[str, str]] = []
    
    def build_from_filtered_runs(self, filtered_runs: List[Dict[str, Any]]) -> None:
        """
        Create instruction-response pairs from filtered runs.
        
        Format:
        {
            "instruction": "<problem>\nContext: <context>\nLessons from previous: <lessons>",
            "response": "<solution>\nImprovement: <improvement>"
        }
        """
        self.dataset = []
        
        for run in filtered_runs:
            instruction = self._build_instruction(run)
            response = self._build_response(run)
            
            self.dataset.append({
                "instruction": instruction,
                "response": response,
                "evaluation": float(run.get('evaluation', 0)),
                "metadata": {
                    "original_problem": run.get('problem', ''),
                    "lessons_learned": run.get('lessons_learned', '')
                }
            })
        
        print(f"\n[DATASET] Created {len(self.dataset)} training pairs")
        if self.dataset:
            avg_eval = sum(d['evaluation'] for d in self.dataset) / len(self.dataset)
            print(f"[DATASET] Average evaluation score: {avg_eval:.3f}")
    
    def _build_instruction(self, run: Dict[str, Any]) -> str:
        """Build instruction from problem, context, and lessons"""
        parts = [
            f"Problem: {run.get('problem', '')}"
        ]
        
        if run.get('context'):
            parts.append(f"Context: {run.get('context', '')}")
        
        if run.get('lessons_learned'):
            parts.append(f"Lessons from previous: {run.get('lessons_learned', '')}")
        
        return "\n".join(parts)
    
    def _build_response(self, run: Dict[str, Any]) -> str:
        """Build response from solution and improvement"""
        parts = [
            f"Solution: {run.get('solution', '')}"
        ]
        
        if run.get('improvement'):
            parts.append(f"Improvement: {run.get('improvement', '')}")
        
        return "\n".join(parts)
    
    def save_dataset(self, format: str = "jsonl") -> Path:
        """Save dataset to file"""
        if format == "jsonl":
            output_file = self.output_dir / f"lora_training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            with open(output_file, 'w') as f:
                for item in self.dataset:
                    f.write(json.dumps(item) + '\n')
        else:
            output_file = self.output_dir / f"lora_training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w') as f:
                json.dump(self.dataset, f, indent=2)
        
        print(f"[DATASET] Saved to {output_file}")
        return output_file
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        if not self.dataset:
            return {}
        
        avg_instruction_len = sum(len(d['instruction']) for d in self.dataset) / len(self.dataset)
        avg_response_len = sum(len(d['response']) for d in self.dataset) / len(self.dataset)
        
        return {
            'total_pairs': len(self.dataset),
            'avg_instruction_length': int(avg_instruction_len),
            'avg_response_length': int(avg_response_len),
            'avg_evaluation': sum(d['evaluation'] for d in self.dataset) / len(self.dataset),
        }


class LoRATrainer:
    """Train LoRA adapter on prepared dataset"""
    
    def __init__(self,
                 base_model: Optional[str] = None,
                 output_dir: str = "./data/adapters",
                 max_seq_length: int = 2048):
        self.base_model = base_model or _resolve_base_model()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_seq_length = max_seq_length
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
                   recipe: Optional[AdapterRecipe] = None) -> Dict[str, Any]:
        """
        Train LoRA adapter on dataset.

        recipe: when provided, PEFT config is applied via apply_recipe().
        Falls back to explicit lora_rank / lora_alpha parameters otherwise.
        """
        
        print(f"\n{'='*70}")
        print("LORA TRAINING")
        print(f"{'='*70}")
        print(f"[TRAINER] Base model: {self.base_model}")
        print(f"[TRAINER] Dataset: {dataset_path}")
        print(f"[TRAINER] Epochs: {num_epochs}")
        print(f"[TRAINER] LoRA rank: {lora_rank}, alpha: {lora_alpha}")
        print(f"[TRAINER] Learning rate: {learning_rate}")
        
        # Load dataset
        dataset = []
        with open(dataset_path, 'r') as f:
            for line in f:
                if line.strip():
                    dataset.append(json.loads(line))

        print(f"[TRAINER] Loaded {len(dataset)} training samples")

        _use_legacy = recipe is None or getattr(recipe, "sft_format", "openai_messages") == "legacy_response_marker"
        for record in dataset:
            if "messages" in record:
                pass  # TRL ≥ 0.24 SFTTrainer handles conversational records natively via apply_chat_template
            elif "prompt" in record and "completion" in record:
                pass  # already in prompt/completion format
            elif _use_legacy:
                # Legacy [SYSTEM]/[USER]/[RESPONSE] wrapping — only when sft_format=legacy_response_marker
                instruction = record.get("instruction", "")
                if instruction.startswith("[SYSTEM]"):
                    record["prompt"] = instruction + "\n\n[RESPONSE]\n"
                else:
                    capability = record.get("capability", "")
                    role = _CAP_TO_ROLE.get(capability, "")
                    sys_prompt = _AGENT_PROMPTS.get(role, "You are an expert AI assistant.")
                    record["prompt"] = (
                        f"[SYSTEM]\n{sys_prompt}\n\n"
                        f"[USER]\n{instruction}\n\n"
                        f"[RESPONSE]\n"
                    )
                if "completion" not in record:
                    record["completion"] = record.get("response", "")
        
        if self.HAS_UNSLOTH:
            return await self._train_unsloth(dataset, num_epochs, learning_rate, lora_rank, lora_alpha, recipe=recipe)
        elif self._check_peft_gpu():
            return await self._train_peft(dataset, num_epochs, learning_rate, lora_rank, lora_alpha, recipe=recipe)
        else:
            return await self._train_mock(dataset, num_epochs)

    async def _train_unsloth(self,
                             dataset: List[Dict],
                             num_epochs: int,
                             learning_rate: float,
                             lora_rank: int,
                             lora_alpha: int,
                             recipe: Optional[AdapterRecipe] = None) -> Dict[str, Any]:
        """Train with actual Unsloth (requires GPU)"""
        try:
            from unsloth import FastLanguageModel
            from trl import SFTTrainer, SFTConfig
            
            print("\n[TRAINER] Loading Unsloth model...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.base_model,
                max_seq_length=self.max_seq_length,
                dtype=None,
                load_in_4bit=True,
            )

            # Ensure pad token is set — required for batched training.
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

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

            # Completion-only masking via TRL v0.24 prompt-completion format:
            # The dataset has "prompt" and "completion" fields.  SFTTrainer
            # tokenizes them separately and creates a completion mask so loss
            # is only computed on the completion tokens.
            # This replaces the removed DataCollatorForCompletionOnlyLM.
            #
            # The "prompt" field ends with "\n\n[RESPONSE]\n" so the model
            # learns to generate immediately after that delimiter — matching
            # the exact prompt format used at inference time.
            #
            # Prompts are already formatted by train() with the correct
            # [SYSTEM]/[USER]/[RESPONSE] structure before this method is called.
            # Only fill in 'completion' for any records that still need it.
            for record in dataset:
                if "messages" not in record and "completion" not in record:
                    record["completion"] = record.get("response", "")

            sft_config = SFTConfig(
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                warmup_steps=100,
                num_train_epochs=num_epochs,
                learning_rate=learning_rate,
                fp16=False,
                bf16=True,
                logging_steps=10,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                output_dir=str(self.output_dir / "checkpoint"),
                max_seq_length=self.max_seq_length,
                # Completion-only loss: only backpropagate on the completion
                # tokens. The prompt (instruction + [RESPONSE]\n) is masked.
                completion_only_loss=True,
                # Never write HF checkpoints during training — we do a single
                # explicit save after training completes.  Without this,
                # SFTTrainer dumps ~200 MB checkpoint dirs every 500 steps
                # (7+ times per capability), consuming ~1.4 GB of wasted disk.
                save_strategy="no",
                # packing=False: do NOT concatenate examples, so each example
                # is trained independently without cross-contamination.
                packing=False,
                # Suppress wandb / tensorboard init — not needed in training
                # containers and causes hangs if wandb credentials are absent.
                report_to="none",
            )
            
            # Convert to HuggingFace Dataset for SFTTrainer compatibility
            try:
                from datasets import Dataset as HFDataset
                train_dataset = HFDataset.from_list(dataset)
            except ImportError:
                train_dataset = dataset

            trainer = SFTTrainer(
                model=model,
                processing_class=tokenizer,
                train_dataset=train_dataset,
                args=sft_config,
            )
            
            print("[TRAINER] Starting Unsloth training...")
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

            return {
                'status':     'success',
                'adapter_id': self.adapter_id,
                'method':     'unsloth',
                'epochs':     num_epochs,
                'samples':    len(dataset),
                'lora_rank':  lora_rank,
                'size_mb':    size_mb,
                'final_loss': round(final_loss, 6),
            }
            
        except Exception as e:
            print(f"[TRAINER] Unsloth training failed: {e}")
            # Re-raise so the subprocess exits non-zero and the Training
            # Service can record the failure accurately.
            raise
    
    async def _train_peft(self,
                          dataset: List[Dict],
                          num_epochs: int,
                          learning_rate: float,
                          lora_rank: int,
                          lora_alpha: int,
                          recipe: Optional[AdapterRecipe] = None) -> Dict[str, Any]:
        """Real GPU training via HuggingFace PEFT + TRL (no Unsloth dependency).

        Used when Unsloth is unavailable (e.g. Windows triton incompatibility).
        Slower than Unsloth (~2x) but produces identical adapter weights.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from trl import SFTTrainer, SFTConfig
        from datasets import Dataset as HFDataset

        print("\n[TRAINER] HF PEFT training (GPU, no Unsloth)...")

        _lora_rank  = recipe.r      if recipe else lora_rank
        _lora_alpha = recipe.alpha  if recipe else lora_alpha
        _dropout    = recipe.dropout if recipe else 0.05
        _modules    = recipe.target_modules if recipe else _ADAPTER_TARGET_MODULES

        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        print(f"[TRAINER] Loading base model: {self.base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=bnb_cfg,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = prepare_model_for_kbit_training(model)
        lora_cfg = LoraConfig(
            r=_lora_rank,
            lora_alpha=_lora_alpha,
            target_modules=_modules,
            lora_dropout=_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            use_rslora=getattr(recipe, "use_rslora", False) if recipe else False,
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

        train_dataset = HFDataset.from_list(dataset)

        sft_config = SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=False,
            bf16=True,
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            output_dir=str(self.output_dir / "checkpoint"),
            max_seq_length=self.max_seq_length,
            completion_only_loss=True,
            save_strategy="no",
            packing=False,
            report_to="none",
        )

        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            args=sft_config,
        )

        print("[TRAINER] Starting HF PEFT training...")
        trainer.train()

        model.save_pretrained(str(self.output_dir))
        tokenizer.save_pretrained(str(self.output_dir))

        weight_file = self.output_dir / "adapter_model.safetensors"
        if not weight_file.exists():
            weight_file = self.output_dir / "adapter_model.bin"
        size_mb = round(weight_file.stat().st_size / 1e6, 2) if weight_file.exists() else 0.0

        final_loss = 0.0
        if hasattr(trainer, "state") and trainer.state.log_history:
            for entry in reversed(trainer.state.log_history):
                if "loss" in entry:
                    final_loss = float(entry["loss"])
                    break

        self.adapter_id = self.output_dir.name
        print(f"[TRAINER] HF PEFT training complete — loss={final_loss:.4f}  size={size_mb:.1f}MB")
        return {
            "status":     "success",
            "adapter_id": self.adapter_id,
            "method":     "hf_peft",
            "epochs":     num_epochs,
            "samples":    len(dataset),
            "lora_rank":  _lora_rank,
            "size_mb":    size_mb,
            "final_loss": round(final_loss, 6),
        }

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


class LoRAAgentTrainerPipeline:
    """Complete 5-stage pipeline for training LoRA on agent outputs"""
    
    def __init__(self,
                 agent_runs_file: Optional[str] = None,
                 base_model: Optional[str] = None):
        self.agent_runs_file = agent_runs_file
        self.base_model = base_model or _resolve_base_model()
        self.dataset_manager = AgentOutputDataset()
        self.trainer_builder = TrainingDatasetBuilder()
        self.lora_trainer = LoRATrainer(base_model=self.base_model)
        
        self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.report = {
            'run_id': self.run_id,
            'stages': {}
        }
    
    async def run(self, 
                  filter_threshold: float = 0.65,
                  num_epochs: int = 3,
                  min_samples: int = 5) -> Dict[str, Any]:
        """Execute full 5-stage pipeline"""
        
        print(f"\n{'='*70}")
        print("LORA AGENT TRAINER PIPELINE")
        print(f"{'='*70}")
        print("Stage 1: Collect agent runs")
        print("Stage 2: Filter successful runs (evaluation >= threshold)")
        print("Stage 3: Build training dataset")
        print("Stage 4: Prepare instruction-response pairs")
        print("Stage 5: Train LoRA adapter")
        print(f"{'='*70}\n")
        
        # Stage 1: Collect runs
        print(f"\n{'='*70}")
        print("STAGE 1: COLLECT AGENT RUNS")
        print(f"{'='*70}")
        
        if self.agent_runs_file:
            print(f"[STAGE 1] Loading from {self.agent_runs_file}...")
            self.dataset_manager.load_runs_from_file(self.agent_runs_file)
        else:
            print("[STAGE 1] No runs file provided, using simulation mode")
            await self._generate_simulation_runs()
        
        print(f"[STAGE 1] Total runs collected: {len(self.dataset_manager.runs)}")
        self.report['stages']['stage_1'] = {
            'total_runs': len(self.dataset_manager.runs),
            'runs_file': str(self.dataset_manager.run_file)
        }
        
        # Stage 2: Filter
        print(f"\n{'='*70}")
        print("STAGE 2: FILTER SUCCESSFUL RUNS")
        print(f"{'='*70}\n")
        
        filtered_runs = self.dataset_manager.filter_by_score(min_score=filter_threshold)
        self.report['stages']['stage_2'] = {
            'threshold': filter_threshold,
            'high_quality_count': len(filtered_runs),
            'filtered_out': len(self.dataset_manager.runs) - len(filtered_runs),
        }
        
        if len(filtered_runs) < min_samples:
            print(f"\n[WARNING] Only {len(filtered_runs)} runs available (need {min_samples})")
            print("Proceeding with available data...")
        
        # Stage 3: Build dataset
        print(f"\n{'='*70}")
        print("STAGE 3: BUILD TRAINING DATASET")
        print(f"{'='*70}\n")
        
        self.trainer_builder.build_from_filtered_runs(filtered_runs)
        stats = self.trainer_builder.get_stats()
        print(f"\n[DATASET] Stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        self.report['stages']['stage_3'] = stats
        
        # Stage 4: Prepare pairs
        print(f"\n{'='*70}")
        print("STAGE 4: SAVE TRAINING DATA")
        print(f"{'='*70}\n")
        
        dataset_path = self.trainer_builder.save_dataset(format="jsonl")
        self.report['stages']['stage_4'] = {
            'dataset_file': str(dataset_path),
            'format': 'jsonl',
            'pairs': len(self.trainer_builder.dataset)
        }
        
        # Stage 5: Train LoRA
        print(f"\n{'='*70}")
        print("STAGE 5: TRAIN LORA ADAPTER")
        print(f"{'='*70}")
        
        train_result = await self.lora_trainer.train(
            dataset_path=dataset_path,
            num_epochs=num_epochs
        )
        
        self.report['stages']['stage_5'] = train_result
        
        # Final report
        self._print_final_report()
        self._save_report()
        
        return self.report
    
    async def _generate_simulation_runs(self) -> None:
        """Generate simulated agent runs for demo"""
        problems = [
            "Refactor nested if statements in authentication handler",
            "Optimize database query performance for user reports",
            "Improve error handling in API endpoints",
            "Consolidate duplicate utility functions",
            "Add proper logging to async tasks",
        ]
        
        print("[STAGE 1] Generating simulated runs for demo...\n")
        
        for i in range(10):
            problem = random.choice(problems)
            score = 0.4 + random.random() * 0.5
            
            run = {
                'problem': problem,
                'context': f"Found in codebase during analysis pass {i+1}",
                'solution': f"Implemented solution with improved structure and error handling",
                'evaluation': round(score, 2),
                'improvement': f"Code quality improved by {int(score*100)}%",
                'lessons_learned': "Focus on maintainability and performance"
            }
            
            self.dataset_manager.add_run(run)
            print(f"[STAGE 1] Run {i+1}: {problem[:50]}... (score: {score:.2f})")
    
    def _print_final_report(self) -> None:
        """Print pipeline completion report"""
        print(f"\n{'='*70}")
        print("PIPELINE COMPLETE")
        print(f"{'='*70}")
        print(f"Run ID: {self.run_id}")
        print(f"\nStages:")
        print(f"  1. Collected: {self.report['stages']['stage_1']['total_runs']} runs")
        print(f"  2. Filtered: {self.report['stages']['stage_2']['high_quality_count']} high-quality")
        print(f"  3. Dataset: {self.report['stages']['stage_3']['total_pairs']} pairs")
        print(f"  4. Saved: {Path(self.report['stages']['stage_4']['dataset_file']).name}")
        print(f"  5. Trained: {self.report['stages']['stage_5']['adapter_id']}")
        print(f"\nAdapter ready for deployment!")
        print(f"{'='*70}\n")
    
    def _save_report(self) -> None:
        """Save final report"""
        report_dir = Path("./data/training_pipeline_reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = report_dir / f"lora_trainer_report_{self.run_id}.json"
        with open(report_file, 'w') as f:
            json.dump(self.report, f, indent=2)
        
        print(f"Report saved: {report_file}\n")


async def main():
    """Example usage"""
    
    # Initialize pipeline
    pipeline = LoRAAgentTrainerPipeline()
    
    # Run complete pipeline
    report = await pipeline.run(
        filter_threshold=0.65,  # Keep runs with score >= 0.65
        num_epochs=3,           # Train for 3 epochs
        min_samples=5           # Need at least 5 samples
    )
    
    # You can also load from existing runs file:
    # pipeline = LoRAAgentTrainerPipeline(
    #     agent_runs_file="./data/agent_runs/my_runs.jsonl"
    # )
    # report = await pipeline.run()


if __name__ == "__main__":
    asyncio.run(main())
