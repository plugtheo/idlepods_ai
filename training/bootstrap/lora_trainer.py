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


def _compute_base_model_hash(base_model_id: str) -> str:
    """
    Resolve the HuggingFace commit hash (snapshot revision) of the base model.

    The snapshot revision uniquely pins the base weights; an adapter only has
    meaning relative to the exact base it was trained against. A mismatch on
    consecutive training rounds means the base drifted under us and the
    adapter's prior learning may no longer compose correctly — this is the
    invariant that protects future merge-to-base.

    Resolution order:
      1. If *base_model_id* is a filesystem path containing
         ".../snapshots/<hash>/", extract <hash> directly (offline-safe).
      2. Otherwise look up the local HF hub cache for the model id and
         return the snapshot directory name.
      3. Fall back to "unknown" when neither path resolves.
    """
    normalized = base_model_id.replace("\\", "/")
    if "/snapshots/" in normalized:
        commit = normalized.split("/snapshots/", 1)[1].split("/", 1)[0]
        if commit and commit not in {"main", "master"}:
            return commit

    try:
        import huggingface_hub.constants as _hfc
        hub_cache = Path(
            getattr(_hfc, "HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub")
        )
        slug = "models--" + base_model_id.replace("/", "--")
        snap_root = hub_cache / slug / "snapshots"
        if snap_root.is_dir():
            for snap in reversed(sorted(snap_root.iterdir())):
                if snap.is_dir() and (snap / "config.json").exists():
                    return snap.name
    except Exception:
        pass

    return "unknown"


def _compute_base_model_tokenizer_hash(base_model_id: str) -> str:
    """
    SHA-256 of the base model's tokenizer.json from the local HF hub cache.
    Falls back to "unknown" when offline or cache is missing so training is
    never blocked — but the hash will always be consistent for the same
    snapshot revision regardless of which host runs the lookup.

    Resolution order:
      1. If *base_model_id* is itself a snapshot directory that exists on
         disk, hash the tokenizer.json inside it.
      2. If *base_model_id* looks like a snapshot path but does not exist on
         this host (e.g., a Linux container path read from metadata on a
         Windows host), parse the slug + commit out of the string and look
         them up in the local HF hub cache.
      3. Treat *base_model_id* as a repo id and let huggingface_hub resolve it.
      4. Fall back to the last snapshot of the slug-derived directory.
    """
    normalized = base_model_id.replace("\\", "/")

    # 1. Direct path read — works inside the training container where the
    #    snapshot directory the trainer wrote into metadata is also live.
    if "/snapshots/" in normalized:
        direct_tok = Path(base_model_id) / "tokenizer.json"
        if direct_tok.exists():
            return hashlib.sha256(direct_tok.read_bytes()).hexdigest()

        # 2. Cross-host re-resolution: extract <slug>/snapshots/<commit> and
        #    look it up in this host's HF cache. Lets a Windows host backfill
        #    hashes from metadata that was written by the Linux container.
        try:
            parts = normalized.split("/snapshots/", 1)
            slug_seg = parts[0].rstrip("/").split("/")[-1]
            commit = parts[1].split("/", 1)[0]
            if slug_seg.startswith("models--"):
                import huggingface_hub.constants as _hfc
                hub_cache = Path(
                    getattr(_hfc, "HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub")
                )
                tok_path = hub_cache / slug_seg / "snapshots" / commit / "tokenizer.json"
                if tok_path.exists():
                    return hashlib.sha256(tok_path.read_bytes()).hexdigest()
        except Exception:
            pass

    # 3. Standard repo-id resolution via huggingface_hub.
    try:
        from huggingface_hub import hf_hub_download
        tok_path = hf_hub_download(base_model_id, "tokenizer.json", local_files_only=True)
        return hashlib.sha256(Path(tok_path).read_bytes()).hexdigest()
    except Exception:
        pass

    # 4. Last-snapshot fallback for repo-id form (Qwen/Qwen3-8B → models--Qwen--Qwen3-8B).
    try:
        import huggingface_hub.constants as _hfc
        hub_cache = Path(
            getattr(_hfc, "HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub")
        )
        slug = "models--" + base_model_id.replace("/", "--")
        snap_root = hub_cache / slug / "snapshots"
        if snap_root.is_dir():
            for snap in reversed(sorted(snap_root.iterdir())):
                tok_path = snap / "tokenizer.json"
                if tok_path.exists():
                    return hashlib.sha256(tok_path.read_bytes()).hexdigest()
    except Exception:
        pass
    return "unknown"


def _zero_pad_adapter_to_rank(
    src_dir: Path,
    dst_dir: Path,
    new_r: int,
    new_alpha: int,
) -> int:
    """
    Expand a saved LoRA adapter to a higher rank by zero-padding every B
    (column-wise) and A (row-wise) matrix in the state dict. Lets a fresh
    higher-rank PEFT model warm-start from a lower-rank checkpoint without
    discarding any prior learning.

    Mathematically:

        B_new = [B_old | 0]        shape (d, new_r)
        A_new = [A_old; 0]         shape (new_r, d)
        B_new @ A_new              =  B_old @ A_old + 0  ==  B_old @ A_old

    So the expanded adapter produces the exact same ΔW at step 0 as the
    source. Training then refines the original rank slots AND fills the
    zero-initialized new slots from the data — the "grow without restart"
    property of LoRA. DoRA's lora_magnitude_vector is rank-independent and
    is copied through untouched.

    Returns the detected old rank for logging.
    """
    from safetensors.torch import load_file, save_file
    import torch

    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Mirror auxiliary files (tokenizer, special tokens, README, etc.) but
    # skip the weights and adapter_config.json — those we rewrite.
    for f in src_dir.iterdir():
        if not f.is_file():
            continue
        if f.name in ("adapter_model.safetensors", "adapter_model.bin", "adapter_config.json"):
            continue
        shutil.copy2(f, dst_dir / f.name)

    src_weights = src_dir / "adapter_model.safetensors"
    if not src_weights.exists():
        raise FileNotFoundError(f"No adapter_model.safetensors at {src_dir}")
    state_dict = load_file(str(src_weights))

    detected_old_r = 0
    out: Dict[str, "torch.Tensor"] = {}
    for key, tensor in state_dict.items():
        if key.endswith("lora_A.weight"):
            r_old, d = tensor.shape
            detected_old_r = max(detected_old_r, r_old)
            if r_old < new_r:
                pad = torch.zeros((new_r - r_old, d), dtype=tensor.dtype, device=tensor.device)
                tensor = torch.cat([tensor, pad], dim=0)
        elif key.endswith("lora_B.weight"):
            d, r_old = tensor.shape
            detected_old_r = max(detected_old_r, r_old)
            if r_old < new_r:
                pad = torch.zeros((d, new_r - r_old), dtype=tensor.dtype, device=tensor.device)
                tensor = torch.cat([tensor, pad], dim=1)
        out[key] = tensor

    save_file(out, str(dst_dir / "adapter_model.safetensors"))

    # adapter_config.json must reflect the new rank/alpha or PEFT will
    # instantiate a model of the wrong shape.
    src_cfg = src_dir / "adapter_config.json"
    if src_cfg.exists():
        try:
            cfg = json.loads(src_cfg.read_text(encoding="utf-8"))
            cfg["r"] = new_r
            cfg["lora_alpha"] = new_alpha
            (dst_dir / "adapter_config.json").write_text(
                json.dumps(cfg, indent=2), encoding="utf-8"
            )
        except (OSError, json.JSONDecodeError):
            shutil.copy2(src_cfg, dst_dir / "adapter_config.json")

    return detected_old_r


def _write_metrics_summary(metrics_path: Path, output_path: Path) -> dict:
    """
    Read train_metrics.jsonl and write a concise loss-curve summary JSON.
    Returns the summary dict (empty dict if no data).
    """
    if not metrics_path.exists():
        return {}
    entries = []
    with open(metrics_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    if not entries:
        return {}

    losses = [e["loss"] for e in entries if "loss" in e]
    epochs_seq = [e.get("epoch", 0) for e in entries if "loss" in e]

    tail = losses[max(0, len(losses) * 4 // 5):]
    slope = 0.0
    if len(tail) >= 2:
        slope = round((tail[-1] - tail[0]) / max(len(tail) - 1, 1), 6)

    summary = {
        "total_steps": len(entries),
        "initial_loss": round(losses[0], 6) if losses else None,
        "final_loss": round(losses[-1], 6) if losses else None,
        "min_loss": round(min(losses), 6) if losses else None,
        "loss_slope_tail": slope,
        "epochs_completed": round(max(epochs_seq), 2) if epochs_seq else None,
    }
    try:
        output_path.write_text(json.dumps(summary, indent=2))
    except OSError:
        pass
    return summary


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
    base_model_hash: str = "unknown",
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

    # ── Base-model drift detection ─────────────────────────────────────────
    # Compare the new base_model_hash against the previous version's hash.
    # A mismatch (both known and different) means the base weights moved
    # under us between training rounds — adapter compositions from previous
    # versions may no longer apply cleanly, and merge-to-base later will
    # need to pick a single base anchor. Record the drift in the new history
    # entry so the merge tool can audit it; don't hard-block (operator may
    # have intentionally upgraded the base, e.g., Qwen3 → Qwen3.5).
    prev_base_hash: Optional[str] = None
    for prev in reversed(old_history):
        candidate = prev.get("base_model_hash") if isinstance(prev, dict) else None
        if candidate and candidate != "unknown":
            prev_base_hash = candidate
            break
    base_drift_warning: Optional[Dict[str, str]] = None
    if (
        prev_base_hash
        and base_model_hash
        and base_model_hash != "unknown"
        and prev_base_hash != base_model_hash
    ):
        base_drift_warning = {
            "previous_base_model_hash": prev_base_hash,
            "new_base_model_hash": base_model_hash,
            "message": (
                "Base model snapshot differs from the previous adapter version. "
                "Adapter weights are only meaningful relative to the base they "
                "were trained against; merge-to-base from this generation forward "
                "must use the new base."
            ),
        }
        print(
            f"  [WARN] base_model drift detected for {capability}_lora: "
            f"prev={prev_base_hash[:12]}… new={base_model_hash[:12]}…",
            flush=True,
        )

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
        "base_model_hash":        base_model_hash,
        "trainer_version":        _tv,
        "experience_record_count": experience_record_count,
        "synthetic_record_count":  synthetic_record_count,
        "experience_data_pct":     _exp_pct,
    }
    if base_drift_warning:
        new_history_entry["base_model_drift"] = base_drift_warning

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
        "base_model_hash":        base_model_hash,
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
    regression_results: Optional[Dict[str, Any]] = None,
    eval_loss: float = 0.0,
    perplexity: Optional[float] = None,
    assistant_masking_enabled: Optional[bool] = None,
    last_experience_timestamp: Optional[str] = None,
) -> None:
    """
    Under manifest file-lock: flip active/previous pointers and write v2 HistoryEntry.
    Also updates metadata.json status from 'staging' to 'active'.

    *last_experience_timestamp* is the ISO-8601 timestamp of the newest
    qualifying experience that contributed to this training run. Stored on
    the role's metadata.json as the per-role retrain cursor so the next
    scheduler tick can gate this role on accumulated new data alone.
    Only advanced on successful promotion (failed runs leave the cursor
    intact so retries are eligible immediately).
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
            if last_experience_timestamp:
                meta["last_trained_experience_timestamp"] = last_experience_timestamp
            for entry in meta.get("history", []):
                if entry.get("version") == new_version:
                    entry["status"] = "active"
                    if smoke_results:
                        entry["smoke"] = smoke_results
                    if eval_results:
                        entry["eval"] = eval_results
                    if last_experience_timestamp:
                        entry["last_trained_experience_timestamp"] = last_experience_timestamp
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
        base_model_hash=new_meta.get("base_model_hash", "unknown"),
        trainer_version=new_meta.get("trainer_version", _compute_trainer_version()),
        n_samples=last_history.get("n_samples", 0),
        final_loss=last_history.get("final_loss", 0.0),
        size_mb=last_history.get("size_mb", 0.0),
        eval_metrics={k: float(v) for k, v in (eval_results or {}).items() if isinstance(v, (int, float))},
        smoke=smoke_results or {},
        eval_loss=eval_loss if eval_loss else None,
        perplexity=perplexity,
        assistant_masking_enabled=assistant_masking_enabled,
        won_against_previous=regression_results.get("won") if regression_results else None,
        regression_delta=regression_results.get("delta") if regression_results else None,
        comparison_prompts_n=regression_results.get("prompts_n", 0) if regression_results else 0,
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

        if previous_backup_path and Path(previous_backup_path).exists():
            prev_path = str(previous_backup_path)
        elif old_active_version:
            prev_path = old_active_path
        else:
            prev_path = ""
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
                   resume_from_checkpoint: Optional[str] = None,
                   eval_dataset: Optional[List[Dict]] = None) -> Dict[str, Any]:
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
            return await self._train_unsloth(dataset, num_epochs, learning_rate, lora_rank, lora_alpha, recipe=recipe, resume_from_checkpoint=resume_from_checkpoint, eval_dataset=eval_dataset)
        else:
            return await self._train_mock(dataset, num_epochs)

    async def _train_unsloth(self,
                             dataset: List[Dict],
                             num_epochs: int,
                             learning_rate: float,
                             lora_rank: int,
                             lora_alpha: int,
                             recipe: Optional[AdapterRecipe] = None,
                             resume_from_checkpoint: Optional[str] = None,
                             eval_dataset: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Train with actual Unsloth (requires GPU)"""
        import time as _time
        _t_start = _time.monotonic()
        try:
            from unsloth import FastLanguageModel
            from trl import SFTTrainer, SFTConfig

            # TODO: Consolidate and move to helpers.py
            if recipe is None or not hasattr(recipe, "max_seq_length") or recipe.max_seq_length is None:
                raise RuntimeError("LoRATrainer.train() requires a recipe with max_seq_length override")

            # packing is reported below at the point it's resolved against the
            # dataset shape — keep this header focused on recipe-time params.
            print(
                f"\n[TRAINER] batch_size={recipe.per_device_train_batch_size} "
                f"grad_accum={recipe.gradient_accumulation_steps} "
                f"effective_batch={recipe.per_device_train_batch_size * recipe.gradient_accumulation_steps} "
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

            # 9b: Tokenizer / model vocab-size parity check — catches wrong tokenizer
            # revision or missing special tokens before any training begins.
            _model_vocab = getattr(getattr(model, "config", None), "vocab_size", None)
            if _model_vocab and tokenizer.vocab_size != _model_vocab:
                print(
                    f"[TRAINER] WARN: tokenizer.vocab_size={tokenizer.vocab_size} != "
                    f"model.vocab_size={_model_vocab} — possible tokenizer mismatch",
                    flush=True,
                )

            # 9e: OOM headroom — verify GPU memory is < 85% after model load so a
            # full training run at max_seq_length won't OOM mid-epoch.
            try:
                import torch as _torch
                _used_frac = (
                    _torch.cuda.memory_allocated()
                    / _torch.cuda.get_device_properties(0).total_memory
                )
                print(f"[TRAINER] GPU memory after model load: {_used_frac:.1%} used", flush=True)
                if _used_frac > 0.85:
                    print(
                        f"[TRAINER] WARN: GPU at {_used_frac:.1%} — high OOM risk at "
                        f"max_seq_length={recipe.max_seq_length}. Consider reducing batch size.",
                        flush=True,
                    )
            except Exception as _oom_e:
                print(f"[TRAINER] OOM headroom check skipped: {_oom_e}", flush=True)

            # Apply model-family patched chat template (adds {%- generation %} markers
            # for proper assistant-only loss masking).  Falls back gracefully when no
            # patch exists for the current model so future model changes don't break
            # training — they just lose the masking.
            _patched = LoRATrainer._load_patched_template(self.base_model)
            if _patched:
                tokenizer.chat_template = _patched
            _use_assistant_masking = LoRATrainer._log_masking_support(tokenizer)

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

            # Unsloth (2026.x) requires a `text` column or `formatting_func`
            # even when TRL 0.24 would auto-render the conversational format.
            # Pre-render each record's `messages` with the (already patched)
            # chat template into a flat `text` column.  Assistant-only masking
            # is then re-introduced AFTER trainer construction via
            # unsloth.chat_templates.train_on_responses_only, which operates
            # on tokenized text by matching the model-family role markers.
            _is_conversational = bool(dataset and "messages" in dataset[0])
            if _is_conversational:
                _rendered: List[Dict[str, Any]] = []
                for _r in dataset:
                    _msgs = _r.get("messages") or []
                    if not _msgs:
                        continue
                    _rendered.append({
                        "text": tokenizer.apply_chat_template(_msgs, tokenize=False),
                    })
                dataset = _rendered
                print(
                    f"[TRAINER] Pre-rendered {len(dataset)} conversational records "
                    "into text column via apply_chat_template",
                    flush=True,
                )

            # Packing must be off when masking is on, because
            # train_on_responses_only matches role markers per example — packed
            # sequences would cross example boundaries and break the mask.
            _effective_packing = bool(recipe.packing) and not _use_assistant_masking
            if recipe.packing and not _effective_packing:
                print(
                    f"[TRAINER] packing: False (recipe={recipe.packing}, overridden — "
                    "train_on_responses_only requires per-example boundaries)",
                    flush=True,
                )
            else:
                print(f"[TRAINER] packing: {_effective_packing}", flush=True)

            sft_config_kwargs = dict(
                per_device_train_batch_size=recipe.per_device_train_batch_size,
                gradient_accumulation_steps=recipe.gradient_accumulation_steps,
                # 10% warmup over the run gives the optimizer enough steps to
                # find a stable trajectory before hitting peak LR. 5% was too
                # aggressive — instability hit right as warmup peaked on the
                # 15k-sample coder run.
                warmup_ratio=0.10,
                num_train_epochs=num_epochs,
                learning_rate=learning_rate,
                # Clip per-step gradient norm. Without this, a single bad batch
                # produced grad_norm > 200 mid-run on the coder bootstrap and
                # applied a destructive weight update. 1.0 is the value used by
                # Llama/Mistral/Qwen reference recipes and is safe across all
                # capabilities.
                max_grad_norm=1.0,
                fp16=False,
                bf16=True,
                logging_steps=10,
                optim="adamw_8bit",
                weight_decay=0.01,
                # Cosine decay is smoother than linear for long runs (7k+ steps),
                # avoids the late-training "LR cliff", and matches Llama/Mistral/
                # Qwen reference recipes.
                lr_scheduler_type="cosine",
                output_dir=str(self.output_dir / "checkpoint"),
                max_seq_length=recipe.max_seq_length,
                # Never write HF checkpoints during training — we do a single
                # explicit save after training completes.  Without this,
                # SFTTrainer dumps ~200 MB checkpoint dirs every 500 steps
                # (7+ times per capability), consuming ~1.4 GB of wasted disk.
                save_strategy="no",
                packing=_effective_packing,
                # Suppress wandb / tensorboard init — not needed in training
                # containers and causes hangs if wandb credentials are absent.
                report_to="none",
                dataset_num_proc=4,
                dataloader_num_workers=2,
            )
            if _is_conversational:
                sft_config_kwargs["dataset_text_field"] = "text"
            # assistant_only_loss is computed post-tokenization by
            # train_on_responses_only (applied to the trainer below) when the
            # model family has known role markers — leaving it off here.
            sft_config = SFTConfig(**sft_config_kwargs)

            from datasets import Dataset as HFDataset
            # Pre-render eval_dataset the same way train data was rendered so
            # the column schema matches what dataset_text_field expects.
            _eval_for_hf = eval_dataset
            if _is_conversational and eval_dataset:
                _eval_for_hf = [
                    {"text": tokenizer.apply_chat_template(_r.get("messages") or [], tokenize=False)}
                    for _r in eval_dataset
                    if _r.get("messages")
                ]
            train_dataset = HFDataset.from_list(dataset)
            eval_hf_dataset = HFDataset.from_list(_eval_for_hf) if _eval_for_hf else None

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
                eval_dataset=eval_hf_dataset,
                args=sft_config,
                callbacks=_callbacks if _callbacks else None,
            )

            # Re-introduce assistant-only loss by zeroing the labels for
            # everything outside the assistant response span.  This is the
            # canonical Unsloth pattern for pre-rendered text + chat templates
            # and is equivalent in effect to TRL's assistant_only_loss=True on
            # a conversational dataset, but compatible with Unsloth's stricter
            # SFTTrainer wrapper.
            _markers = LoRATrainer._response_markers(self.base_model)
            if _use_assistant_masking and _markers:
                try:
                    from unsloth.chat_templates import train_on_responses_only
                    trainer = train_on_responses_only(
                        trainer,
                        instruction_part=_markers[0],
                        response_part=_markers[1],
                    )
                    print(
                        f"[TRAINER] assistant-only masking applied via "
                        f"train_on_responses_only (instruction={_markers[0]!r}, "
                        f"response={_markers[1]!r})",
                        flush=True,
                    )
                except Exception as _wrap_exc:
                    print(
                        f"[TRAINER] WARN: train_on_responses_only unavailable "
                        f"({_wrap_exc}) — training on full sequence loss",
                        flush=True,
                    )
                    _use_assistant_masking = False
            elif _use_assistant_masking:
                print(
                    "[TRAINER] WARN: assistant masking template detected but no "
                    "response markers registered for this model family — "
                    "training on full sequence loss",
                    flush=True,
                )
                _use_assistant_masking = False

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

            # One-shot eval on held-out split (no GPU overhead during training).
            eval_loss = 0.0
            if eval_hf_dataset is not None:
                try:
                    eval_out = trainer.evaluate()
                    eval_loss = float(eval_out.get("eval_loss", 0.0))
                    print(f"[TRAINER] Eval loss: {eval_loss:.6f}", flush=True)
                except Exception as _eval_exc:
                    print(f"[TRAINER] Eval skipped: {_eval_exc}", flush=True)

            # Distil per-step metrics into a compact summary for the diff report.
            _metrics_summary = _write_metrics_summary(
                _metrics_path,
                self.output_dir / "train_metrics_summary.json",
            )

            duration_s = round(_time.monotonic() - _t_start, 1)
            print(
                f"training_complete method=unsloth role={self.output_dir.name} "
                f"epochs={num_epochs} samples={len(dataset)} "
                f"final_loss={final_loss:.6f} duration_s={duration_s}",
                flush=True,
            )

            import math as _math
            return {
                'status':          'success',
                'adapter_id':      self.adapter_id,
                'method':          'unsloth',
                'epochs':          num_epochs,
                'samples':         len(dataset),
                'lora_rank':       lora_rank,
                'size_mb':         size_mb,
                'final_loss':      round(final_loss, 6),
                'eval_loss':       round(eval_loss, 6) if eval_loss else None,
                'perplexity':      round(_math.exp(eval_loss), 4) if eval_loss else None,
                'metrics_summary': _metrics_summary,
                'duration_s':      duration_s,
                'assistant_masking_enabled': _use_assistant_masking,
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

    @staticmethod
    def _load_patched_template(model_id: str) -> "Optional[str]":
        """
        Load a chat template patched with {%- generation %}/{%- endgeneration %}
        markers for the given model.  Falls back to None so callers can disable
        assistant_only_loss gracefully when no patch exists.

        Template files live at training/bootstrap/templates/{ModelFamily}.jinja.
        Match is by substring: "Qwen3-8B" matches "Qwen3.jinja".  Adding support
        for a new model family requires only dropping a new .jinja file — no code
        change.
        """
        templates_dir = Path(__file__).parent / "templates"
        if not templates_dir.is_dir():
            return None
        # Search the full model_id string (not just the last component) so that
        # local HF cache paths like /cache/models--Qwen--Qwen3-8B/snapshots/abc123
        # still match.  "qwen3" appears in the path slug even when the last
        # component is a commit hash.
        model_id_lower = model_id.lower()
        for tmpl in templates_dir.glob("*.jinja"):
            if tmpl.stem.lower() in model_id_lower:
                return tmpl.read_text(encoding="utf-8")
        return None

    # Role markers used by train_on_responses_only to locate the assistant
    # response span in a pre-rendered chat. Substring-matched against the
    # base model id, mirroring _load_patched_template. Extend by adding a
    # new entry when supporting a new model family.
    _RESPONSE_MARKERS: Dict[str, tuple] = {
        "qwen3":  ("<|im_start|>user\n", "<|im_start|>assistant\n"),
        "qwen2":  ("<|im_start|>user\n", "<|im_start|>assistant\n"),
        "qwen":   ("<|im_start|>user\n", "<|im_start|>assistant\n"),
        "llama":  ("<|start_header_id|>user<|end_header_id|>\n\n",
                   "<|start_header_id|>assistant<|end_header_id|>\n\n"),
    }

    @staticmethod
    def _response_markers(model_id: str) -> "Optional[tuple]":
        """Return (instruction_part, response_part) for *model_id* or None.

        Mirrors _load_patched_template's substring match so adding a model
        family requires updating only the _RESPONSE_MARKERS table.
        """
        mid = model_id.lower()
        for key, markers in LoRATrainer._RESPONSE_MARKERS.items():
            if key in mid:
                return markers
        return None

    @staticmethod
    def _log_masking_support(tokenizer) -> bool:
        """
        Check whether the active chat template has {%- generation %} markers and
        log the result.  Returns True when assistant-only loss masking is safe to
        enable.

        NOTE: TRL 0.24 raises RuntimeError when assistant_only_loss=True is used
        with a template that lacks these markers — there is NO token-boundary
        fallback.  This method MUST be called after any template patch is applied.
        """
        template_source = getattr(tokenizer, "chat_template", "") or ""
        has_marker = "{% generation %}" in template_source or "{%- generation %}" in template_source
        if has_marker:
            print("[TRAINER] chat template has {%- generation %} markers — assistant_only_loss enabled.", flush=True)
        else:
            print(
                "[TRAINER] WARN: chat template lacks {%- generation %} markers — "
                "assistant_only_loss disabled (full-sequence loss). "
                "Add a patched template to training/bootstrap/templates/ to enable masking.",
                flush=True,
            )
        return has_marker

