"""
Trainer entry point (subprocess adapter)
==========================================
This module is invoked as a subprocess by trainer_launcher.py.
It reads ExperienceEvent JSONL produced by the Experience Service and
converts it into SFT (Supervised Fine-Tuning) training pairs for LoRATrainer.

Usage:
    python -m services.training.app.trainer_entry \\
        --data-path /tmp/data.jsonl \\
        --base-model <model_id from models.yaml> \\
        --output-dir /data/lora_checkpoints \\
        --role coder

JSONL schema (ExperienceEvent, written by orchestration/app/routes/run.py):
    {
        "session_id": str,
        "prompt": str,                       # original user prompt
        "final_output": str,                 # best pipeline output
        "final_score": float,                # quality score 0-1
        "converged": bool,
        "agent_chain": [str],
        "iteration_scores": [float],
        "intent": str,
        "complexity": str,
        "contributions": [
            {
                "role": str,
                "output": str,               # compact extracted version
                "full_output": str,          # complete raw LLM response ← used for SFT
                "quality_score": float,
                "iteration": int,
                "messages": [               # full prompt sent to LLM ← used for SFT
                    {"role": "system"|"user"|"assistant", "content": str}
                ]
            }
        ]
    }

Each contribution that has both `messages` and `full_output` becomes one SFT pair.
Falls back to (prompt → final_output) for older records without per-contribution messages.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

# Add project root to sys.path so we can import from training/
# In Docker: file lives at /app/services/training/app/trainer_entry.py
#   parents[0] = /app/services/training/app
#   parents[1] = /app/services/training
#   parents[2] = /app/services
#   parents[3] = /app  ← project root (PYTHONPATH=/app already covers this)
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shared.contracts.experience import AgentContribution
from shared.contracts.sft_builder import build_sft_pair

from training.bootstrap.lora_trainer import (
    LoRATrainer,
    _pre_train_backup,
    _post_train_version,
    _post_train_stage,
    _promote_to_active,
    _mark_failed,
)
from training.bootstrap.smoke_gate import run_smoke
from shared.contracts.training import AdapterRecipe, lookup_recipe

# System prompts — imported from the single source of truth so that training and
# inference always use byte-identical strings.  Do NOT redefine these here.
from shared.contracts.agent_prompts import AGENT_PROMPTS as _CAPABILITY_SYSTEM_PROMPTS, BOOTSTRAP_CAP_TO_ROLE, ROLE_TO_BOOTSTRAP_CAP
# Add the service root (training/) to sys.path so that `app.config.settings`
# resolves to training/app/config/settings.py in both environments:
#   Docker:     /app/services/training/  (from COPY training/app/ /app/services/training/app/)
#   Bare-metal: <project_root>/training/
_SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SERVICE_ROOT))

from app.config.settings import settings as _training_settings

# Maps capability label → curated training dataset filename
_CAPABILITY_TO_CURATED: dict[str, str] = {
    "coder":      "coding_dataset.jsonl",
    "debugger":   "debugging_dataset.jsonl",
    "reviewer":   "review_dataset.jsonl",
    "planner":    "planning_dataset.jsonl",
    "researcher": "research_dataset.jsonl",
    "critic":     "criticism_dataset.jsonl",
}

# Maps capability label (from Experience Service) to the adapter directory name
# that vLLM expects at /lora_checkpoints/<adapter_name>.
# Must stay in sync with ROLE_ADAPTER in orchestration/app/agents/prompts.py.
_CAPABILITY_TO_ADAPTER: dict[str, str] = {
    "coder":      "coding_lora",
    "debugger":   "debugging_lora",
    "reviewer":   "review_lora",
    "planner":    "planning_lora",
    "researcher": "research_lora",
    "critic":     "criticism_lora",
}

# Safety floors — keep as module constants, do NOT promote to env.
# These guard against catastrophic LoRA weight corruption; making them easily
# configurable invites disabling the guard without understanding the risk.
MIN_QUALITY_SCORE  = 0.65   # minimum score; records below this are excluded from SFT pairs
MIN_SFT_PAIRS      = 10     # refuse training on fewer pairs; near-zero data destroys adapter weights
MAX_TRAINING_SAMPLES = 10_000  # cap to keep training runtime bounded


def _is_clean_output(text: str) -> bool:
    """
    Return True if *text* looks like a genuine LLM response.

    Rejects strings that are pure JSON objects/arrays (orchestration metadata
    accidentally written to contributions[].full_output in earlier pipeline
    versions).
    """
    stripped = text.strip()
    if not stripped:
        return False
    # Reject bare JSON objects/arrays — they are metadata, not code/prose
    if stripped[0] in ("{", "[") and stripped[-1] in ("}", "]"):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, (dict, list)):
                return False
        except json.JSONDecodeError:
            pass  # Not valid JSON — content is fine
    # Reject strings that contain known orchestration metadata keys
    contamination_markers = (
        "session_id", "pipeline_metadata", "agent_chain",
        "iteration_scores", "ORCHESTRATION_",
    )
    lower = stripped.lower()
    if any(marker.lower() in lower for marker in contamination_markers):
        return False
    return True


def _load_sft_pairs(
    data_path: str,
    capability: str,
    recipe: "AdapterRecipe",
    min_score: float = MIN_QUALITY_SCORE,
) -> List[Dict]:
    """
    Read ExperienceEvent JSONL and produce SFT pairs shaped by recipe.sft_format.

    For openai_messages: {"messages": [...], "score": float}
    """
    pairs: List[Dict] = []
    skipped_score = 0
    skipped_no_data = 0

    with open(data_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[trainer-entry] Skipping malformed JSONL line: {exc}", flush=True)
                continue

            score = float(record.get("final_score", 0.0))
            if score < min_score:
                skipped_score += 1
                continue

            added = 0
            for contribution in record.get("contributions", []):
                if contribution.get("role", "") != capability:
                    continue

                messages: Optional[List[Dict]] = contribution.get("messages")
                full_output: Optional[str] = contribution.get("full_output") or contribution.get("output")

                if messages and full_output:
                    if not _is_clean_output(full_output):
                        print(
                            f"[trainer-entry] Skipping contaminated full_output for role={contribution.get('role')}",
                            flush=True,
                        )
                        continue
                    # role-agnostic: any role with tool_turns produces tool-call SFT pairs
                    tool_turns = contribution.get("tool_turns") or []
                    contrib = AgentContribution(
                        role=capability,
                        output=full_output,
                        quality_score=score,
                        iteration=contribution.get("iteration", 1),
                        messages=list(messages) + tool_turns,
                        tool_calls=contribution.get("tool_calls") or None,
                        tool_results=contribution.get("tool_results") or None,
                    )
                    sft_pair = build_sft_pair(contrib, recipe, capability)
                    # build_sft_pair may return a single dict or a list of dicts (tool-call + full)
                    if isinstance(sft_pair, list):
                        for rec in sft_pair:
                            pairs.append({**rec, "score": score})
                    else:
                        pairs.append({**sft_pair, "score": score})

                    added += 1

            if added == 0:
                skipped_no_data += 1
                if record.get("contributions"):
                    # Optional debug log for auditing; keep it lightweight.
                    pass

    if skipped_no_data > 0:
        print(
            f"[trainer-entry][debug] {skipped_no_data} records had contributions but none for role={capability}",
            flush=True,
        )

    print(
        f"seed_data_status role={capability} experience_count={len(pairs)} data_path={data_path}",
        flush=True,
    )
    print(
        f"[trainer-entry] Loaded {len(pairs)} SFT pairs "
        f"(skipped {skipped_score} below score threshold, "
        f"{skipped_no_data} with no usable data)",
        flush=True,
    )
    return pairs


def _load_curated_pairs(capability: str, data_root: Path, max_samples: int, recipe: "AdapterRecipe") -> List[Dict]:
    """
    Load pre-curated (instruction, response) pairs from the training_data_curated
    directory.  These are high-quality public-dataset samples that bootstrap
    each adapter before experience-driven fine-tuning takes over.

    Returns up to *max_samples* records, sampled randomly if the dataset is larger.
    """
    filename = _CAPABILITY_TO_CURATED.get(capability)
    if not filename:
        return []
    curated_path = data_root / "training_data_curated" / filename
    if not curated_path.exists():
        print(
            f"seed_data_status role={capability} curated_count=0 path={curated_path} exists=False",
            flush=True,
        )
        return []

    pairs: List[Dict[str, str]] = []
    with open(curated_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            instruction = rec.get("instruction", "")
            response = rec.get("response", "")
            if instruction and response:
                sys_prompt = _CAPABILITY_SYSTEM_PROMPTS.get(capability, "You are a helpful AI assistant.")
                contrib = AgentContribution(
                    role=capability,
                    output=response,
                    quality_score=1.0,
                    iteration=1,
                )
                sft_pair = build_sft_pair(contrib, recipe, capability,
                                         system_prompt=sys_prompt, user_prompt=instruction)
                pairs.append({**sft_pair, "score": 1.0})

    if len(pairs) > max_samples:
        import random
        pairs = random.sample(pairs, max_samples)

    print(
        f"seed_data_status role={capability} curated_count={len(pairs)} path={curated_path} exists=True",
        flush=True,
    )
    return pairs


def _load_synthetic_pairs(capability: str, recipe: "AdapterRecipe", max_samples: int) -> List[Dict]:
    """
    Load synthetic SFT pairs from `recipe.synthetic_dataset_path` if configured.

    The future synthesis pipeline writes to that path; until it lands the path
    is unset / file is absent, and this returns []. Same JSONL schema as
    curated: {"instruction": ..., "response": ...}.
    """
    raw_path = getattr(recipe, "synthetic_dataset_path", None)
    if not raw_path:
        print(
            f"seed_data_status role={capability} synthetic_count=0 path=<unset>",
            flush=True,
        )
        return []
    path = Path(raw_path)
    if not path.exists():
        print(
            f"seed_data_status role={capability} synthetic_count=0 path={path} exists=False",
            flush=True,
        )
        return []

    pairs: List[Dict] = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            instruction = rec.get("instruction", "")
            response = rec.get("response", "")
            if instruction and response:
                sys_prompt = _CAPABILITY_SYSTEM_PROMPTS.get(capability, "You are a helpful AI assistant.")
                contrib = AgentContribution(
                    role=capability,
                    output=response,
                    quality_score=1.0,
                    iteration=1,
                )
                sft_pair = build_sft_pair(contrib, recipe, capability,
                                         system_prompt=sys_prompt, user_prompt=instruction)
                pairs.append({**sft_pair, "score": 1.0})

    if len(pairs) > max_samples:
        import random
        pairs = random.sample(pairs, max_samples)

    print(
        f"seed_data_status role={capability} synthetic_count={len(pairs)} path={path} exists=True",
        flush=True,
    )
    return pairs


def _save_sft_pairs(pairs: List[Dict[str, str]], output_dir: Path) -> Path:
    """Write SFT pairs to a JSONL file that LoRATrainer.train() will read."""
    from datetime import datetime
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"sft_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"[trainer-entry] SFT dataset saved: {out_path}  ({len(pairs)} pairs)", flush=True)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LoRA training from experience JSONL")
    parser.add_argument("--data-path",   required=True,  help="Path to ExperienceEvent JSONL")
    parser.add_argument("--base-model",  required=True,  help="HuggingFace model ID")
    parser.add_argument("--output-dir",  required=True,  help="Base adapter output directory (e.g. /data/lora_checkpoints)")
    parser.add_argument("--capability",  default="general", help="Capability label (e.g. coder, researcher)")
    parser.add_argument("--recipe-name", default=None, help="Override recipe registry lookup (role name, e.g. coder)")
    args = parser.parse_args()

    # ---------------------------------------------------------------------------
    # Normalize capability to both forms.
    #
    # _infer_capability() (experience/app/routes/record.py) sends bootstrap-style
    # labels:  "coding", "debugging", "review", "planning", "research", "criticism"
    # But internal lookups (experience contribution role filter, curated data, system
    # prompt dict) all use role names: "coder", "debugger", "reviewer", etc.
    # Backup/versioning functions use bootstrap labels for adapter directory naming.
    #
    # We resolve once here so every downstream call uses the right form.
    # ---------------------------------------------------------------------------
    if args.capability in BOOTSTRAP_CAP_TO_ROLE:
        # Already a bootstrap label (e.g. "coding") — normal path from trigger
        cap_label = args.capability
        role_name = BOOTSTRAP_CAP_TO_ROLE[cap_label]
    elif args.capability in ROLE_TO_BOOTSTRAP_CAP:
        # Already a role name (e.g. "coder") — CLI / test path
        role_name = args.capability
        cap_label = ROLE_TO_BOOTSTRAP_CAP[role_name]
    else:
        # Unknown capability (e.g. "general") — fall through with identity
        cap_label = args.capability
        role_name = args.capability

    print(
        f"[trainer-entry] capability={args.capability!r}  "
        f"→ role_name={role_name!r}  cap_label={cap_label!r}",
        flush=True,
    )

    # Resolve AdapterRecipe — registry lookup unless --recipe-name overrides it.
    try:
        from shared.contracts.models import load_registry
        backend = load_registry().default_backend
    except Exception:
        backend = "primary"
    _recipe_role = args.recipe_name if args.recipe_name else role_name
    recipe: AdapterRecipe = lookup_recipe(backend, _recipe_role)

    # ── PEFT-type readiness gate ─────────────────────────────────────────────
    _VALID_PEFT_TYPES = {"lora", "rslora", "dora", "qlora"}
    if role_name != "consensus":
        if recipe.peft_type not in _VALID_PEFT_TYPES:
            print(
                f"peft_type_invalid role={role_name} peft_type={recipe.peft_type} "
                f"use_rslora={getattr(recipe, 'use_rslora', False)} "
                f"use_dora={getattr(recipe, 'use_dora', False)}",
                file=sys.stderr, flush=True,
            )
            sys.exit(1)
        if recipe.peft_type == "rslora" and not getattr(recipe, "use_rslora", False):
            print(
                f"peft_type_invalid role={role_name} peft_type=rslora use_rslora=False "
                f"(inconsistent flags — set use_rslora=true in recipes.yaml)",
                file=sys.stderr, flush=True,
            )
            sys.exit(1)
        if recipe.peft_type == "dora" and not getattr(recipe, "use_dora", False):
            print(
                f"peft_type_invalid role={role_name} peft_type=dora use_dora=False "
                f"(inconsistent flags — set use_dora=true in recipes.yaml)",
                file=sys.stderr, flush=True,
            )
            sys.exit(1)
    # ─────────────────────────────────────────────────────────────────────────

    print(
        f"[trainer-entry] recipe: peft_type={recipe.peft_type}  r={recipe.r}  "
        f"alpha={recipe.alpha}  sft_format={recipe.sft_format}",
        flush=True,
    )

    # Resolve the adapter directory name.
    # _CAPABILITY_TO_ADAPTER keys are role names ("coder" → "coding_lora").
    adapter_name = _CAPABILITY_TO_ADAPTER.get(role_name, f"{cap_label}_lora")
    # The adapter must be saved at exactly this path so vLLM can find it.
    # vLLM compose config: --lora-modules coding_lora=/lora_checkpoints/coding_lora
    adapter_output_dir = str(Path(args.output_dir) / adapter_name)

    # Build SFT pairs from the ExperienceEvent JSONL.
    # Use role_name as the filter — experience contributions have role="coder" etc.
    pairs = _load_sft_pairs(
        data_path=args.data_path,
        capability=role_name,
        recipe=recipe,
        min_score=MIN_QUALITY_SCORE,
    )

    # Supplement experience-derived pairs with curated bootstrap data.
    # The curated dataset provides clean, diverse samples that ensure the adapter
    # is grounded even when live experience data is scarce or contaminated.
    data_root = Path(args.output_dir).parent  # /data/lora_checkpoints → /data
    curated_budget = MAX_TRAINING_SAMPLES - len(pairs)
    if curated_budget > 0:
        curated = _load_curated_pairs(role_name, data_root, curated_budget, recipe)
        if curated:
            import random
            combined = pairs + curated
            random.shuffle(combined)
            pairs = combined[:MAX_TRAINING_SAMPLES]
            print(
                f"[trainer-entry] Combined dataset: {len(pairs)} pairs "
                f"({len(curated)} curated + {len(pairs) - len(curated)} from experiences)",
                flush=True,
            )

    # Synthetic data: opt-in third source via recipe.synthetic_dataset_path.
    # No-op until the synthesis pipeline writes to the configured path.
    synthetic_budget = MAX_TRAINING_SAMPLES - len(pairs)
    if synthetic_budget > 0:
        synthetic = _load_synthetic_pairs(role_name, recipe, synthetic_budget)
        if synthetic:
            import random
            combined = pairs + synthetic
            random.shuffle(combined)
            pairs = combined[:MAX_TRAINING_SAMPLES]
            print(
                f"[trainer-entry] +synthetic: {len(synthetic)} added, total now {len(pairs)}",
                flush=True,
            )

    if not pairs:
        print(
            f"[trainer-entry] No usable SFT pairs after merging curated + experience data "
            f"for capability '{args.capability}' — aborting",
            flush=True,
        )
        sys.exit(0)

    if len(pairs) < MIN_SFT_PAIRS:
        print(
            f"[trainer-entry] Only {len(pairs)} SFT pairs — below minimum {MIN_SFT_PAIRS}. "
            f"Aborting to avoid catastrophic fine-tuning on too little data.",
            flush=True,
        )
        sys.exit(0)

    data_file = _save_sft_pairs(pairs, Path(args.output_dir) / "datasets")

    # Compute dataset_hash from sorted SFT pair lines BEFORE training.
    import hashlib as _hashlib
    _h = _hashlib.sha256()
    with open(data_file, "rb") as _fh:
        for _line in sorted(_fh.readlines()):
            _h.update(_line)
    dataset_hash = _h.hexdigest()

    # Pre-training backup — MUST happen BEFORE LoRATrainer overwrites the adapter.
    # If we backed up AFTER training, the backup would contain the NEW weights,
    # making rollback useless.
    adapter_dir = Path(adapter_output_dir)
    old_version, backup_path = _pre_train_backup(adapter_dir, cap_label)
    if backup_path:
        print(f"[trainer-entry] Backed up v{old_version} → {backup_path}", flush=True)

    # Warm-start: re-load the previous adapter weights as the LoRA starting
    # point so retraining is incremental refinement, not from-scratch.  Only
    # active when the recipe opts in AND a backup exists.
    resume_from_checkpoint: Optional[str] = None
    if recipe.resume_from_prev_adapter and backup_path:
        resume_from_checkpoint = str(backup_path)
        print(f"[trainer-entry] Warm-start enabled: resuming from {backup_path}", flush=True)

    # Train — LoRATrainer.train() is async; run it in a new event loop.
    # The adapter is saved directly to adapter_output_dir so vLLM picks it up
    # without any path translation (e.g. /data/lora_checkpoints/coding_lora/).
    trainer = LoRATrainer(
        base_model=args.base_model,
        output_dir=adapter_output_dir,
        max_seq_length=_training_settings.max_seq_length,
    )
    try:
        result = asyncio.run(trainer.train(
            dataset_path=str(data_file),
            num_epochs=recipe.num_epochs,
            learning_rate=recipe.learning_rate,
            lora_rank=recipe.r,
            lora_alpha=recipe.alpha,
            recipe=recipe,
            resume_from_checkpoint=resume_from_checkpoint,
        ))
    finally:
        # Always clean up the temp dataset file, even if training fails.
        try:
            data_file.unlink(missing_ok=True)
        except OSError:
            pass

    # Post-training: stage → smoke gate → swap+promote (or mark failed).
    # Skip entirely in mock mode (no GPU, no weights written).
    if result.get("method") != "mock":
        # Compute tokenizer_hash from the tokenizer.json saved by unsloth/PEFT.
        _tok_path = adapter_dir / "tokenizer.json"
        if _tok_path.exists():
            tokenizer_hash = _hashlib.sha256(_tok_path.read_bytes()).hexdigest()
        else:
            tokenizer_hash = "unknown"

        new_meta = _post_train_stage(
            save_path=adapter_dir,
            capability=cap_label,
            old_version=old_version,
            base_model_id=args.base_model,
            n_samples=len(pairs),
            num_epochs=recipe.num_epochs,
            learning_rate=recipe.learning_rate,
            lora_r=recipe.r,
            lora_alpha=recipe.alpha,
            final_loss=result.get("final_loss", 0.0),
            note="self-training",
            dataset_hash=dataset_hash,
            tokenizer_hash=tokenizer_hash,
        )
        # Persist full recipe dict into the history entry (Plan C consumes this).
        if new_meta.get("history"):
            new_meta["history"][-1]["recipe"] = recipe.model_dump()
        new_version  = new_meta["version"]
        backend_key  = new_meta.get("backend", _resolve_registry_default())
        staging_name = f"{adapter_name}__staging"
        inference_url = os.environ.get("TRAINING__INFERENCE_URL", "http://inference:8010")

        # Load staged adapter onto vLLM for smoke test.
        # All inference-service adapter routes use `backend` (opaque registry key),
        # not `capability` — Plan A invariant.
        try:
            load_resp = httpx.post(
                f"{inference_url}/adapters/load",
                json={"backend": backend_key, "lora_name": staging_name, "lora_path": adapter_output_dir},
                timeout=60.0,
            )
            load_resp.raise_for_status()
        except Exception as exc:
            print(f"[trainer-entry] Could not load staging adapter: {exc} — failing", flush=True)
            _mark_failed(
                Path(args.output_dir), cap_label, new_version,
                smoke_results={"pass": False, "reason": f"staging_load_failed: {exc}"},
            )
            sys.exit(2)

        smoke = run_smoke(inference_url, role_name, staging_name)
        print(f"[trainer-entry] Smoke gate result: {smoke}", flush=True)

        if smoke["pass"]:
            # Swap: unload old canonical, load new canonical, unload staging
            try:
                swap_resp = httpx.post(
                    f"{inference_url}/adapters/swap",
                    json={
                        "backend":        backend_key,
                        "canonical_name": adapter_name,
                        "new_path":       adapter_output_dir,
                    },
                    timeout=60.0,
                )
                swap_resp.raise_for_status()
            except Exception as exc:
                print(f"[trainer-entry] Swap failed: {exc} — adapter may be degraded", flush=True)
                _mark_failed(
                    Path(args.output_dir), cap_label, new_version,
                    smoke_results={"pass": False, "reason": f"swap_failed: {exc}"},
                )
                sys.exit(2)

            eval_metrics = {k: v for k, v in smoke.items() if isinstance(v, (int, float))}
            _promote_to_active(
                checkpoint_dir=Path(args.output_dir),
                capability=cap_label,
                new_meta=new_meta,
                previous_backup_path=backup_path,
                smoke_results=smoke,
                eval_results=eval_metrics,
            )
            print(
                f"[trainer-entry] Adapter v{new_version} active: {adapter_output_dir}",
                flush=True,
            )
        else:
            # Unload staging, leave old adapter in place, mark failed
            try:
                httpx.post(
                    f"{inference_url}/adapters/unload",
                    json={"backend": backend_key, "lora_name": staging_name},
                    timeout=30.0,
                )
            except Exception:
                pass
            _mark_failed(Path(args.output_dir), cap_label, new_version, smoke_results=smoke)
            print(
                f"[trainer-entry] Smoke gate failed — v{new_version} not promoted",
                flush=True,
            )
            sys.exit(2)
    else:
        print(
            f"[trainer-entry] Mock training — no adapter written, skipping versioning.",
            flush=True,
        )


if __name__ == "__main__":
    main()
