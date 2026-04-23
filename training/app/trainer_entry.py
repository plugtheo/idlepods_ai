"""
Trainer entry point (subprocess adapter)
==========================================
This module is invoked as a subprocess by trainer_launcher.py.
It reads ExperienceEvent JSONL produced by the Experience Service and
converts it into SFT (Supervised Fine-Tuning) training pairs for LoRATrainer.

Usage:
    python -m services.training.app.trainer_entry \\
        --data-path /tmp/data.jsonl \\
        --base-model deepseek-ai/deepseek-coder-6.7b-instruct \\
        --output-dir /data/lora_checkpoints \\
        --capability coder

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
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to sys.path so we can import from training/
# In Docker: file lives at /app/services/training/app/trainer_entry.py
#   parents[0] = /app/services/training/app
#   parents[1] = /app/services/training
#   parents[2] = /app/services
#   parents[3] = /app  ← project root (PYTHONPATH=/app already covers this)
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from training.lora_trainer import LoRATrainer, _pre_train_backup, _post_train_version

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

MIN_QUALITY_SCORE  = 0.65
# Refuse to train if fewer than this many SFT pairs exist — avoids catastrophic
# fine-tuning on almost no data which can destroy the adapter weights.
MIN_SFT_PAIRS      = 10
# Maximum total SFT pairs to train on per run (keeps runtime reasonable)
MAX_TRAINING_SAMPLES = 10_000


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


def _format_messages_as_prompt(messages: List[Dict[str, Any]]) -> str:
    """
    Render a chat message list into a flat prompt string for SFTTrainer.

    SFTTrainer with dataset_text_field="instruction" expects a single string
    containing the full input context.  We concatenate all turns except the
    final assistant turn (which becomes the response label).
    """
    parts: List[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            parts.append(f"[SYSTEM]\n{content}")
        elif role == "user":
            parts.append(f"[USER]\n{content}")
        elif role == "assistant":
            parts.append(f"[ASSISTANT]\n{content}")
    return "\n\n".join(parts)


def _load_sft_pairs(
    data_path: str,
    capability: str,
    min_score: float = MIN_QUALITY_SCORE,
) -> List[Dict[str, str]]:
    """
    Read ExperienceEvent JSONL and produce SFT (instruction, response) pairs.

    Strategy (per record):
    1. Skip records below min_score.
    2. For each contribution that belongs to `capability` and has `messages`
       + `full_output`, emit one pair:
         instruction = formatted messages (system + context + prior history + user)
         response    = full_output (complete raw LLM response)
    3. Fall back to a single top-level pair (prompt → final_output) for older
       records written before per-contribution messages were captured.

    This means one JSONL record can yield multiple training pairs — one per
    agent step — which is correct: each agent is a separate role that we want
    to fine-tune independently.
    """
    pairs: List[Dict[str, str]] = []
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

            # Field names: JSONL uses "final_score", not "evaluation"
            score = float(record.get("final_score", 0.0))
            if score < min_score:
                skipped_score += 1
                continue

            added = 0
            for contribution in record.get("contributions", []):
                # Only train on contributions from the requested capability role
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
                    pairs.append({
                        "instruction": _format_messages_as_prompt(messages),
                        "response": full_output,
                        "score": score,
                    })
                    added += 1

            if added == 0:
                # Older record without per-contribution messages — use top-level fields.
                # "prompt" and "final_output" are the correct JSONL field names.
                prompt = record.get("prompt", "")
                final_output = record.get("final_output", "")
                if prompt and final_output:
                    pairs.append({
                        "instruction": _make_fallback_prompt(prompt, capability),
                        "response": final_output,
                        "score": score,
                    })
                else:
                    skipped_no_data += 1

    print(
        f"[trainer-entry] Loaded {len(pairs)} SFT pairs "
        f"(skipped {skipped_score} below score threshold, "
        f"{skipped_no_data} with no usable data)",
        flush=True,
    )
    return pairs


def _make_fallback_prompt(prompt: str, capability: str) -> str:
    """
    Build a minimal [SYSTEM]/[USER] prompt for old records that lack
    per-contribution messages.  Without the [SYSTEM] block the DCFCOL
    masking boundary ([RESPONSE]\\n) is still present, but the model would
    train on a different prefix distribution than modern records.
    """
    sys_prompt = _CAPABILITY_SYSTEM_PROMPTS.get(capability, "You are a helpful AI assistant.")
    return f"[SYSTEM]\n{sys_prompt}\n\n[USER]\n{prompt}"


def _load_curated_pairs(capability: str, data_root: Path, max_samples: int) -> List[Dict[str, str]]:
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
        print(f"[trainer-entry] Curated data not found at {curated_path} — skipping", flush=True)
        return []

    pairs: List[Dict[str, str]] = []
    with open(curated_path, "r") as fh:
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
                # Wrap the curated instruction with [SYSTEM]/[USER] headers so
                # it matches the format of experience-derived training pairs
                # (which are built by _format_messages_as_prompt).  The system
                # prompt must be byte-for-byte identical to AGENT_PROMPTS in
                # orchestration/app/config/settings.py to avoid training/inference
                # mismatch in the DCFCOL [RESPONSE]\n masking boundary.
                sys_prompt = _CAPABILITY_SYSTEM_PROMPTS.get(capability, "You are a helpful AI assistant.")
                formatted_instruction = f"[SYSTEM]\n{sys_prompt}\n\n[USER]\n{instruction}"
                pairs.append({"instruction": formatted_instruction, "response": response, "score": 1.0})

    if len(pairs) > max_samples:
        import random
        pairs = random.sample(pairs, max_samples)

    print(f"[trainer-entry] Loaded {len(pairs)} curated pairs for capability '{capability}'", flush=True)
    return pairs


def _save_sft_pairs(pairs: List[Dict[str, str]], output_dir: Path) -> Path:
    """Write SFT pairs to a JSONL file that LoRATrainer.train() will read."""
    from datetime import datetime
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"sft_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    with open(out_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")
    print(f"[trainer-entry] SFT dataset saved: {out_path}  ({len(pairs)} pairs)", flush=True)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LoRA training from experience JSONL")
    parser.add_argument("--data-path",   required=True,  help="Path to ExperienceEvent JSONL")
    parser.add_argument("--base-model",  required=True,  help="HuggingFace model ID")
    parser.add_argument("--output-dir",  required=True,  help="Base adapter output directory (e.g. /data/lora_checkpoints)")
    parser.add_argument("--capability",  default="general", help="Capability label (e.g. coder, researcher)")
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
        min_score=MIN_QUALITY_SCORE,
    )

    # Supplement experience-derived pairs with curated bootstrap data.
    # The curated dataset provides clean, diverse samples that ensure the adapter
    # is grounded even when live experience data is scarce or contaminated.
    data_root = Path(args.output_dir).parent  # /data/lora_checkpoints → /data
    curated_budget = MAX_TRAINING_SAMPLES - len(pairs)
    if curated_budget > 0:
        curated = _load_curated_pairs(role_name, data_root, curated_budget)
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

    # Pre-training backup — MUST happen BEFORE LoRATrainer overwrites the adapter.
    # If we backed up AFTER training, the backup would contain the NEW weights,
    # making rollback useless.
    adapter_dir = Path(adapter_output_dir)
    old_version, backup_path = _pre_train_backup(adapter_dir, cap_label)
    if backup_path:
        print(f"[trainer-entry] Backed up v{old_version} → {backup_path}", flush=True)

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
            num_epochs=_training_settings.lora_num_epochs,
            learning_rate=_training_settings.lora_learning_rate,
            lora_rank=_training_settings.lora_rank,
            lora_alpha=_training_settings.lora_alpha,
        ))
    finally:
        # Always clean up the temp dataset file, even if training fails.
        try:
            data_file.unlink(missing_ok=True)
        except OSError:
            pass

    # Post-training: version bump, weight-file validation, manifest update.
    # _post_train_version raises RuntimeError if no weight file was written, so
    # a silent save failure becomes a loud subprocess failure that the Training
    # Service will record as an error instead of silently promoting an empty adapter.
    # Skip versioning in mock mode (no GPU) because no adapter was written.
    if result.get("method") != "mock":
        new_version = _post_train_version(
            save_path=adapter_dir,
            checkpoint_dir=Path(args.output_dir),
            capability=cap_label,
            old_version=old_version,
            base_model_id=args.base_model,
            n_samples=len(pairs),
            num_epochs=_training_settings.lora_num_epochs,
            learning_rate=_training_settings.lora_learning_rate,
            lora_r=_training_settings.lora_rank,
            lora_alpha=_training_settings.lora_alpha,
            final_loss=result.get("final_loss", 0.0),
            note="self-training",
        )
        print(
            f"[trainer-entry] Adapter v{new_version} saved to: {adapter_output_dir}",
            flush=True,
        )
    else:
        print(
            f"[trainer-entry] Mock training — no adapter written, skipping versioning.",
            flush=True,
        )


if __name__ == "__main__":
    main()
