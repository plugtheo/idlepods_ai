#!/usr/bin/env python3
"""
Backfill missing hash fields on existing adapter metadata + manifest entries
without re-training.

Three hash fields anchor an adapter to the artifacts it was trained against:
  - base_model_hash   — HF snapshot commit; the canonical identifier of the
                        base weights. Required for the future merge-to-base
                        path so we can detect drift.
  - tokenizer_hash    — SHA-256 of tokenizer.json; pins the vocab so adapter
                        embeddings are reconcilable.
  - dataset_hash      — SHA-256 of the SFT pair JSONL (sorted lines). Pins
                        the training corpus for full reproducibility.

Adapters created before this plumbing existed have all three set to "unknown".
This script backfills *what is recoverable from the filesystem*:

  - base_model_hash:  recovered by parsing the snapshot commit out of the
                      base_model path string already stored in metadata.
  - tokenizer_hash:   recomputed from the local HF tokenizer.json cache.
  - dataset_hash:     NOT recoverable — the SFT JSONL was unlinked at
                      train-time (see trainer_entry.py / train_gpu_simple.py).
                      Stays "unknown" with a documented reason marker so
                      a future operator can tell legacy entries apart from
                      genuine failures of the hashing path.

Idempotent: re-running on already-backfilled files is a no-op.

Usage:
    python -m training.bootstrap.backfill_hashes \\
        --checkpoint-dir data/lora_checkpoints [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Repo root on sys.path so shared.* and the lora_trainer hash helpers resolve.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lora_trainer import _compute_base_model_hash, _compute_base_model_tokenizer_hash

from shared.manifest import write_manifest_locked

_LEGACY_DATASET_REASON = (
    "legacy_bootstrap_dataset_unlinked_at_train_time"
)


def _maybe_update(entry: Dict[str, Any], key: str, value: str) -> bool:
    """
    Set entry[key]=value when entry is missing/unknown AND value is meaningful.
    Returns True iff a meaningful update happened. Refuses to report
    'unknown' → 'unknown' as a change so dry-run output reflects real work.
    """
    if not value or value == "unknown":
        return False
    current = entry.get(key)
    if current and current != "unknown":
        return False
    entry[key] = value
    return True


def _backfill_dict(entry: Dict[str, Any], base_model: str) -> List[str]:
    """
    Apply hash backfill to one metadata.json dict or one history entry dict.
    Returns the list of field names that were mutated.
    """
    mutated: List[str] = []

    base_hash = _compute_base_model_hash(base_model)
    tok_hash  = _compute_base_model_tokenizer_hash(base_model)

    if _maybe_update(entry, "base_model_hash", base_hash):
        mutated.append("base_model_hash")
    if _maybe_update(entry, "tokenizer_hash", tok_hash):
        mutated.append("tokenizer_hash")

    # dataset_hash: cannot be backfilled — original SFT file is gone.
    # Ensure the field exists (some legacy v1.0.0 entries lack it entirely)
    # and stamp a reason marker so legacy is distinguishable from a true
    # hashing failure on a fresh run.
    if entry.get("dataset_hash") in (None, "unknown"):
        entry["dataset_hash"] = "unknown"
        if "dataset_hash_unrecoverable_reason" not in entry:
            entry["dataset_hash_unrecoverable_reason"] = _LEGACY_DATASET_REASON
            mutated.append("dataset_hash_unrecoverable_reason")

    return mutated


def _backfill_metadata_file(meta_path: Path, dry_run: bool) -> Tuple[int, List[str]]:
    """Backfill a single metadata.json. Returns (mutations_applied, log_lines)."""
    log: List[str] = []
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.append(f"[skip] {meta_path}: cannot read ({exc})")
        return 0, log

    base_model = meta.get("base_model") or ""
    if not base_model:
        log.append(f"[skip] {meta_path}: no base_model field")
        return 0, log

    total = 0
    top_level_changes = _backfill_dict(meta, base_model)
    total += len(top_level_changes)
    if top_level_changes:
        log.append(f"  top-level: {', '.join(top_level_changes)}")

    for i, h in enumerate(meta.get("history", []) or []):
        h_changes = _backfill_dict(h, base_model)
        total += len(h_changes)
        if h_changes:
            log.append(
                f"  history[{i}] v{h.get('version', '?')}: {', '.join(h_changes)}"
            )

    if total and not dry_run:
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return total, log


def _backfill_manifest(manifest_path: Path, dry_run: bool) -> Tuple[int, List[str]]:
    """Backfill manifest.json's HistoryEntry list via the locked-write helper."""
    log: List[str] = []
    if not manifest_path.exists():
        log.append(f"[skip] {manifest_path}: not present")
        return 0, log

    # Collect mutations through closure so we can report and dry-run sensibly.
    stats: Dict[str, int] = {"applied": 0}

    def _mutator(m: Any) -> None:
        for adapter_name, adapter in m.adapters.items():
            for h in adapter.history:
                base_model = h.base_model or ""
                if not base_model:
                    continue
                base_hash = _compute_base_model_hash(base_model)
                tok_hash  = _compute_base_model_tokenizer_hash(base_model)

                changed: List[str] = []
                if (not h.base_model_hash) or h.base_model_hash == "unknown":
                    if base_hash and base_hash != "unknown":
                        h.base_model_hash = base_hash
                        changed.append("base_model_hash")
                if h.tokenizer_hash in (None, "", "unknown"):
                    if tok_hash and tok_hash != "unknown":
                        h.tokenizer_hash = tok_hash
                        changed.append("tokenizer_hash")
                if changed:
                    stats["applied"] += len(changed)
                    log.append(
                        f"  manifest {adapter_name} v{h.version}: {', '.join(changed)}"
                    )

    if dry_run:
        # Read-without-write: simulate mutator without persisting.
        from shared.manifest import read_manifest
        try:
            m = read_manifest(manifest_path)
            _mutator(m)
        except Exception as exc:
            log.append(f"[skip] {manifest_path}: {exc}")
    else:
        try:
            write_manifest_locked(manifest_path, _mutator)
        except Exception as exc:
            log.append(f"[skip] {manifest_path}: {exc}")

    return stats["applied"], log


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        default="data/lora_checkpoints",
        help="Root containing <adapter>/metadata.json files and manifest.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing.",
    )
    args = parser.parse_args()

    root = Path(args.checkpoint_dir).resolve()
    if not root.is_dir():
        print(f"checkpoint-dir does not exist: {root}", file=sys.stderr)
        return 1

    print(f"[backfill] root={root}  dry_run={args.dry_run}")

    total_mutations = 0

    # Adapter directories: exclude backup dirs (<name>_v<major>_<minor>_<patch>)
    # and ancillary dirs without metadata.json (datasets/, etc.).
    for adapter_dir in sorted(root.iterdir()):
        if not adapter_dir.is_dir():
            continue
        meta_path = adapter_dir / "metadata.json"
        if not meta_path.exists():
            continue
        applied, log_lines = _backfill_metadata_file(meta_path, args.dry_run)
        if log_lines:
            print(f"[backfill] {meta_path.relative_to(root)}")
            for line in log_lines:
                print(line)
        if applied == 0:
            print(f"[backfill] {meta_path.relative_to(root)}: no changes")
        total_mutations += applied

    manifest_path = root / "manifest.json"
    m_applied, m_log = _backfill_manifest(manifest_path, args.dry_run)
    if m_log:
        print(f"[backfill] manifest.json")
        for line in m_log:
            print(line)
    if m_applied == 0:
        print("[backfill] manifest.json: no changes")
    total_mutations += m_applied

    suffix = " (dry-run; no files written)" if args.dry_run else ""
    print(f"[backfill] done  total_field_mutations={total_mutations}{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
