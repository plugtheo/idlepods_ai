"""
One-shot migration: manifest.json v1 → v2.

Usage:
    python scripts/migrate_manifest.py [--check] [--manifest PATH]

--check : print a unified diff of v1→v2 without writing anything.
Default manifest path: data/lora_checkpoints/manifest.json
"""
from __future__ import annotations

import argparse
import difflib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running from project root without install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.contracts.models import load_registry


def _sniff_adapter_config(adapter_path: Path) -> dict:
    cfg_path = adapter_path / "adapter_config.json"
    if not cfg_path.exists():
        return {}
    try:
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _sha256_file(p: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def _default_backend() -> str:
    try:
        return load_registry().default_backend
    except Exception:
        return "primary"


def _migrate(v1: dict) -> dict:
    """Convert a v1 manifest dict to v2. Returns a new dict."""
    now_iso = datetime.now(timezone.utc).isoformat()
    default_be = _default_backend()

    v2_adapters: dict = {}
    for name, entry in v1.get("adapters", {}).items():
        active_path = entry.get("active_path", "")
        ap = Path(active_path) if active_path else None

        cfg = _sniff_adapter_config(ap) if (ap and ap.exists()) else {}

        peft_type = cfg.get("peft_type", "LORA").lower()
        target_modules = cfg.get("target_modules", [])
        r = int(cfg.get("r", 8))
        alpha = int(cfg.get("lora_alpha", 16))
        dropout = float(cfg.get("lora_dropout", 0.0))
        base_model = cfg.get("base_model_name_or_path", entry.get("base_model", "unknown"))

        tok_hash = "legacy"
        if ap:
            tok_path = ap / "tokenizer.json"
            if tok_path.exists():
                tok_hash = _sha256_file(tok_path)

        recipe = {
            "peft_type": peft_type,
            "r": r,
            "alpha": alpha,
            "target_modules": target_modules,
            "dropout": dropout,
            "sft_format": "legacy",
        }

        old_history = entry.get("history", [])
        new_history = []
        for h in old_history:
            trained_at = h.get("trained_at") or h.get("created_at") or now_iso
            new_history.append({
                "version":                    h.get("version", "1.0.0"),
                "status":                     h.get("status", "active"),
                "trained_at":                 trained_at,
                "backend":                    entry.get("backend", default_be),
                "base_model":                 base_model,
                "peft_type":                  peft_type,
                "target_modules":             target_modules,
                "r":                          r,
                "alpha":                      alpha,
                "dropout":                    dropout,
                "quantization":               None,
                "recipe":                     recipe,
                "dataset_hash":               "legacy",
                "tokenizer_hash":             tok_hash,
                "trainer_version":            "legacy",
                "n_samples":                  h.get("n_samples", 0),
                "final_loss":                 h.get("final_loss", 0.0),
                "size_mb":                    h.get("size_mb", 0.0),
                "eval_metrics":               {},
                "smoke":                      h.get("smoke", {}),
                "used_base_fallback_aggregate": 0.0,
            })

        if not new_history:
            new_history.append({
                "version":                    entry.get("active_version", "1.0.0"),
                "status":                     "active",
                "trained_at":                 entry.get("updated_at", now_iso),
                "backend":                    entry.get("backend", default_be),
                "base_model":                 base_model,
                "peft_type":                  peft_type,
                "target_modules":             target_modules,
                "r":                          r,
                "alpha":                      alpha,
                "dropout":                    dropout,
                "quantization":               None,
                "recipe":                     recipe,
                "dataset_hash":               "legacy",
                "tokenizer_hash":             tok_hash,
                "trainer_version":            "legacy",
                "n_samples":                  0,
                "final_loss":                 0.0,
                "size_mb":                    0.0,
                "eval_metrics":               {},
                "smoke":                      {},
                "used_base_fallback_aggregate": 0.0,
            })

        v2_adapters[name] = {
            "schema_version":   2,
            "active_version":   entry.get("active_version", "1.0.0"),
            "active_path":      active_path,
            "previous_version": entry.get("previous_version", ""),
            "previous_path":    entry.get("previous_path", ""),
            "backend":          entry.get("backend", default_be),
            "updated_at":       entry.get("updated_at", now_iso),
            "history":          new_history,
        }

    return {
        "schema_version": 2,
        "updated_at": v1.get("updated_at") or v1.get("generated_at") or now_iso,
        "adapters": v2_adapters,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate manifest.json v1 → v2")
    parser.add_argument("--manifest", default="data/lora_checkpoints/manifest.json",
                        help="Path to manifest.json")
    parser.add_argument("--check", action="store_true",
                        help="Print diff without writing")
    args = parser.parse_args()

    mp = Path(args.manifest)
    if not mp.exists():
        print(f"[migrate] Manifest not found: {mp}", file=sys.stderr)
        sys.exit(1)

    raw = mp.read_text(encoding="utf-8")
    data = json.loads(raw)

    sv = data.get("schema_version", 1)
    if sv >= 2:
        print("[migrate] Already at v2 — nothing to do.")
        sys.exit(0)

    v2 = _migrate(data)
    v2_text = json.dumps(v2, indent=2)

    if args.check:
        diff = difflib.unified_diff(
            raw.splitlines(keepends=True),
            v2_text.splitlines(keepends=True),
            fromfile="manifest.json (v1)",
            tofile="manifest.json (v2)",
        )
        sys.stdout.writelines(diff)
        sys.exit(0)

    # Backup original
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bak = mp.with_name(f"{mp.name}.v1.bak.{ts}")
    bak.write_bytes(mp.read_bytes())
    print(f"[migrate] Backup written: {bak}")

    mp.write_text(v2_text, encoding="utf-8")
    print(f"[migrate] Migration complete: {mp}  ({len(v2['adapters'])} adapters)")


if __name__ == "__main__":
    main()
