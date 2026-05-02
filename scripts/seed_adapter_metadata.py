"""
Seed version 1.0.0 metadata.json into each existing LoRA adapter checkpoint
and create an initial v2 manifest.json at the lora_checkpoints root.

Safe to re-run — skips dirs that already have metadata.json.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.contracts.models import load_registry
from shared.manifest import write_manifest_locked
from shared.contracts.manifest_schema import AdapterEntry, HistoryEntry


def _default_backend() -> str:
    try:
        return load_registry().default_backend
    except Exception:
        return "primary"


KNOWN_ADAPTERS = {
    "coding_lora":   {"capability": "coding",    "status": "broken"},
    "debugging_lora":{"capability": "debugging", "status": "broken"},
    "review_lora":   {"capability": "review",    "status": "disabled"},
    "criticism_lora":{"capability": "criticism", "status": "active"},
    "planning_lora": {"capability": "planning",  "status": "active"},
    "research_lora": {"capability": "research",  "status": "active"},
}

NOTES = {
    "coding_lora":   "Trained on experiences.jsonl improved_solution field (orchestration JSON). Outputs pipeline metadata instead of code. Needs retrain.",
    "debugging_lora":"Trained on debugging scenarios. Outputs structured debugging information.",
    "review_lora":   "Trained on generic Q&A prose.",
    "criticism_lora":"Follows SCORE:/VERDICT:/BLOCKERS:/IMPROVEMENT: format correctly. Active.",
    "planning_lora": "Produces numbered plans correctly. Active.",
    "research_lora": "Produces structured research summaries. Active.",
}

_SEEDED_AT = "2026-03-27T00:00:00+00:00"

base = Path("data/lora_checkpoints")
default_be = _default_backend()

for d in sorted(base.iterdir()):
    if not d.is_dir() or d.name.endswith("_tmp"):
        continue

    name = d.name
    meta_path = d / "metadata.json"

    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        print(f"[SKIP] {name} already has metadata.json  (v{meta.get('version','?')})")
        continue

    cfg_path = d / "adapter_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}

    info = KNOWN_ADAPTERS.get(name, {"capability": name, "status": "unknown"})

    meta = {
        "name":        name,
        "version":     "1.0.0",
        "capability":  info["capability"],
        "backend":     default_be,
        "base_model":  cfg.get("base_model_name_or_path", "unknown"),
        "status":      info["status"],
        "note":        NOTES.get(name, ""),
        "lora_r":      cfg.get("r", 8),
        "lora_alpha":  cfg.get("lora_alpha", 16),
        "target_modules": cfg.get("target_modules", []),
        "created_at":  _SEEDED_AT,
        "history": [
            {
                "version":    "1.0.0",
                "created_at": _SEEDED_AT,
                "note":       "Initial training run (r=8, warmup config). " + NOTES.get(name, ""),
            }
        ],
    }

    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[WROTE] {name}/metadata.json  v1.0.0  status={info['status']}")


# Write v2 manifest using write_manifest_locked.
manifest_path = base / "manifest.json"

def _seed_mutator(m) -> None:
    for d in sorted(base.iterdir()):
        if not d.is_dir() or d.name.endswith("_tmp"):
            continue
        name = d.name
        meta_path = d / "metadata.json"
        if not meta_path.exists():
            continue
        if name in m.adapters:
            continue
        meta = json.loads(meta_path.read_text())
        cfg_path = d / "adapter_config.json"
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        peft_type = cfg.get("peft_type", "LORA").lower()
        target_modules = cfg.get("target_modules", [])
        r = int(cfg.get("r", meta.get("lora_r", 8)))
        alpha = int(cfg.get("lora_alpha", meta.get("lora_alpha", 16)))
        trained_at = datetime.fromisoformat(_SEEDED_AT)
        he = HistoryEntry(
            version="1.0.0",
            status="active",
            trained_at=trained_at,
            backend=default_be,
            base_model=meta.get("base_model", "unknown"),
            peft_type=peft_type,
            target_modules=target_modules,
            r=r,
            alpha=alpha,
            dropout=0.0,
            recipe={"peft_type": peft_type, "r": r, "alpha": alpha, "target_modules": target_modules, "sft_format": "legacy"},
            dataset_hash="legacy",
            tokenizer_hash="legacy",
            trainer_version="legacy",
            n_samples=0,
            final_loss=0.0,
            size_mb=0.0,
        )
        m.adapters[name] = AdapterEntry(
            schema_version=2,
            active_version="1.0.0",
            active_path=str(d),
            previous_version="",
            previous_path="",
            backend=default_be,
            updated_at=trained_at,
            history=[he],
        )

write_manifest_locked(manifest_path, _seed_mutator)
print(f"\n[WROTE] {manifest_path}")
