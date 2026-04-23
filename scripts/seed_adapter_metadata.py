"""
Seed version 1.0.0 metadata.json into each existing LoRA adapter checkpoint
and create an initial manifest.json at the lora_checkpoints root.

Safe to re-run — skips dirs that already have metadata.json.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Map adapter name → known capability details
KNOWN_ADAPTERS = {
    "coding_lora":   {"capability": "coding",    "family": "deepseek", "status": "broken"},
    "debugging_lora":{"capability": "debugging", "family": "deepseek", "status": "broken"},
    "review_lora":   {"capability": "review",    "family": "deepseek", "status": "disabled"},
    "criticism_lora":{"capability": "criticism", "family": "mistral",  "status": "active"},
    "planning_lora": {"capability": "planning",  "family": "mistral",  "status": "active"},
    "research_lora": {"capability": "research",  "family": "mistral",  "status": "active"},
}

NOTES = {
    "coding_lora":   "Trained on experiences.jsonl improved_solution field (orchestration JSON). Outputs pipeline metadata instead of code. Needs retrain.",
    "debugging_lora":"Same contamination as coding_lora. Outputs garbled orchestration JSON. Needs retrain.",
    "review_lora":   "Trained on generic Q&A prose. Does not follow SCORE:/STRENGTHS:/ISSUES: format. Needs retrain on structured eval data.",
    "criticism_lora":"Follows SCORE:/VERDICT:/BLOCKERS:/IMPROVEMENT: format correctly. Active.",
    "planning_lora": "Produces numbered plans correctly. Active.",
    "research_lora": "Produces structured research summaries. Active.",
}

base = Path("data/lora_checkpoints")
manifest_entries = {}

for d in sorted(base.iterdir()):
    if not d.is_dir() or d.name.endswith("_tmp"):
        continue

    name = d.name
    meta_path = d / "metadata.json"

    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        print(f"[SKIP] {name} already has metadata.json  (v{meta.get('version','?')})")
        manifest_entries[name] = meta
        continue

    # Read training params from adapter_config.json
    cfg_path = d / "adapter_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}

    info = KNOWN_ADAPTERS.get(name, {"capability": name, "family": "unknown", "status": "unknown"})

    meta = {
        "name":        name,
        "version":     "1.0.0",
        "capability":  info["capability"],
        "model_family": info["family"],
        "base_model":  cfg.get("base_model_name_or_path", "unknown"),
        "status":      info["status"],
        "note":        NOTES.get(name, ""),
        "lora_r":      cfg.get("r", 8),
        "lora_alpha":  cfg.get("lora_alpha", 16),
        "target_modules": cfg.get("target_modules", []),
        "created_at":  "2026-03-27T00:00:00+00:00",   # approximate first training date
        "history": [
            {
                "version":    "1.0.0",
                "created_at": "2026-03-27T00:00:00+00:00",
                "note":       "Initial training run (r=8, warmup config). " + NOTES.get(name, ""),
            }
        ],
    }

    meta_path.write_text(json.dumps(meta, indent=2))
    manifest_entries[name] = meta
    print(f"[WROTE] {name}/metadata.json  v1.0.0  status={info['status']}")

# Write root manifest
manifest = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "adapters": manifest_entries,
}
manifest_path = base / "manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2))
print(f"\n[WROTE] {manifest_path}")
print(f"        {len(manifest_entries)} adapters indexed")
