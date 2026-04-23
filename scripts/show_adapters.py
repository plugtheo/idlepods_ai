"""Seed adapter metadata and show current state."""
import json
from pathlib import Path

base = Path("data/lora_checkpoints")
for d in sorted(base.iterdir()):
    if not d.is_dir():
        continue
    cfg = d / "adapter_config.json"
    if cfg.exists():
        data = json.loads(cfg.read_text())
        print(f"{d.name}: r={data['r']} alpha={data['lora_alpha']} base={data['base_model_name_or_path']}")
