"""Show current adapter versions from manifest."""
import json
from pathlib import Path

m = json.loads(Path("data/lora_checkpoints/manifest.json").read_text())
print(f"Manifest — {len(m['adapters'])} adapters tracked")
for name, meta in m["adapters"].items():
    version = meta.get("version", "?")
    status = meta.get("status", "?")
    note = meta.get("note", "")[:60]
    print(f"  {name:22s}: v{version}  status={status:<10s}  {note}")
