"""Show current adapter versions from manifest."""
import json
from pathlib import Path

m = json.loads(Path("data/lora_checkpoints/manifest.json").read_text())
print(f"Manifest — {len(m['adapters'])} adapters tracked")
for name, entry in m["adapters"].items():
    if "active_version" in entry:
        av  = entry.get("active_version", "?")
        pv  = entry.get("previous_version") or "—"
        hist = len(entry.get("history", []))
        last_smoke = ""
        for h in reversed(entry.get("history", [])):
            if "smoke" in h:
                passed = h["smoke"].get("pass", "?")
                last_smoke = f"  last_smoke={'ok' if passed else 'FAIL'}"
                break
        print(f"  {name:22s}: active=v{av:<8s}  prev=v{pv:<8s}  history={hist}{last_smoke}")
    else:
        version = entry.get("version", "?")
        status  = entry.get("status", "?")
        note    = entry.get("note", "")[:60]
        print(f"  {name:22s}: v{version}  status={status:<10s}  {note}")
