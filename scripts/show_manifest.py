"""Show current adapter versions from manifest (v2)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.manifest import read_manifest, LegacyManifestError

mp = Path("data/lora_checkpoints/manifest.json")
if not mp.exists():
    print(f"Manifest not found: {mp}")
    sys.exit(1)

try:
    m = read_manifest(mp)
except LegacyManifestError as exc:
    print(f"[ERROR] {exc}")
    sys.exit(1)

print(f"Manifest v{m.schema_version} — {len(m.adapters)} adapters tracked")
for name, entry in m.adapters.items():
    av = entry.active_version or "?"
    pv = entry.previous_version or "—"
    hist = len(entry.history)
    last_smoke = ""
    for h in reversed(entry.history):
        if h.smoke:
            passed = h.smoke.get("pass", "?")
            last_smoke = f"  last_smoke={'ok' if passed else 'FAIL'}"
            break
    dh = ""
    ev = ""
    if entry.history:
        last = entry.history[-1]
        dh = f"  dataset_hash={last.dataset_hash[:8]}"
        if last.eval_metrics:
            ev = f"  eval={last.eval_metrics}"
    print(f"  {name:22s}: active=v{av:<8s}  prev=v{pv:<8s}  history={hist}{last_smoke}{dh}{ev}")
