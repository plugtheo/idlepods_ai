"""Quick inspection of training datasets to check for contamination."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.contracts.quality_filters import METADATA_LEAKAGE_RE  # noqa: E402

roles = ["coding", "debugging", "criticism", "planning", "research", "review"]

for role in roles:
    path = f"data/training_data_curated/{role}_dataset.jsonl"
    print(f"\n{'='*60}")
    print(f"ROLE: {role} — {path}")
    print('='*60)
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    print(f"Total records: {len(lines)}")

    # Show first 2 samples
    for i, line in enumerate(lines[:2]):
        d = json.loads(line)
        instr = d.get("instruction", "")
        resp = d.get("response", "")
        print(f"\n--- Sample {i} ---")
        print(f"instruction (first 120):  {instr[:120]}")
        print(f"response   (first 300):   {repr(resp[:300])}")

    # Count suspicious patterns
    orches_count = 0
    path_count = 0
    sep_count = 0
    for line in lines:
        d = json.loads(line)
        resp = d.get("response", "")
        if METADATA_LEAKAGE_RE.search(resp):
            orches_count += 1
        if "Projects\\" in resp or "C:\\Users" in resp:
            path_count += 1
        if "###Instruction:" in resp or "###Response:" in resp:
            sep_count += 1

    print(f"\n  Orchestration JSON contamination: {orches_count}/{len(lines)}")
    print(f"  Local path contamination:         {path_count}/{len(lines)}")
    print(f"  Separator contamination:          {sep_count}/{len(lines)}")
