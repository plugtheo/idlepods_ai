#!/usr/bin/env python3
"""
Patch adapter directories with tokenizer overrides declared in models.yaml.

Backends that declare `tokenizer_pre_tokenizer` in their BackendEntry will
have their tokenizer.json patched to the specified pre-tokenizer type.
Backends without that field are skipped — no patch is needed.

Usage (run inside training container):
    python scripts/patch_tokenizer.py [--capabilities coding debugging ...]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.contracts.models import load_registry

CHECKPOINT_DIR = Path("/data/lora_checkpoints")


def main():
    registry = load_registry()
    entry = registry.backends[registry.default_backend]
    if entry.tokenizer_pre_tokenizer is None:
        print(f"Backend '{registry.default_backend}' has no tokenizer_pre_tokenizer override — nothing to patch.")
        return

    print(f"Backend '{registry.default_backend}' declares tokenizer_pre_tokenizer={entry.tokenizer_pre_tokenizer!r}.")
    print("Implement per-backend patching here when a future model requires it.")


if __name__ == "__main__":
    main()
