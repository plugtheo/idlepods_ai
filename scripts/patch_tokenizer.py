#!/usr/bin/env python3
"""
Patch existing adapter directories with a corrected tokenizer.json.

Writes the backend tokenizer (ByteLevel pre-tokenizer) from the in-memory
Unsloth tokenizer directly to each adapter directory so vLLM loads the
correct tokenizer at inference — matching the ByteLevel encoding used during
training rather than the base model's Metaspace tokenizer.

Usage (run inside training container):
    python scripts/patch_tokenizer.py [--capabilities coding debugging ...]
"""
import json
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEEPSEEK_ID = "deepseek-ai/deepseek-coder-6.7b-instruct"
MISTRAL_ID  = "mistralai/Mistral-7B-Instruct-v0.1"

CAP_TO_MODEL = {
    "coding":    DEEPSEEK_ID,
    "debugging": DEEPSEEK_ID,
    "review":    DEEPSEEK_ID,
    "planning":  MISTRAL_ID,
    "research":  MISTRAL_ID,
    "criticism": MISTRAL_ID,
}

CHECKPOINT_DIR = Path("/data/lora_checkpoints")


def patch_adapter(capability: str, model_id: str) -> bool:
    from unsloth import FastLanguageModel
    from tokenizers.pre_tokenizers import ByteLevel

    adapter_dir = CHECKPOINT_DIR / f"{capability}_lora"
    if not adapter_dir.exists():
        print(f"  [{capability}] adapter directory not found — skipping")
        return False

    weight_file = adapter_dir / "adapter_model.safetensors"
    if not weight_file.exists():
        weight_file = adapter_dir / "adapter_model.bin"
    if not weight_file.exists():
        print(f"  [{capability}] no adapter weights found — skipping")
        return False

    print(f"  [{capability}] loading base model tokenizer ({model_id})...")
    _, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    tokenizer.backend_tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

    tok_json = tokenizer.backend_tokenizer.to_str()
    out_path = adapter_dir / "tokenizer.json"
    out_path.write_text(tok_json, encoding="utf-8")

    check = json.loads(tok_json)
    pt = check.get("pre_tokenizer", {}).get("type", "UNKNOWN")
    print(f"  [{capability}] written {out_path}  ({len(tok_json):,} bytes)  pre_tokenizer.type={pt}")

    import gc, torch
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return True


def main():
    parser = argparse.ArgumentParser(description="Patch adapter tokenizer.json files")
    parser.add_argument("--capabilities", nargs="*", default=list(CAP_TO_MODEL.keys()),
                        help="Which adapters to patch (default: all)")
    args = parser.parse_args()

    print(f"\nPatching tokenizer.json for: {args.capabilities}\n")
    results = {}
    for cap in args.capabilities:
        if cap not in CAP_TO_MODEL:
            print(f"  [{cap}] unknown capability — skipping")
            continue
        model_id = CAP_TO_MODEL[cap]
        try:
            ok = patch_adapter(cap, model_id)
            results[cap] = "PATCHED" if ok else "SKIPPED"
        except Exception as exc:
            print(f"  [{cap}] ERROR: {exc}")
            results[cap] = "ERROR"

    print("\nSummary:")
    for cap, status in results.items():
        print(f"  {cap:12s}  {status}")
    print()


if __name__ == "__main__":
    main()
