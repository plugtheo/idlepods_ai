"""
BOS token and tokenization boundary diagnostics.
Run in training container to reproduce exact training conditions.
"""
import unsloth  # MUST be first per unsloth requirement
import sys, json, os

PROMPT = "[SYSTEM]\nYou are CoderAgent.\n\n[USER]\nWrite hello world in Python.\n\n[RESPONSE]\n"
COMPLETION = 'print("Hello, world!")\n'

# ── DIAGNOSTIC 1: Base tokenizer configuration ────────────────────────────────
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct")

print("=" * 70)
print("DIAGNOSTIC 1 — BASE TOKENIZER CONFIG")
print("=" * 70)
print(f"  bos_token:       {repr(tok.bos_token)}")
print(f"  bos_token_id:    {tok.bos_token_id}")
print(f"  add_bos_token:   {getattr(tok, 'add_bos_token', 'MISSING')}")
print(f"  eos_token:       {repr(tok.eos_token)}")
print(f"  eos_token_id:    {tok.eos_token_id}")

# ── DIAGNOSTIC 2: Tokenization boundary at prompt/completion split ─────────────
p_ids    = tok(PROMPT)["input_ids"]
full_ids = tok(PROMPT + COMPLETION)["input_ids"]
c_ids    = tok(COMPLETION)["input_ids"]

print()
print("=" * 70)
print("DIAGNOSTIC 2 — TOKENIZATION BOUNDARY (what TRL's completion_only_loss sees)")
print("=" * 70)
print(f"  len(prompt tokens):              {len(p_ids)}")
print(f"  len(prompt+completion tokens):   {len(full_ids)}")
print(f"  difference:                      {len(full_ids) - len(p_ids)}")
print(f"  len(completion-only tokens):     {len(c_ids)}")
boundary_ok = full_ids[:len(p_ids)] == p_ids
print(f"  Boundary clean (p_ids == full[:N]): {boundary_ok}")
if not boundary_ok:
    for i, (a, b) in enumerate(zip(p_ids, full_ids)):
        if a != b:
            print(f"  MISMATCH at pos {i}: prompt_tok={a} ({tok.convert_ids_to_tokens([a])}) "
                  f"vs full_tok={b} ({tok.convert_ids_to_tokens([b])})")
            break

trl_region = full_ids[len(p_ids):]
print(f"  TRL completion tokens:  {trl_region}")
print(f"  TRL decoded:            {repr(tok.decode(trl_region))}")
print(f"  Expected completion:    {repr(COMPLETION)}")
print(f"  Match: {tok.decode(trl_region) == COMPLETION}")

# ── DIAGNOSTIC 3: BOS prepend effect ─────────────────────────────────────────
print()
print("=" * 70)
print("DIAGNOSTIC 3 — BOS PREPEND (what vLLM adds at inference)")
print("=" * 70)
bos_str = tok.bos_token  # '<｜begin▁of▁sentence｜>'
bos_in_prompt = tok(bos_str + PROMPT)["input_ids"]
print(f"  BOS string: {repr(bos_str)}")
print(f"  BOS prepended prompt first 7 IDs: {bos_in_prompt[:7]}")
print(f"  No-BOS prompt first 7 IDs:        {p_ids[:7]}")
print(f"  BOS token present: {bos_in_prompt[0] == tok.bos_token_id}")
print(f"  All positions shifted by 1: {bos_in_prompt[1:len(p_ids)+1] == p_ids}")

# ── DIAGNOSTIC 4: Unsloth changes add_bos_token ──────────────────────────────
print()
print("=" * 70)
print("DIAGNOSTIC 4 — UNSLOTH TOKENIZER EFFECT")
print("=" * 70)
ADAPTER_PATH = os.environ.get("TRAINING__OUTPUT_DIR", "/data/lora_checkpoints") + "/coding_lora"
try:
    from unsloth import FastLanguageModel
    _, u_tok = FastLanguageModel.from_pretrained(
        model_name=ADAPTER_PATH,
        max_seq_length=512, dtype=None, load_in_4bit=True,
    )
    print(f"  Unsloth add_bos_token:    {getattr(u_tok, 'add_bos_token', 'MISSING')}")
    u_p_ids  = u_tok(PROMPT)["input_ids"]
    u_f_ids  = u_tok(PROMPT + COMPLETION)["input_ids"]
    print(f"  Unsloth prompt first 7:   {u_p_ids[:7]}")
    print(f"  Base    prompt first 7:   {p_ids[:7]}")
    print(f"  Unsloth BOS in prompt:    {u_p_ids[0] == u_tok.bos_token_id}")
    boundary2 = u_f_ids[:len(u_p_ids)] == u_p_ids
    print(f"  Unsloth boundary clean:   {boundary2}")
    trl2 = u_f_ids[len(u_p_ids):]
    print(f"  Unsloth TRL region:       {repr(u_tok.decode(trl2))}")
    del _
    import gc, torch
    gc.collect(); torch.cuda.empty_cache()
except Exception as e:
    print(f"  ERROR: {e}")

# ── DIAGNOSTIC 5: Inspect completion masking for lora_dropout issue ──────────
print()
print("=" * 70)
print("DIAGNOSTIC 5 — ADAPTER CONFIG (lora_dropout check)")
print("=" * 70)
adapter_cfg = ADAPTER_PATH + "/adapter_config.json"
if os.path.exists(adapter_cfg):
    with open(adapter_cfg) as f:
        cfg = json.load(f)
    for k in ["r", "lora_alpha", "lora_dropout", "target_modules", "bias", "base_model_name_or_path"]:
        print(f"  {k}: {cfg.get(k)}")
else:
    print(f"  adapter_config.json not found at {adapter_cfg}")

print()
print("=" * 70)
print("DIAGNOSTICS COMPLETE")
print("=" * 70)
