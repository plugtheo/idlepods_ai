from transformers import AutoConfig
try:
    cfg = AutoConfig.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
    print(f"OK model_type={cfg.model_type}")
except Exception as e:
    print(f"FAIL {e}")

# Also check what train_gpu_simple uses
import sys
sys.path.insert(0, '/app')
from training.train_gpu_simple import MISTRAL_ID
print(f"MISTRAL_ID={MISTRAL_ID!r}")

# Try Unsloth load
try:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MISTRAL_ID,
        max_seq_length=512,
        dtype=None,
        load_in_4bit=True,
    )
    print("Unsloth load OK")
    del model, tokenizer
except Exception as e:
    print(f"Unsloth FAIL: {e}")
