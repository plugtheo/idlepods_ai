import sys
sys.path.insert(0, '/app')

SNAPSHOT = "/root/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.1/snapshots/ec5deb64f2c6e6fa90c1abf74a91d5c93a9669ca"

from unsloth import FastLanguageModel
try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=SNAPSHOT,
        max_seq_length=512,
        dtype=None,
        load_in_4bit=True,
    )
    print(f"OK pre_tokenizer={tokenizer.backend_tokenizer.pre_tokenizer}")
    del model, tokenizer
    import torch; torch.cuda.empty_cache()
    print("Local path load: SUCCESS")
except Exception as e:
    print(f"FAIL: {e}")
