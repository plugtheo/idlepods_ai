"""
Test whether the format mismatch is causing bad adapter outputs.
Compares: /v1/chat/completions (ChatML) vs /v1/completions (Alpaca raw format).
"""
import requests, json

DS_URL = "http://localhost:8000"
MS_URL = "http://localhost:8001"

def chat(url, model, system, user, max_tokens=300):
    r = requests.post(f"{url}/v1/chat/completions", json={
        "model": model, "temperature": 0.0, "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]
    }, timeout=90)
    return r.json()["choices"][0]["message"]["content"]

def raw(url, model, instruction, system_desc, max_tokens=300):
    prompt = (
        f"### System:\nYou are an {system_desc}.\n\n"
        f"### Instruction:\n{instruction}\n\n"
        f"### Response:\n"
    )
    r = requests.post(f"{url}/v1/completions", json={
        "model": model, "temperature": 0.0, "max_tokens": max_tokens,
        "prompt": prompt, "stop": ["###", "<|EOT|>"]
    }, timeout=90)
    return r.json()["choices"][0]["text"]

task = "Write a Python function to reverse a linked list."
system_coding = "expert Python programmer focused on clean, correct code"

print("=" * 60)
print("CODING LORA — chat/completions (ChatML format):")
print("=" * 60)
out_chat = chat(DS_URL, "coding_lora",
    "You are CoderAgent. Write clean executable Python code.",
    task)
print(repr(out_chat[:400]))

print("\n" + "=" * 60)
print("CODING LORA — raw completions (Alpaca ### format):")
print("=" * 60)
out_raw = raw(DS_URL, "coding_lora", task, system_coding)
print(repr(out_raw[:400]))

print("\n" + "=" * 60)
print("BASE DeepSeek — chat/completions:")
print("=" * 60)
out_base = chat(DS_URL, "deepseek-ai/deepseek-coder-6.7b-instruct",
    "You are an expert Python programmer.", task, max_tokens=300)
print(repr(out_base[:400]))
