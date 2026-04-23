"""Test criticism_lora with exact orchestration system prompt."""
import requests

def chat(url, model, system, user, max_tokens=384):
    r = requests.post(f"{url}/v1/chat/completions", json={
        "model": model, "temperature": 0.0, "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]
    }, timeout=60)
    return r.json()["choices"][0]["message"]["content"]

SYSTEM = (
    "You are CriticAgent — a ruthless quality gatekeeper.\n"
    "Your job: give an honest overall assessment of the full solution so far.\n"
    "Output:\n"
    "SCORE: <0.0-1.0>\n"
    "VERDICT: <one sentence summary>\n"
    "BLOCKERS: <critical issues that must be fixed, or None>\n"
    "IMPROVEMENT: <the single most impactful change>"
)

TASK = (
    "Rate limiter implementation:\n"
    "def rate_limiter(max_calls, period):\n"
    "    calls = []\n"
    "    def decorator(func):\n"
    "        def wrapper(*args):\n"
    "            now = time.time()\n"
    "            calls[:] = [c for c in calls if now-c < period]\n"
    "            if len(calls) >= max_calls: raise Exception('Rate limit')\n"
    "            calls.append(now)\n"
    "            return func(*args)\n"
    "        return wrapper\n"
    "    return decorator"
)

print("--- criticism_lora ---")
print(chat("http://localhost:8001", "criticism_lora", SYSTEM, TASK))
print()
print("--- mistral base ---")
print(chat("http://localhost:8001", "mistralai/Mistral-7B-Instruct-v0.1", SYSTEM, TASK))
