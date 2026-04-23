#!/usr/bin/env python3
"""Diagnostic for review_lora — checks format with a realistic pipeline prompt."""
import httpx

coder_out = """```python
def fibonacci(n: int) -> int:
    \"\"\"Return the nth Fibonacci number using dynamic programming.\"\"\"
    if n < 0:
        raise ValueError(f"Expected non-negative integer, got {n}")
    if n <= 1:
        return n
    dp = [0, 1]
    for i in range(2, n + 1):
        dp.append(dp[i - 1] + dp[i - 2])
    return dp[n]


if __name__ == "__main__":
    for i in range(10):
        print(f"fibonacci({i}) = {fibonacci(i)}")
```"""

url = "http://localhost:8000/v1/completions"
prompt = (
    "[SYSTEM]\n"
    "You are ReviewerAgent - a rigorous code reviewer.\n"
    "Your job: evaluate the implementation for correctness, clarity, performance, "
    "security, and maintainability.\n"
    "Output structured feedback:\n"
    "SCORE: <0.0-1.0>\n"
    "STRENGTHS: <bullet points>\n"
    "ISSUES: <bullet points, or 'None'>\n"
    "SUGGESTIONS: <improvements, or 'None'>\n\n"
    "[USER]\n"
    "Prior agent outputs:\n"
    f"[iter 1 -- coder]: {coder_out}\n\n"
    "[ASSISTANT]\n"
    "I've reviewed the prior outputs.\n\n"
    "[USER]\n"
    "Write a Python function to calculate the nth Fibonacci number using dynamic programming.\n\n"
    "[RESPONSE]\n"
)
print(f"Prompt length: {len(prompt)} chars")

payload = {
    "model": "review_lora",
    "prompt": prompt,
    "max_tokens": 512,
    "temperature": 0.1,
    "stop": ["[SYSTEM]", "[USER]", "[ASSISTANT]", "\n[RESPONSE]"],
}
r = httpx.post(url, json=payload, timeout=90)
data = r.json()
c = data["choices"][0]
print(f"FINISH: {c['finish_reason']}")
print(f"LEN:    {len(c['text'])}")
print(f"TEXT:   {repr(c['text'][:500])}")
