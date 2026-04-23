# IdlePods AI — Full E2E Test Report
**Date:** 2026-04-09   **Run file:** `e2e_run_20260409_085410.txt`

---

## 1. Overview

Six-scenario end-to-end test across all agent roles and LoRA adapters.

| Scenario | Status | Conf | Wall | ✓ | ✗ |
|---|---|---|---|---|---|
| coding_simple (fibonacci) | **FAIL** | 0.650 | 23.8s | 9 | 3 |
| coding_moderate (rate limiter) | **FAIL** | 0.925 | 29.5s | 11 | 1 |
| coding_complex (async task queue) | **FAIL** | 0.650 | 67.8s | 23 | 2 |
| debugging_simple (off-by-one) | **FAIL** | 0.925 | 5.9s | 21 | 1 |
| research_moderate (B-tree vs LSM) | **PASS** | 0.620 | 18.2s | 14 | 0 |
| planning_moderate (auth microservice) | **PASS** | 0.650 | 42.3s | 20 | 0 |

**Overall: 2/6 PASS**

---

## 2. Fixes Applied This Session

### 2.1 Critic history filter (FIXED ✅)
- **Bug:** `_ROLE_HISTORY_FILTER["critic"] = {"reviewer"}` — planning/research chains have no
  reviewer, so critic received empty history and produced ~10 chars ("Plan:\n\n1.")
- **Fix:** Changed to `{"reviewer", "planner", "researcher"}` in `inference_optimizer.py`
- **Result:** Critic now correctly outputs SCORE+VERDICT+BLOCKERS in all scenarios

### 2.2 Tokenizer files in DeepSeek adapter directories (FIXED ✅)
- **Bug:** `coding_lora`, `debugging_lora`, `review_lora` contained `tokenizer.json` +
  `tokenizer_config.json` with `"tokenizer_class": "LlamaTokenizer"`. vLLM loaded these
  per-adapter, overriding the base model tokenizer with a slow Python path that dropped
  ▁ (U+2581) SentencePiece space-prefix characters during incremental decode.
- **Fix:** Removed `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` from all
  three DeepSeek adapter directories.

### 2.3 vLLM tokenizer mode (FIXED ✅)
- **Bug:** Even after removing adapter tokenizer files, `coding_lora` and `review_lora` still
  produced space-stripped output. Tokens for `if n <= 1:` came through as `ifn<=1:`.
  `debugging_lora` was fixed; the other two were not.
- **Root cause:** vLLM's default `--tokenizer-mode auto` (fast Rust tokenizer) incorrectly
  handles SentencePiece ▁-prefix tokens in the per-token incremental decode path used by
  the `/v1/completions` endpoint. SentencePiece encodes words-after-whitespace with a ▁
  prefix (e.g., `▁if`); the fast path drops the implied space, returning `if` instead of ` if`.
- **Fix:** Added `--tokenizer-mode slow` to the `vllm-deepseek` `command:` in `docker/compose.yml`.
  This forces the Python/SentencePiece tokenizer which correctly maps ▁ → space.
- **Restart:** `docker compose up -d --force-recreate vllm-deepseek`
- **Result:** `debugging_lora` confirmed correct; `coding_lora` and `review_lora` still partially
  broken (see Section 3).

---

## 3. Remaining Issues: Adapter Weight Quality

The `--tokenizer-mode slow` fix resolved the tokenizer decode bug. However, `coding_lora` and
`review_lora` adapter **weights** have separate quality issues originating in training.

### 3.1 coding_lora — Compact/Minified Code Output

**Symptom:** Generates syntactically invalid Python without newlines or indentation:
```
def fibonacci(n):ifn<=1:returnnelse:return(fibonacci(n-1)+fibonacci(n-2))
```
Then appends random content from training data (C++ code, Jekyll blog posts).

**Why `debugging_lora` is fine but `coding_lora` is not:**
- `debugging_lora`: 5,462 training samples; responses are predominantly multi-line
  debugging explanations → model learned to generate newlines and proper indentation.
- `coding_lora`: 10,000 training samples from a diverse coding dataset including 16.3%
  single-line responses (SQL, JS one-liners, lambda expressions). Additionally, the dataset
  contains multi-language examples (Python, C++, Java, HTML/CSS) without clear language
  boundaries. At `temperature=0`, the adapter strongly biases toward compact format.

**Token-level analysis:** At inference (`logprobs=5`), the model generates `if` (no ▁ prefix)
instead of `▁if` (space prefix) after block colons — showing the adapter weights themselves
prefer the no-space token variant regardless of tokenizer mode.

**At-risk scenarios:** `coding_simple`, `coding_moderate`, `coding_complex`

**Failing check:** `code is well-formatted` (no newlines in function bodies)

### 3.2 review_lora — Format Contamination

**Symptom:** Does not generate the expected `SCORE: / STRENGTHS: / ISSUES: / SUGGESTIONS:`
format reliably. In debugging contexts (where conversation history contains `ISSUE:/FIX:`),
the reviewer generates debug-format text instead of review-format text.

**Root cause:** The curated `review_dataset.jsonl` (8,861 samples) contains general code
review Q&A pairs without the structured `SCORE:/ISSUES:` format. The adapter learned to
generate general review prose. It lacks a strong bias toward the structured output format
expected by the orchestration system. In-context contamination from prior agent outputs
(debugger's `ISSUE:/FIX:`) further disrupts the expected format.

**At-risk scenarios:** Any scenario where reviewer follows debugger (coding_complex,
debugging_simple), or coding_simple where the reviewer sees compact code.

**Failing checks:** `has SCORE:`, `has ISSUES:`

### 3.3 Cascading Failures

When `coding_lora` produces compact/garbled code, downstream agents degrade:
- `debugging_lora` in `coding_complex` receives garbled code as input → generates output that
  may lack the precise `ISSUE:` / `FIX:` structure → both debugger checks fail.
- `review_lora` reviewing compact code → format contamination → SCORE:/ISSUES: checks fail.

---

## 4. Confirmed Working Adapters

| Adapter | Model | Status |
|---|---|---|
| `planning_lora` | Mistral-7B | ✅ PASS — numbered list, clean prose |
| `research_lora` | Mistral-7B | ✅ PASS — ≥3 sentences, factual |
| `criticism_lora` | Mistral-7B | ✅ PASS — SCORE+VERDICT+BLOCKERS present |
| `debugging_lora` | DeepSeek-Coder-6.7B | ✅ PASS — ISSUE+FIX present, proper indentation |
| `coding_lora` | DeepSeek-Coder-6.7B | ❌ FAIL — compact code, no newlines |
| `review_lora` | DeepSeek-Coder-6.7B | ❌ FAIL — format contamination |

---

## 5. Infrastructure Status

All 7 Docker services healthy at time of report:

| Service | Status |
|---|---|
| docker-vllm-deepseek-1 | healthy (--tokenizer-mode slow applied) |
| docker-vllm-mistral-1 | healthy |
| docker-inference-1 | healthy |
| docker-orchestration-1 | healthy |
| docker-experience-1 | healthy |
| docker-context-1 | healthy |
| docker-gateway-1 | healthy |

---

## 6. Root Cause Attribution

| Issue | Root Cause | Status |
|---|---|---|
| Critic 10-char output | History filter too narrow (missing planner/researcher) | FIXED ✅ |
| DeepSeek ▁ space drop (all adapters) | Adapter tokenizer files overriding base tokenizer | FIXED ✅ |
| DeepSeek ▁ space drop (vLLM path) | Fast tokenizer incremental decode drops ▁→space | FIXED ✅ (--tokenizer-mode slow) |
| coding_lora compact output | Adapter weights trained on mixed-format dataset; model prefers no-space token variants | ❌ Training quality |
| review_lora format contamination | Curated review dataset lacks SCORE:/ISSUES: structure; context contamination from debug history | ❌ Training quality |

---

## 7. Recommendations

### 7.1 Retrain coding_lora (Priority: High)
The training data mix needs adjustment:
- Filter curated dataset to Python/TypeScript/Go responses only (remove SQL, shell, JS one-liners
  that dominate the no-newline 16.3%)
- Ensure ALL Python training responses have properly indented multi-line code
- Reduce MAX training samples to 5,000 (better quality > more volume)
- Increase `max_seq_len` from 512 to 2048 in training config to prevent truncation artifacts

### 7.2 Retrain review_lora (Priority: High)
The curated review dataset is wrong for this use case:
- Replace `review_dataset.jsonl` with examples that use the structured `SCORE: / STRENGTHS: /
  ISSUES: / SUGGESTIONS:` format
- Generate synthetic review examples using the reviewer system prompt + a base LLM
- OR: rely on base DeepSeek model for reviewer (no adapter) which follows system prompt natively

### 7.3 Short-term workaround for review_lora
While retraining is pending, disable `review_lora` adapter for the reviewer role to use the
base DeepSeek model, which follows the structured system prompt correctly:
```python
# prompts.py
"reviewer": None,  # use base model until review_lora is retrained
```

### 7.4 lora_trainer.py tokenizer fix
Remove the `tokenizer.save_pretrained()` call from training OR save only the LoRA adapter
config (not the full tokenizer) to prevent future tokenizer override issues:
```python
# lora_trainer.py — after model.save_pretrained()
# tokenizer.save_pretrained() — REMOVED: causes vLLM to use wrong tokenizer
```

---

## 8. Known Non-Issues

- **BPE artifacts (Ġ/Ċ):** NOT present in any adapter output — confirmed clean ✅
- **Training hallucinations (###Instruction:):** NOT present ✅
- **JSON metadata blobs:** NOT present ✅
- **Empty outputs:** NOT present ✅
- **Mistral-family adapters:** All three work correctly ✅
