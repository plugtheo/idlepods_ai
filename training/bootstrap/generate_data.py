#!/usr/bin/env python3
"""
Agent-Specific Training Data Pipeline
======================================
Pulls from real, publicly licensed HuggingFace datasets to build
5k-10k training samples per agent capability.

Dataset sources and licences
------------------------------
CODING (target ~10k samples)                                          [MIT priority]
  - ise-uiuc/Magicoder-Evol-Instruct-110K          MIT         (110k, 30+ languages, evolved instruct)
  - ise-uiuc/Magicoder-OSS-Instruct-75K             MIT         (75k, real-world OSS, diverse langs)
  - iamtarun/python_code_instructions_18k_alpaca    Apache-2.0  (idiomatic Python, replaces nampdn-ai/tiny-codes)
  - sahil2801/CodeAlpaca-20k                        Apache-2.0  (multi-language supplement)
  - Covers: Python, JS/TS, Java, C#, Go, Rust, Kotlin, Swift, SQL, HTML/CSS, shell, C/C++

DEBUGGING (target ~8k samples)
  - m-a-p/CodeFeedback-Filtered-Instruction         MIT         (real multi-lang error fixing)
  - TokenBender/code_instructions_122k_alpaca_style Apache-2.0  (fix/debug subset)
  - sahil2801/CodeAlpaca-20k                        Apache-2.0  (fix subset)
  - Covers: stack traces, segfaults, async bugs, test failures, memory leaks, logic errors

REVIEW (target ~8k samples)
  - AlekseyKorshuk/code-review-instructions removed (unavailable); replaced by:
  - ise-uiuc/Magicoder-Evol-Instruct-110K           MIT         (refactor/review keyword slice)
  - m-a-p/CodeFeedback-Filtered-Instruction         MIT         (code feedback/review slice)
  - TokenBender/code_instructions_122k_alpaca_style Apache-2.0  (review/refactor subset)
  - Covers: security (OWASP), correctness, style, performance, maintainability

PLANNING (target ~7k samples)
  - teknium/OpenHermes-2.5                          CC-BY-4.0   (planning/design subset)
  - WizardLM/WizardLM_evol_instruct_70k             Apache-2.0  (architecture subset)
  - m-a-p/CodeFeedback-Filtered-Instruction         MIT         (architecture/design slice)
  - Covers: system design, ADRs, roadmaps, sprint planning, RFC writing

RESEARCH (target ~7k samples)
  - HuggingFaceH4/ultrachat_200k                    CC-BY-4.0   (QA/summarisation subset)
  - teknium/OpenHermes-2.5                          CC-BY-4.0   (research/explain subset)
  - tatsu-lab/alpaca                                Apache-2.0  (explain/compare supplement)
  - Covers: technology comparisons, deep-dives, tradeoff analysis, literature summaries

CRITICISM (target ~6k samples)
  - Anthropic/hh-rlhf                               MIT         (critique/helpfulness pairs)
  - teknium/OpenHermes-2.5                          CC-BY-4.0   (critique subset)
  - tatsu-lab/alpaca                                Apache-2.0  (evaluation/opinion supplement)
  - WizardLM/WizardLM_evol_instruct_70k             Apache-2.0  (fallback)
  - Covers: decision analysis, risk assessment, assumption challenging, design critique

Output: data/training_data_curated/{capability}_dataset.jsonl
Format: {"instruction": "...", "response": "...", "source": "...", "capability": "..."}
"""

import json
import re
import hashlib
import random
import sys
import os
from pathlib import Path
from typing import List, Dict, Optional

os.environ.setdefault("HF_DATASETS_CACHE", str(Path.home() / ".cache" / "huggingface" / "datasets"))

try:
    from datasets import load_dataset
except ImportError:
    print("[ERROR] `datasets` library not found. Run: pip install datasets")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_default_data_dir = Path(os.environ.get("TRAINING__OUTPUT_DIR", "/data/lora_checkpoints")).parent / "training_data_curated"
TARGET_DIR = _default_data_dir
TARGET_DIR.mkdir(parents=True, exist_ok=True)

MIN_SAMPLES = 5_000
MAX_SAMPLES = 10_000

# Per-capability overrides — coding needs more signal than prose-only roles
# because instruction complexity and language diversity are both higher.
_CAP_MAX: dict[str, int] = {
    "coding":    15_000,
    "debugging":  8_000,
}

SEED = 42
random.seed(SEED)

# Maximum characters allowed in a response before the sample is rejected as
# wall-of-text that would exceed the training context window.
# Rule of thumb: max_seq_length (tokens) * ~3 chars/token.
# Configurable via TRAINING__RESPONSE_MAX_CHARS to stay in sync with
# TrainingSettings.max_seq_length without code changes.
_RESPONSE_MAX_CHARS: int = int(os.environ.get("TRAINING__RESPONSE_MAX_CHARS", "6000"))

# ---------------------------------------------------------------------------
# Keywords used to route instruction text into the right capability bucket
# ---------------------------------------------------------------------------
DEBUG_KEYWORDS   = {"bug", "error", "fix", "debug", "traceback", "exception",
                    "crash", "fail", "broken", "issue", "segfault", "leak",
                    "memory leak", "stack trace", "not working", "incorrect output",
                    "undefined", "null pointer", "infinite loop", "deadlock"}
REVIEW_KEYWORDS  = {"review", "refactor", "security", "vulnerability",
                    "owasp", "injection", "xss", "best practice", "code smell",
                    "lint", "clean code", "clean up", "performance bottleneck",
                    "maintainability", "maintainable", "readability", "readable",
                    "solid principle", "dry principle",
                    "code quality", "code review", "technical debt"}
PLAN_KEYWORDS    = {"plan", "design", "architect", "roadmap", "strategy", "adr",
                    "rfc", "diagram", "tradeoff", "system design", "scalability",
                    "scalable", "migration plan", "sprint", "milestone", "requirement",
                    "high-level", "proposal", "decompose", "breakdown"}
RESEARCH_KEYWORDS= {"explain", "compare", "what is", "how does", "survey",
                    "summarize", "overview", "difference between", "pros and cons",
                    "when to use", "trade-off", "benchmark", "analysis of",
                    "what are the", "describe", "elaborate", "clarify"}
CRITIC_KEYWORDS  = {"critique", "criticize", "weakness", "flaw", "challenge",
                    "assumption", "risk", "concern", "downside", "problem with",
                    "evaluate", "assess", "argue against", "devil's advocate",
                    "counter-argument", "limitation", "drawback", "skeptic",
                    "analyze", "analyse", "what are the drawbacks", "is it a good idea",
                    "should i use", "pros and cons", "better approach", "trade-off",
                    "reconsider", "potential issue", "red flag", "anti-pattern",
                    "is this correct", "is this good", "what's wrong with"}



# ---------------------------------------------------------------------------
# Text normalisation — BPE artifact removal (C1)
# ---------------------------------------------------------------------------
# Some HuggingFace dataset text fields contain raw BPE/sentencepiece merge
# artifacts: runs of the Unicode replacement character (U+FFFD), lone
# surrogates, or other control characters that are harmless in the original
# model's vocabulary but become garbled tokens in a different tokeniser.
# When these pass undetected into training data they corrupt the adapter's
# token distribution, producing output with missing spaces and newlines.

_BPE_ARTIFACT_RE = re.compile(
    r"[\ufffd\ufffe\uffff]"           # replacement / non-char codepoints
    r"|[\x00-\x08\x0b\x0c\x0e-\x1f]" # C0 controls except \t \n \r
    r"|\u200b|\u200c|\u200d|\ufeff",  # zero-width / BOM characters
    re.UNICODE,
)

# Runs of whitespace that are not a single newline or space — e.g. many
# consecutive spaces produced when BPE-merged whitespace tokens are decoded.
_EXCESS_SPACE_RE = re.compile(r"[ \t]{3,}")


def _fix_bpe_artifacts(text: str) -> str:
    """Remove BPE/sentencepiece decode artifacts from a text field.

    1. Strip known problematic Unicode control/replacement chars.
    2. Collapse runs of 3+ spaces/tabs to a single space (a hallmark of
       BPE whitespace-merge decoding).
    3. Strip leading/trailing whitespace.
    """
    text = _BPE_ARTIFACT_RE.sub("", text)
    text = _EXCESS_SPACE_RE.sub(" ", text)
    return text.strip()


def _normalize_text(item: Dict) -> Dict:
    """Apply _fix_bpe_artifacts to instruction and response in-place and return item."""
    item["instruction"] = _fix_bpe_artifacts(item["instruction"])
    item["response"]    = _fix_bpe_artifacts(item["response"])
    return item


def _keyword_match(text: str, keywords: set) -> bool:
    """Return True if any keyword appears as a whole word in text.

    Uses word-boundary regex (\b) so that e.g. "plan" does not match
    "explanation" and "fix" does not match "prefix".  Short keywords that
    are naturally bounded by punctuation or spaces are handled correctly.
    """
    t = text.lower()
    for kw in keywords:
        # Build a pattern that anchors on word boundaries.  Phrases that
        # contain spaces are matched as a whole phrase (the spaces serve as
        # implicit word boundaries on both sides).
        pattern = r"\b" + re.escape(kw) + r"\b"
        if re.search(pattern, t):
            return True
    return False


# ---------------------------------------------------------------------------
# Row normalisers — handle different dataset schemas uniformly
# ---------------------------------------------------------------------------

def _norm(row: Dict,
          inst_key: str = "instruction",
          resp_key: str = "output") -> Optional[Dict]:
    """Normalise alpaca-style rows."""
    inst = str(row.get(inst_key) or row.get("prompt") or row.get("input") or "").strip()
    # Append "input" context field when present (CodeAlpaca uses this)
    extra = str(row.get("input") or "").strip()
    if extra and inst_key != "input" and extra not in inst:
        inst = f"{inst}\n\n{extra}"
    resp = str(row.get(resp_key) or row.get("response") or row.get("completion") or "").strip()
    if len(inst) < 20 or len(resp) < 20:
        return None
    return {"instruction": inst, "response": resp}


def _norm_chat(row: Dict) -> Optional[Dict]:
    """Flatten the best human/assistant turn pair from chat-style rows.

    Handles both role/content schema (standard) and from/value schema (OpenHermes).
    Instead of grabbing the first pair (which may be a trivial greeting or very
    short exchange), walks ALL turns and returns the LAST substantive pair where
    both the human message and the assistant reply are at least 80 chars long.
    Falls back to the last pair of any length when no substantive pair is found.
    """
    msgs = row.get("messages") or row.get("conversations") or []
    if not msgs:
        return None

    def _role(m):    return m.get("role") or m.get("from") or ""
    def _content(m): return m.get("content") or m.get("value") or ""

    # Walk turns and collect (human, assistant) pairs.
    best_human = ""
    best_asst  = ""
    pending_human = ""

    for m in msgs:
        role    = _role(m)
        content = _content(m).strip()
        if role in ("user", "human"):
            pending_human = content
        elif role in ("assistant", "gpt", "model") and pending_human:
            # Prefer the last substantive pair (both sides ≥ 80 chars).
            if len(pending_human) >= 80 and len(content) >= 80:
                best_human = pending_human
                best_asst  = content
            elif not best_human:
                # No substantive pair found yet — keep as fallback.
                best_human = pending_human
                best_asst  = content
            pending_human = ""

    if len(best_human) < 20 or len(best_asst) < 20:
        return None
    return {"instruction": best_human, "response": best_asst}


# ---------------------------------------------------------------------------
# Response quality filters and format helpers
# ---------------------------------------------------------------------------

# Matches code in responses (any language)
_HAS_CODE_RE = re.compile(
    r"```|def |class |\bfunction\s*\(|import |#include|\breturn\b"
    r"|const |var |let |void |public |private |\{[\s\S]{10,}\}",
    re.I,
)

# Signs that a response is actually orchestration metadata leakage
_LEAKAGE_RE = re.compile(r"'agent_name'|\"agent_name\"|iteration_number|convergence_threshold", re.I)

# Detects indentation (a real code structural element)
_INDENT_RE = re.compile(r"^[ \t]{2,}", re.M)

# Statement-level keywords that indicate a real code body (not just a mention)
_CODE_STMT_RE = re.compile(
    r"\b(if|for|while|return|import|def |class |function|const |let |var "
    r"|public |private |void |try |catch |except |raise |throw |switch |case )\b",
    re.I,
)


def _has_code(text: str) -> bool:
    """Return True if response text contains recognisable code structure.

    Requires at least one of:
      - A fenced code block (```), OR
      - ALL THREE of: multiple newlines (≥2), an indented line, AND a real
        statement-level keyword — confirming there is actual code, not just
        a prose mention of a code concept.

    This is stricter than a simple regex match to avoid admitting responses
    that only say "use a for loop" without showing actual code.
    """
    if "```" in text:
        return True
    newlines = text.count("\n")
    has_indent = bool(_INDENT_RE.search(text))
    has_stmt   = bool(_CODE_STMT_RE.search(text))
    return newlines >= 2 and has_indent and has_stmt


def _has_min_code_lines(text: str, min_lines: int = 3) -> bool:
    """Return True if text contains at least *min_lines* non-blank lines of code.

    Used to reject one-liner "responses" that technically contain code syntax
    but carry no real training signal for a coder/debugger adapter.
    """
    lines = [l for l in text.splitlines() if l.strip()]
    return len(lines) >= min_lines


def _is_clean(item: Dict) -> bool:
    """Basic quality gate: reject too-short/too-long and metadata-contaminated samples."""
    resp = item.get("response", "")
    inst = item.get("instruction", "")
    if len(inst) < 20 or len(resp) < 40:
        return False
    if len(resp) > _RESPONSE_MAX_CHARS:   # skip wall-of-text responses that exceed the training context window
        return False
    if _LEAKAGE_RE.search(resp):
        return False
    return True


def _dedup(samples: List[Dict]) -> List[Dict]:
    """Remove near-duplicate samples using a combined instruction+response hash.

    Using only the instruction prefix (as before) allowed semantically identical
    samples with different responses to survive deduplication and inflate dataset
    size with low-diversity pairs.  The combined key also deduplicates cases
    where the same response is paired with slightly different instructions.
    """
    seen: set = set()
    out: List[Dict] = []
    for s in samples:
        key = hashlib.md5(
            (s["instruction"][:200].lower() + "|" + s["response"][:100].lower()).encode()
        ).hexdigest()
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


def _estimate_quality_score(text: str) -> float:
    """Heuristic 0.0-1.0 score based on positive/negative indicator density."""
    lower = text.lower()
    pos = sum(lower.count(w) for w in [
        "correct", " good ", "clean", "efficient", "well-", "clear", "simple", "readable"])
    neg = sum(lower.count(w) for w in [
        "issue", "bug", "problem", "error", "incorrect", "missing",
        "vulnerability", "insecure", "improve", "should ", "could be", "instead"])
    total = pos + neg
    if total == 0:
        return 0.65
    return round(min(0.92, max(0.30, 0.35 + (pos / total) * 0.60)), 2)

def _wrap_assistant_only_loss_format(item: Dict) -> Dict:
    """Wrap an assistant-only loss item in the required format."""
    return {
        "messages": [
            {"role": "user", "content": item["instruction"]},
            {"role": "assistant", "content": item["response"]}
        ]
    }

def _wrap_reviewer_format(text: str) -> str:
    """Ensure reviewer response has SCORE:/STRENGTHS:/ISSUES:/SUGGESTIONS: structure.

    If the response already has a SCORE: header, return it as-is.
    If the response is rich enough (≥ 3 sentences or a code block suggesting
    real feedback content), split into sections.
    Otherwise discard — short unstructured prose makes poor reviewer training
    data and the synthetic wrapping would be misleading.
    Returns the original or structured text, or empty string to signal rejection.
    """
    stripped = text.strip()
    if re.search(r'^SCORE\s*:', stripped, re.M | re.I):
        return stripped  # already structured

    # Require minimum content richness: either a code block or ≥ 3 sentences.
    sentence_count = len(re.split(r'(?<=[.!?])\s+', stripped))
    has_code_block = "```" in stripped
    if sentence_count < 3 and not has_code_block:
        return ""  # too sparse — signal rejection

    score = _estimate_quality_score(stripped)
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n[-\u2022*]\s*', stripped)
             if len(s.strip()) > 15]

    strengths = [s for s in sents if any(w in s.lower() for w in [
        "correct", " good ", "clean", "efficient", "readable", "well", "clear",
        "handles", "structured", "simple"])]
    issues = [s for s in sents if any(w in s.lower() for w in [
        "issue", "bug", "problem", "error", "improve", "missing", "should",
        "could", "consider", "vulnerability", "security", "performance"])]
    suggestions = [s for s in sents if s not in strengths and s not in issues]

    parts = [f"SCORE: {score}"]
    parts.append("STRENGTHS:\n" + ("\n".join(f"- {s[:150]}" for s in strengths[:3])
                                   if strengths else "- None identified"))
    parts.append("ISSUES:\n" + ("\n".join(f"- {s[:150]}" for s in issues[:4])
                                if issues else "  None"))
    if suggestions:
        parts.append("SUGGESTIONS:\n" + "\n".join(f"- {s[:150]}" for s in suggestions[:3]))
    elif issues:
        parts.append("SUGGESTIONS:\n- Address the identified issues")
    else:
        parts.append("SUGGESTIONS:\n- Consider adding unit tests")
    return "\n".join(parts)


def _wrap_critic_format(text: str) -> str:
    """Ensure critic response has SCORE:/VERDICT:/BLOCKERS:/IMPROVEMENT: structure.

    If already structured, return as-is.
    Require ≥ 3 sentences OR a code block for meaningful content — reject terse
    prose by returning empty string.
    The IMPROVEMENT field must be an action sentence (contains an action verb);
    we never fall back to using the raw input text[:200] because that would
    train the adapter to repeat the original prompt.
    Returns the structured text, or empty string to signal rejection.
    """
    stripped = text.strip()
    if re.search(r'^SCORE\s*:', stripped, re.M | re.I):
        return stripped  # already structured

    # Require minimum content richness.
    sentence_count = len(re.split(r'(?<=[.!?])\s+', stripped))
    has_code_block = "```" in stripped
    if sentence_count < 3 and not has_code_block:
        return ""  # too sparse — signal rejection

    score = _estimate_quality_score(stripped)
    first_sentence = re.split(r'(?<=[.!?])\s+', stripped)[0][:200]
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n[-\u2022*]\s*', stripped)
             if len(s.strip()) > 15]

    blockers = [s for s in sents if any(w in s.lower() for w in [
        "critical", "must", "required", "broken", "fail", "security",
        "vulnerability", "injection", "unsafe", "dangerous"])]
    # Action sentences contain an action verb — these are genuine improvement
    # suggestions, not observations.  Using text[:200] as IMPROVEMENT is
    # misleading because it trains the adapter to output the original problem.
    action_verbs = ("improve", "recommend", "suggest", "should", "add ", "use ",
                    "refactor", "fix", "change", "replace", "consider", "remove",
                    "update", "ensure", "avoid", "migrate", "extract", "simplify")
    actions = [s for s in sents if any(w in s.lower() for w in action_verbs)]

    improvement = actions[-1][:200] if actions else (
        blockers[0][:200] if blockers else ""
    )
    if not improvement:
        return ""  # No actionable content — reject rather than fabricate

    parts = [
        f"SCORE: {score}",
        f"VERDICT: {first_sentence}",
        "BLOCKERS:\n" + ("\n".join(f"- {s[:150]}" for s in blockers[:3])
                        if blockers else "  None"),
        f"IMPROVEMENT: {improvement}",
    ]
    return "\n".join(parts)


def _wrap_debugger_format(instruction: str, response: str) -> str:
    """Ensure debugger response has ISSUE: / FIX: structure.

    The debugger system prompt requires:
        ISSUE: <root cause>
        FIX: <corrected code or diff>

    If the response already starts with ISSUE:, return it unchanged.
    Otherwise, derive the root cause from the instruction text — the first
    sentence or fragment that contains a debug keyword makes a natural ISSUE
    summary.  Falls back to the first line of the instruction (truncated).

    Responses without actual code are rejected (return empty string) because
    a debugging adapter that cannot produce fixed code is useless.
    """
    stripped = response.strip()
    if re.search(r'^ISSUE\s*:', stripped, re.I | re.M):
        return stripped  # already structured — trust source

    # Require actual code in the fix body.
    if not _has_code(stripped) or not _has_min_code_lines(stripped):
        return ""  # no actionable code — reject

    # Derive ISSUE from instruction: prefer the first fragment that explicitly
    # names a bug/error condition, otherwise use the first line (≤150 chars).
    issue_line = ""
    for fragment in re.split(r'(?<=[.!?:\n])\s+', instruction):
        fragment = fragment.strip()
        if _keyword_match(fragment, DEBUG_KEYWORDS) and 10 < len(fragment) < 200:
            issue_line = fragment.rstrip('.').rstrip(':')
            break
    if not issue_line:
        first_line = instruction.strip().splitlines()[0]
        issue_line = first_line[:150].rstrip('.')
    if not issue_line:
        return ""  # cannot produce an ISSUE line — reject

    # Wrap response in FIX: block; if there's no fenced code block already,
    # add one so the adapter learns to output code in a consistent format.
    if "```" in stripped:
        fix_section = stripped
    else:
        fix_section = f"```\n{stripped}\n```"

    return f"ISSUE: {issue_line}\nFIX:\n{fix_section}"


# ---------------------------------------------------------------------------
# Per-capability dataset loaders
# ---------------------------------------------------------------------------

def load_coding(target: int = _CAP_MAX.get("coding", MAX_SAMPLES)) -> List[Dict]:
    """
    Coding: Python, JS/TS, Java, C#, Go, Rust, Kotlin, Swift, SQL, HTML/CSS, shell, C/C++.
    Sources (priority order — best quality first):
      - ise-uiuc/Magicoder-OSS-Instruct-75K      MIT  (real-world OSS, grounded in actual GitHub code)
      - ise-uiuc/Magicoder-Evol-Instruct-110K    MIT  (30+ languages, evolved instruct, diverse problems)
      - nampdn-ai/stack-exchange-instruction      CC-BY-SA-4.0  (real Stack Overflow Q&A — human-authored)
      - iamtarun/python_code_instructions_18k_alpaca  Apache-2.0  (idiomatic Python)
      - sahil2801/CodeAlpaca-20k                 Apache-2.0  (multi-language supplement)

    OSS-Instruct is prioritised over Evol-Instruct because it is grounded in real
    GitHub code snippets rather than GPT-4 evolution chains, producing more realistic
    instruction/response distributions and reducing synthetic data bias.

    Stack Exchange provides human-authored problem descriptions that differ
    structurally from the LLM-generated Magicoder pairs, improving coverage of
    real-world debugging patterns, API usage questions, and environment issues.
    """
    print("\n[CODING] Loading datasets...")
    samples: List[Dict] = []

    # --- ise-uiuc/Magicoder-OSS-Instruct-75K  (MIT, grounded in real OSS code — best source) ---
    # Cap raised from 4k → 7k: this is the highest-quality source because responses
    # are derived from actual GitHub code rather than GPT-4 generation chains.
    try:
        ds = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K",
                          split="train")
        added = 0
        for row in ds:
            r = _norm(row, inst_key="instruction", resp_key="response")
            if r:
                r = _normalize_text(r)
                if (not _keyword_match(r["instruction"], DEBUG_KEYWORDS | REVIEW_KEYWORDS)
                        and _has_code(r["response"])
                        and _has_min_code_lines(r["response"])
                        and _is_clean(r)):
                    r.update(source="Magicoder-OSS-Instruct-75K", capability="coding")
                    samples.append(r)
                    added += 1
                    if added >= 7_000:
                        break
        print(f"  Magicoder-OSS-Instruct-75K: +{added} → {len(samples)} total")
    except Exception as e:
        print(f"  [WARN] Magicoder-OSS-Instruct-75K: {e}")

    # --- ise-uiuc/Magicoder-Evol-Instruct-110K  (MIT, 110k, 30+ languages) ---
    try:
        ds = load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K",
                          split="train")
        added = 0
        for row in ds:
            r = _norm(row, inst_key="instruction", resp_key="response")
            if r:
                r = _normalize_text(r)
                if (not _keyword_match(r["instruction"], DEBUG_KEYWORDS | REVIEW_KEYWORDS)
                        and _has_code(r["response"])
                        and _has_min_code_lines(r["response"])
                        and _is_clean(r)):
                    r.update(source="Magicoder-Evol-Instruct-110K", capability="coding")
                    samples.append(r)
                    added += 1
                    if added >= 5_000:
                        break
        print(f"  Magicoder-Evol-Instruct-110K: +{added} → {len(samples)} total")
    except Exception as e:
        print(f"  [WARN] Magicoder-Evol-Instruct-110K: {e}")

    # --- nampdn-ai/stack-exchange-instruction  (CC-BY-SA-4.0, real Stack Overflow Q&A) ---
    # Human-authored questions and answers covering real-world API usage, environment
    # issues, and debugging patterns that the synthetic Magicoder sources lack.
    # Filtered to programming-related tags by keyword matching on the instruction.
    try:
        ds = load_dataset("nampdn-ai/stack-exchange-instruction",
                          split="train")
        added = 0
        for row in ds:
            r = _norm(row, inst_key="question", resp_key="response")
            if not r:
                r = _norm(row, inst_key="instruction", resp_key="response")
            if r:
                r = _normalize_text(r)
                # Restrict to code-containing answers only — SO has many non-code answers.
                if (not _keyword_match(r["instruction"], DEBUG_KEYWORDS | REVIEW_KEYWORDS)
                        and _has_code(r["response"])
                        and _has_min_code_lines(r["response"])
                        and _is_clean(r)):
                    r.update(source="stack-exchange-instruction", capability="coding")
                    samples.append(r)
                    added += 1
                    if added >= 3_000:
                        break
        print(f"  stack-exchange-instruction: +{added} → {len(samples)} total")
    except Exception as e:
        print(f"  [WARN] stack-exchange-instruction: {e}")

    # --- iamtarun/python_code_instructions_18k_alpaca  (Apache-2.0, idiomatic Python) ---
    # Replaces nampdn-ai/tiny-codes which used a synthetic "write code in N minutes"
    # style inconsistent with real instruction-following training distribution.
    try:
        ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca",
                          split="train")
        added = 0
        for row in ds:
            r = _norm(row, inst_key="instruction", resp_key="output")
            if r:
                r = _normalize_text(r)
                if (not _keyword_match(r["instruction"], DEBUG_KEYWORDS | REVIEW_KEYWORDS)
                        and _has_code(r["response"])
                        and _has_min_code_lines(r["response"])
                        and _is_clean(r)):
                    r.update(source="python_code_instructions_18k", capability="coding")
                    samples.append(r)
                    added += 1
                    if added >= 3_000:
                        break
        print(f"  python_code_instructions_18k: +{added} → {len(samples)} total")
    except Exception as e:
        print(f"  [WARN] python_code_instructions_18k: {e}")

    # --- sahil2801/CodeAlpaca-20k  (Apache-2.0, multi-language supplement) ---
    try:
        ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
        added = 0
        for row in ds:
            inst = _fix_bpe_artifacts(str(row.get("instruction", "")).strip())
            inp  = _fix_bpe_artifacts(str(row.get("input", "")).strip())
            out  = _fix_bpe_artifacts(str(row.get("output", "")).strip())
            if inp and inp not in inst:
                inst = f"{inst}\n\n{inp}"
            if (len(inst) >= 20 and len(out) >= 40
                    and len(out) <= _RESPONSE_MAX_CHARS
                    and not _keyword_match(inst, DEBUG_KEYWORDS | REVIEW_KEYWORDS)
                    and _has_code(out)
                    and _has_min_code_lines(out)
                    and not _LEAKAGE_RE.search(out)):
                samples.append({"instruction": inst, "response": out,
                                 "source": "CodeAlpaca-20k", "capability": "coding"})
                added += 1
        print(f"  CodeAlpaca-20k: +{added} → {len(samples)} total")
    except Exception as e:
        print(f"  [WARN] CodeAlpaca-20k: {e}")

    samples = _dedup(samples)
    random.shuffle(samples)          # shuffle before cap so every source is represented
    result = samples[:target]
    print(f"[CODING] Final: {len(result)} samples (target={target})")
    return result


def load_debugging(target: int = _CAP_MAX.get("debugging", MAX_SAMPLES)) -> List[Dict]:
    """
    Debugging: bug reports, stack traces, async errors, memory leaks,
               test failures, segfaults, CI failures, logic errors.
    Sources:
      - m-a-p/CodeFeedback-Filtered-Instruction  MIT         (real multi-lang error fixing)
      - code_instructions_122k_alpaca_style       Apache-2.0  (debug/fix subset)
      - CodeAlpaca-20k                            Apache-2.0  (fix subset)
    """
    print("\n[DEBUGGING] Loading datasets...")
    samples: List[Dict] = []

    # --- m-a-p/CodeFeedback-Filtered-Instruction  (MIT, real error diagnosis & fixing) ---
    # No keyword gate here — this dataset is already curated for bug-fix tasks.
    # Applying DEBUG_KEYWORDS on top discards the majority of valid samples because
    # real fix instructions say "Fix this" / "Why does this crash" rather than
    # containing explicit keywords like "traceback" or "segfault".
    try:
        ds = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction",
                          split="train")
        added = 0
        for row in ds:
            r = _norm(row, inst_key="query", resp_key="answer")
            if r:
                r = _normalize_text(r)
                if (_has_code(r["response"])
                        and _has_min_code_lines(r["response"])
                        and _is_clean(r)):
                    wrapped = _wrap_debugger_format(r["instruction"], r["response"])
                    if not wrapped:
                        continue
                    r["response"] = wrapped
                    r.update(source="CodeFeedback-Filtered-Instruction", capability="debugging")
                    samples.append(r)
                    added += 1
                    if added >= 6_000:
                        break
        print(f"  CodeFeedback-Filtered-Instruction: +{added} → {len(samples)} total")
    except Exception as e:
        print(f"  [WARN] CodeFeedback-Filtered-Instruction: {e}")

    # --- nampdn-ai/stack-exchange-instruction  (CC-BY-SA-4.0, real SO bug-fix Q&A) ---
    # Human-authored: real developers asking about real crashes, errors, and broken code.
    # Keyword-filtered to debugging domain since this is a general SO dump.
    try:
        ds = load_dataset("nampdn-ai/stack-exchange-instruction",
                          split="train")
        added = 0
        for row in ds:
            r = _norm(row, inst_key="question", resp_key="response")
            if not r:
                r = _norm(row, inst_key="instruction", resp_key="response")
            if r:
                r = _normalize_text(r)
                if (_keyword_match(r["instruction"], DEBUG_KEYWORDS)
                        and _has_code(r["response"])
                        and _has_min_code_lines(r["response"])
                        and _is_clean(r)):
                    wrapped = _wrap_debugger_format(r["instruction"], r["response"])
                    if not wrapped:
                        continue
                    r["response"] = wrapped
                    r.update(source="stack-exchange-instruction", capability="debugging")
                    samples.append(r)
                    added += 1
                    if added >= 3_000:
                        break
        print(f"  stack-exchange-instruction (debug slice): +{added} → {len(samples)} total")
    except Exception as e:
        print(f"  [WARN] stack-exchange-instruction debug: {e}")

    # --- code_instructions_122k — debug/fix subset ---
    try:
        ds = load_dataset("TokenBender/code_instructions_122k_alpaca_style",
                          split="train")
        added = 0
        for row in ds:
            r = _norm(row, inst_key="instruction", resp_key="output")
            if r:
                r = _normalize_text(r)
                if (_keyword_match(r["instruction"], DEBUG_KEYWORDS)
                        and _has_code(r["response"])
                        and _has_min_code_lines(r["response"])
                        and _is_clean(r)):
                    wrapped = _wrap_debugger_format(r["instruction"], r["response"])
                    if not wrapped:
                        continue
                    r["response"] = wrapped
                    r.update(source="code_instructions_122k", capability="debugging")
                    samples.append(r)
                    added += 1
                    if added >= 3_000:
                        break
        print(f"  code_instructions_122k (debug slice): +{added} → {len(samples)} total")
    except Exception as e:
        print(f"  [WARN] code_instructions_122k debug: {e}")

    # --- CodeAlpaca-20k — fix/debug subset ---
    try:
        ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
        added = 0
        for row in ds:
            inst = _fix_bpe_artifacts(str(row.get("instruction", "")).strip())
            inp  = _fix_bpe_artifacts(str(row.get("input", "")).strip())
            out  = _fix_bpe_artifacts(str(row.get("output", "")).strip())
            if inp and inp not in inst:
                inst = f"{inst}\n\n{inp}"
            if (len(inst) >= 20 and len(out) >= 40
                    and len(out) <= _RESPONSE_MAX_CHARS
                    and _keyword_match(inst, DEBUG_KEYWORDS)
                    and _has_code(out)
                    and _has_min_code_lines(out)
                    and not _LEAKAGE_RE.search(out)):
                wrapped = _wrap_debugger_format(inst, out)
                if not wrapped:
                    continue
                samples.append({"instruction": inst, "response": wrapped,
                                 "source": "CodeAlpaca-20k", "capability": "debugging"})
                added += 1
        print(f"  CodeAlpaca-20k (debug slice): +{added} → {len(samples)} total")
    except Exception as e:
        print(f"  [WARN] CodeAlpaca-20k debug: {e}")

    samples = _dedup(samples)
    random.shuffle(samples)          # M4: shuffle before cap
    result = samples[:target]
    print(f"[DEBUGGING] Final: {len(result)} samples")
    return result


def load_review(target: int = MAX_SAMPLES) -> List[Dict]:
    """
    Review: security audits, refactoring, OWASP, style, performance,
            code smells, accessibility, SOLID/DRY violations.
    Sources:
      - ise-uiuc/Magicoder-Evol-Instruct-110K      MIT         (refactor/review keyword slice)
      - m-a-p/CodeFeedback-Filtered-Instruction    MIT         (code feedback/review slice)
      - TokenBender/code_instructions_122k_alpaca_style Apache-2.0 (review/refactor subset)
    """
    print("\n[REVIEW] Loading datasets...")
    samples: List[Dict] = []

    # --- Magicoder-Evol-Instruct-110K  (MIT, refactor/review keyword slice) ---
    try:
        ds = load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K",
                          split="train")
        added = 0
        for row in ds:
            r = _norm(row, inst_key="instruction", resp_key="response")
            if r:
                r = _normalize_text(r)
                if _keyword_match(r["instruction"], REVIEW_KEYWORDS) and _is_clean(r):
                    wrapped = _wrap_reviewer_format(r["response"])
                    if not wrapped:    # H4: reject samples that fail format gate
                        continue
                    r["response"] = wrapped
                    r.update(source="Magicoder-Evol-Instruct-110K", capability="review")
                    samples.append(r)
                    added += 1
                    if added >= 4_000:
                        break
        print(f"  Magicoder-Evol-Instruct-110K (review slice): +{added} → {len(samples)} total")
    except Exception as e:
        print(f"  [WARN] Magicoder-Evol-Instruct-110K review: {e}")

    # --- CodeFeedback-Filtered-Instruction  (MIT, code feedback/improvement slice) ---
    # No keyword gate — CodeFeedback is already a code-quality feedback dataset.
    # REVIEW_KEYWORDS on top discards most samples because instructions phrase
    # feedback requests as "Improve this" or "What's wrong" not "review" or "refactor".
    try:
        ds = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction",
                          split="train")
        added = 0
        for row in ds:
            r = _norm(row, inst_key="query", resp_key="answer")
            if r:
                r = _normalize_text(r)
                if _is_clean(r):
                    wrapped = _wrap_reviewer_format(r["response"])
                    if not wrapped:
                        continue
                    r["response"] = wrapped
                    r.update(source="CodeFeedback-Filtered-Instruction", capability="review")
                    samples.append(r)
                    added += 1
                    if added >= 5_000:
                        break
        print(f"  CodeFeedback-Filtered-Instruction (review slice): +{added} → {len(samples)} total")
    except Exception as e:
        print(f"  [WARN] CodeFeedback-Filtered-Instruction review: {e}")

    # --- code_instructions_122k  (Apache-2.0, review/refactor subset) ---
    try:
        ds = load_dataset("TokenBender/code_instructions_122k_alpaca_style",
                          split="train")
        added = 0
        for row in ds:
            r = _norm(row, inst_key="instruction", resp_key="output")
            if r:
                r = _normalize_text(r)
                if _keyword_match(r["instruction"], REVIEW_KEYWORDS) and _is_clean(r):
                    wrapped = _wrap_reviewer_format(r["response"])
                    if not wrapped:    # H4: reject samples that fail format gate
                        continue
                    r["response"] = wrapped
                    r.update(source="code_instructions_122k", capability="review")
                    samples.append(r)
                    added += 1
                    if added >= 5_000:
                        break
        print(f"  code_instructions_122k (review slice): +{added} → {len(samples)} total")
    except Exception as e:
        print(f"  [WARN] code_instructions_122k review: {e}")

    samples = _dedup(samples)
    random.shuffle(samples)          # M4: shuffle before cap
    result = samples[:target]
    print(f"[REVIEW] Final: {len(result)} samples")
    return result


def load_planning(target: int = MAX_SAMPLES) -> List[Dict]:
    """
    Planning: system design, ADRs, sprint planning, roadmaps, architecture
              decisions, RFCs, migration plans, scalability proposals.
    Sources:
      - teknium/OpenHermes-2.5                      CC-BY-4.0  (planning/design subset)
      - WizardLM/WizardLM_evol_instruct_70k         Apache-2.0 (architecture subset)
      - m-a-p/CodeFeedback-Filtered-Instruction     MIT        (architecture/design slice)
    """
    print("\n[PLANNING] Loading datasets...")
    samples: List[Dict] = []

    # --- teknium/OpenHermes-2.5 ---
    try:
        ds = load_dataset("teknium/OpenHermes-2.5", split="train")
        added = 0
        for row in ds:
            r = _norm_chat(row) or _norm(row, inst_key="instruction", resp_key="output")
            if r:
                r = _normalize_text(r)
                if _keyword_match(r["instruction"], PLAN_KEYWORDS) and _is_clean(r):
                    r.update(source="OpenHermes-2.5", capability="planning")
                    samples.append(r)
                    added += 1
                    if added >= 5_000:
                        break
        print(f"  OpenHermes-2.5 (planning slice): +{added} → {len(samples)} total")
    except Exception as e:
        print(f"  [WARN] OpenHermes-2.5: {e}")

    # --- WizardLM/WizardLM_evol_instruct_70k ---
    try:
        ds = load_dataset("WizardLM/WizardLM_evol_instruct_70k",
                          split="train")
        added = 0
        for row in ds:
            r = _norm(row, inst_key="instruction", resp_key="output")
            if r:
                r = _normalize_text(r)
                if _keyword_match(r["instruction"], PLAN_KEYWORDS) and _is_clean(r):
                    r.update(source="WizardLM_evol_instruct_70k", capability="planning")
                    samples.append(r)
                    added += 1
                    if added >= 5_000:
                        break
        print(f"  WizardLM_evol_instruct_70k (planning slice): +{added} → {len(samples)} total")
    except Exception as e:
        print(f"  [WARN] WizardLM_evol_instruct_70k: {e}")

    # --- CodeFeedback-Filtered-Instruction  (MIT, architecture/design tasks) ---
    try:
        ds = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction",
                          split="train")
        added = 0
        for row in ds:
            r = _norm(row, inst_key="query", resp_key="answer")
            if r:
                r = _normalize_text(r)
                if _keyword_match(r["instruction"], PLAN_KEYWORDS) and _is_clean(r):
                    r.update(source="CodeFeedback-Filtered-Instruction", capability="planning")
                    samples.append(r)
                    added += 1
                    if added >= 3_000:
                        break
        print(f"  CodeFeedback-Filtered-Instruction (planning slice): +{added} → {len(samples)} total")
    except Exception as e:
        print(f"  [WARN] CodeFeedback-Filtered-Instruction planning: {e}")

    samples = _dedup(samples)
    random.shuffle(samples)          # M4: shuffle before cap
    result = samples[:target]
    print(f"[PLANNING] Final: {len(result)} samples")
    return result


def load_research(target: int = MAX_SAMPLES) -> List[Dict]:
    """
    Research: technology comparisons, deep-dives, QA, literature summaries,
              benchmark analysis, tradeoff explanations.
    Sources:
      - HuggingFaceH4/ultrachat_200k  CC-BY-4.0  (long-form QA/summarisation subset)
      - teknium/OpenHermes-2.5        CC-BY-4.0  (research/explain subset)
      - tatsu-lab/alpaca               Apache-2.0 (explain/compare supplement)
    """
    print("\n[RESEARCH] Loading datasets...")
    samples: List[Dict] = []

    # --- HuggingFaceH4/ultrachat_200k ---
    try:
        ds = load_dataset("HuggingFaceH4/ultrachat_200k",
                          split="train_sft")
        added = 0
        for row in ds:
            r = _norm_chat(row)
            if r:
                r = _normalize_text(r)
                if _keyword_match(r["instruction"], RESEARCH_KEYWORDS) and _is_clean(r):
                    r.update(source="ultrachat_200k", capability="research")
                    samples.append(r)
                    added += 1
                    if added >= 5_000:
                        break
        print(f"  ultrachat_200k (research slice): +{added} → {len(samples)} total")
    except Exception as e:
        print(f"  [WARN] ultrachat_200k: {e}")

    # --- OpenHermes-2.5 research/explain subset ---
    try:
        ds = load_dataset("teknium/OpenHermes-2.5", split="train")
        added = 0
        for row in ds:
            r = _norm_chat(row) or _norm(row, inst_key="instruction", resp_key="output")
            if r:
                r = _normalize_text(r)
                if _keyword_match(r["instruction"], RESEARCH_KEYWORDS) and _is_clean(r):
                    r.update(source="OpenHermes-2.5", capability="research")
                    samples.append(r)
                    added += 1
                    if added >= 5_000:
                        break
        print(f"  OpenHermes-2.5 (research slice): +{added} → {len(samples)} total")
    except Exception as e:
        print(f"  [WARN] OpenHermes-2.5 research: {e}")

    # --- tatsu-lab/alpaca  (Apache-2.0, explain/compare supplement) ---
    try:
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        added = 0
        for row in ds:
            r = _norm(row, inst_key="instruction", resp_key="output")
            if r:
                r = _normalize_text(r)
                if _keyword_match(r["instruction"], RESEARCH_KEYWORDS) and _is_clean(r):
                    r.update(source="alpaca", capability="research")
                    samples.append(r)
                    added += 1
                    if added >= 3_000:
                        break
        print(f"  alpaca (research slice): +{added} → {len(samples)} total")
    except Exception as e:
        print(f"  [WARN] alpaca research: {e}")

    samples = _dedup(samples)
    random.shuffle(samples)          # M4: shuffle before cap
    result = samples[:target]
    print(f"[RESEARCH] Final: {len(result)} samples")
    return result


def load_criticism(target: int = MAX_SAMPLES) -> List[Dict]:
    """
    Criticism: decision critique, risk identification, assumption challenging,
               devil's advocate reasoning, design flaw analysis, evaluation.
    Sources:
      - Anthropic/hh-rlhf                   MIT        (chosen=high-quality critique responses)
      - teknium/OpenHermes-2.5              CC-BY-4.0  (critique/evaluation subset)
      - tatsu-lab/alpaca                    Apache-2.0 (evaluation/opinion supplement)
      - WizardLM/WizardLM_evol_instruct_70k Apache-2.0 (fallback)
    Note: CRITIC_KEYWORDS are intentionally broad to avoid sparse matching.
    """
    print("\n[CRITICISM] Loading datasets...")
    samples: List[Dict] = []

    # --- Anthropic/hh-rlhf  (MIT — extract last Human/Assistant turn from chosen) ---
    try:
        ds = load_dataset("Anthropic/hh-rlhf", split="train")
        added = 0
        for row in ds:
            chosen = str(row.get("chosen", "")).strip()
            # H5: Include the full conversation context as the instruction, not just
            # the last Human turn.  The critique often only makes sense with context.
            # We reconstruct all prior turns as a quoted conversation prefix and
            # append the final human question so the model sees the full picture.
            if "\n\nHuman:" in chosen and "\n\nAssistant:" in chosen:
                # Split off the final Assistant response
                pre_resp, resp = chosen.rsplit("\n\nAssistant:", 1)
                resp = _fix_bpe_artifacts(resp.strip())

                # Split all prior turns to build context prefix
                turn_texts = re.split(r"\n\n(Human|Assistant):", pre_resp)
                # turn_texts alternates: [lead, role, content, role, content, ...]
                context_lines: list[str] = []
                i = 1
                last_human = ""
                while i + 1 < len(turn_texts):
                    role_name = turn_texts[i].strip()
                    content   = _fix_bpe_artifacts(turn_texts[i + 1].strip())
                    if role_name == "Human":
                        last_human = content
                        if context_lines:         # prior turns → add as context
                            context_lines.append(f"Human: {content}")
                        else:
                            context_lines.append(f"Human: {content}")
                    elif role_name == "Assistant" and content:
                        context_lines.append(f"Assistant: {content}")
                    i += 2

                if len(context_lines) > 1:
                    # Multi-turn: prefix with abbreviated context
                    inst = "\n".join(context_lines[:-1]) + "\n\n" + context_lines[-1]
                else:
                    inst = last_human

                if (len(inst) >= 20 and len(resp) >= 40
                        and _keyword_match(inst, CRITIC_KEYWORDS)
                        and _is_clean({"instruction": inst, "response": resp})):
                    wrapped = _wrap_critic_format(resp)
                    if not wrapped:    # H4: reject samples that fail format gate
                        continue
                    samples.append({"instruction": inst, "response": wrapped,
                                    "source": "hh-rlhf", "capability": "criticism"})
                    added += 1
                    if added >= 3_000:
                        break
        print(f"  hh-rlhf (criticism slice): +{added} → {len(samples)} total")
    except Exception as e:
        print(f"  [WARN] hh-rlhf: {e}")

    # --- OpenHermes-2.5 critique/evaluation subset ---
    try:
        ds = load_dataset("teknium/OpenHermes-2.5", split="train")
        added = 0
        for row in ds:
            r = _norm_chat(row) or _norm(row, inst_key="instruction", resp_key="output")
            if r:
                r = _normalize_text(r)
                if _keyword_match(r["instruction"], CRITIC_KEYWORDS) and _is_clean(r):
                    wrapped = _wrap_critic_format(r["response"])
                    if not wrapped:    # H4: reject samples that fail format gate
                        continue
                    r["response"] = wrapped
                    r.update(source="OpenHermes-2.5", capability="criticism")
                    samples.append(r)
                    added += 1
                    if added >= 5_000:
                        break
        print(f"  OpenHermes-2.5 (criticism slice): +{added} → {len(samples)} total")
    except Exception as e:
        print(f"  [WARN] OpenHermes-2.5 criticism: {e}")

    # --- tatsu-lab/alpaca  (Apache-2.0, evaluation/opinion supplement) ---
    try:
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        added = 0
        for row in ds:
            r = _norm(row, inst_key="instruction", resp_key="output")
            if r:
                r = _normalize_text(r)
                if _keyword_match(r["instruction"], CRITIC_KEYWORDS) and _is_clean(r):
                    wrapped = _wrap_critic_format(r["response"])
                    if not wrapped:    # H4: reject samples that fail format gate
                        continue
                    r["response"] = wrapped
                    r.update(source="alpaca", capability="criticism")
                    samples.append(r)
                    added += 1
                    if added >= 3_000:
                        break
        print(f"  alpaca (criticism slice): +{added} → {len(samples)} total")
    except Exception as e:
        print(f"  [WARN] alpaca criticism: {e}")

    # --- WizardLM fallback if still below minimum ---
    if len(samples) < MIN_SAMPLES:
        try:
            ds = load_dataset("WizardLM/WizardLM_evol_instruct_70k",
                              split="train")
            added = 0
            for row in ds:
                r = _norm(row, inst_key="instruction", resp_key="output")
                if r:
                    r = _normalize_text(r)
                    if _keyword_match(r["instruction"], CRITIC_KEYWORDS) and _is_clean(r):
                        wrapped = _wrap_critic_format(r["response"])
                        if not wrapped:
                            continue
                        r["response"] = wrapped
                        r.update(source="WizardLM_evol_instruct_70k", capability="criticism")
                        samples.append(r)
                        added += 1
                        if len(samples) >= target:
                            break
            print(f"  WizardLM fallback (criticism): +{added} → {len(samples)} total")
        except Exception as e:
            print(f"  [WARN] WizardLM criticism fallback: {e}")

    samples = _dedup(samples)
    random.shuffle(samples)          # M4: shuffle before cap
    result = samples[:target]
    print(f"[CRITICISM] Final: {len(result)} samples")
    return result


# ---------------------------------------------------------------------------
# Save + stats helpers
# ---------------------------------------------------------------------------

def save_jsonl(data: List[Dict], capability: str) -> Path:
    out = TARGET_DIR / f"{capability}_dataset.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(_wrap_assistant_only_loss_format(item), ensure_ascii=False) + "\n")
    size_mb = out.stat().st_size / 1e6
    print(f"  → Saved {len(data):,} samples to {out}  ({size_mb:.1f} MB)")

    # M3: Post-write validation — re-read and verify every line parses and has
    # the required keys.  Catches filesystem truncation or encoding errors.
    bad_lines = 0
    with open(out, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                bad_lines += 1
                continue
            try:
                rec = json.loads(line)
                if not rec.get("instruction") or not rec.get("response"):
                    bad_lines += 1
            except json.JSONDecodeError:
                bad_lines += 1
    if bad_lines:
        print(f"  [WARN] Post-write validation: {bad_lines} invalid/empty lines in {out.name}")
    else:
        print(f"  [OK] Post-write validation passed ({len(data):,} lines)")
    return out


def print_stats(capability: str, data: List[Dict]) -> None:
    sources: Dict[str, int] = {}
    lengths = [len(d["instruction"]) + len(d["response"]) for d in data]
    for d in data:
        s = d.get("source", "unknown")
        sources[s] = sources.get(s, 0) + 1
    print(f"\n  [{capability.upper()}] {len(data):,} samples")
    for src, cnt in sorted(sources.items(), key=lambda x: -x[1]):
        pct = cnt / len(data) * 100
        print(f"    {src}: {cnt:,} ({pct:.1f}%)")
    if lengths:
        avg = sum(lengths) / len(lengths)
        print(f"    avg text length: {avg:.0f} chars")


# ---------------------------------------------------------------------------
# Loader registry
# ---------------------------------------------------------------------------
LOADERS = {
    "coding":    load_coding,
    "debugging": load_debugging,
    "review":    load_review,
    "planning":  load_planning,
    "research":  load_research,
    "criticism": load_criticism,
}


# ---------------------------------------------------------------------------
# Public API — called by train_gpu_simple.py
# ---------------------------------------------------------------------------

def generate_datasets(capabilities: List[str] = None,
                      target: int = MAX_SAMPLES) -> Dict[str, Path]:
    """
    Generate JSONL datasets for the requested capabilities.

    Args:
        capabilities: list of capability names, or None for all six.
        target: fallback max samples when no per-capability override exists.
                Per-capability overrides in _CAP_MAX take precedence.

    Returns:
        dict mapping capability name → output JSONL path.
    """
    caps = capabilities or list(LOADERS.keys())

    print("\n" + "="*70)
    print("  AGENT TRAINING DATA GENERATION")
    print(f"  Capabilities : {caps}")
    print(f"  Target range : {MIN_SAMPLES:,} – {target:,} samples (base; coding/debugging use higher caps)")
    print(f"  Output dir   : {TARGET_DIR.resolve()}")
    print("="*70)

    output_paths: Dict[str, Path] = {}

    for cap in caps:
        if cap not in LOADERS:
            print(f"[WARN] Unknown capability '{cap}' — skipping")
            continue
        cap_target = _CAP_MAX.get(cap, target)
        data = LOADERS[cap](target=cap_target)
        if len(data) < MIN_SAMPLES:
            print(f"  [WARN] {cap}: only {len(data):,} samples "
                  f"(target {MIN_SAMPLES:,}). Training will still proceed.")
        print_stats(cap, data)
        path = save_jsonl(data, cap)
        output_paths[cap] = path

    print("\n" + "="*70)
    print("  DATA GENERATION COMPLETE")
    print("="*70)
    for cap, path in output_paths.items():
        n = sum(1 for _ in open(path, encoding="utf-8"))
        print(f"  {cap:12s}  {n:>7,} samples  {path}")
    print()
    return output_paths


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate agent training datasets from public HF sources")
    parser.add_argument(
        "--capabilities", nargs="*", default=None,
        metavar="CAP",
        help="Which capabilities to generate (default: all). "
             "Choices: " + ", ".join(LOADERS.keys()))
    parser.add_argument(
        "--target", type=int, default=MAX_SAMPLES,
        help=f"Max samples per capability (default: {MAX_SAMPLES})")
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing JSONL files (default: skip if file already exists)")
    args = parser.parse_args()

    caps = args.capabilities or list(LOADERS.keys())
    invalid = [c for c in caps if c not in LOADERS]
    if invalid:
        print(f"[ERROR] Unknown capabilities: {invalid}. Valid: {list(LOADERS.keys())}")
        sys.exit(1)

    if not args.force:
        skip = [c for c in caps if (TARGET_DIR / f"{c}_dataset.jsonl").exists()]
        if skip:
            print(f"[INFO] Skipping already-generated: {skip}  (use --force to overwrite)")
        caps = [c for c in caps if c not in skip]
        if not caps:
            print("[INFO] Nothing to generate. Use --force to regenerate.")
            sys.exit(0)

    generate_datasets(capabilities=caps, target=args.target)
