"""
QueryRouter — shared implementation
=====================================
Stateless prompt classifier.  Determines intent + complexity from the raw
prompt text using regex patterns and word-count heuristics, then maps the
pair to an agent_chain list.

No model calls, no I/O.  Sub-millisecond on any hardware.

Canonical location
------------------
This module is the single source of truth for routing logic.  Both the
Gateway Service and the Orchestration Service import from here so that
routing behaviour is identical regardless of which service a request enters.

    from shared.routing.query_router import QueryRouter, RouteDecision
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List


class Intent(str, Enum):
    CODING    = "coding"
    DEBUGGING = "debugging"
    RESEARCH  = "research"
    ANALYSIS  = "analysis"
    PLANNING  = "planning"
    QA        = "qa"
    GENERAL   = "general"


class Complexity(str, Enum):
    SIMPLE   = "simple"
    MODERATE = "moderate"
    COMPLEX  = "complex"


@dataclass
class RouteDecision:
    intent: str
    complexity: str
    agent_chain: List[str]
    matched_keywords: List[str] = field(default_factory=list)


# ── Intent patterns — ALL matching patterns are scored; highest-count wins.
# First-match is gone: "implement a fix for the traceback" now correctly
# scores CODING (implement=1) vs DEBUGGING (fix=1) and picks whichever
# appears more, falling back to DEBUGGING since it's listed first on ties.

_INTENT_PATTERNS = [
    (Intent.DEBUGGING, re.compile(r"\b(debug|fix|bug|error|traceback|exception|stacktrace|broken)\b", re.I)),
    (Intent.CODING,    re.compile(r"\b(implement|code|write|create|build|function|class|script|program)\b", re.I)),
    (Intent.RESEARCH,  re.compile(r"\b(research|investigate|survey|find|what is|explain|describe|overview)\b", re.I)),
    (Intent.ANALYSIS,  re.compile(r"\b(analyse|analyze|review|evaluate|assess|compare|critique)\b", re.I)),
    (Intent.PLANNING,  re.compile(r"\b(plan|design|architect|roadmap|structure|outline|breakdown)\b", re.I)),
    (Intent.QA,        re.compile(r"\b(how|why|when|where|who|which|can you|is it|does it)\b", re.I)),
]

# ── Complexity markers — COMPLEX is checked before SIMPLE so that a prompt
# containing both (e.g. "a simple comprehensive system") resolves to COMPLEX.
# This is intentional: complexity keywords like "comprehensive" or "enterprise"
# signal scope; simplicity keywords like "quick" or "simple" are often hedges
# that don't override the actual work required.

_COMPLEXITY_MARKERS = {
    Complexity.COMPLEX: re.compile(
        r"\b(comprehensive|production.ready|end.to.end|enterprise|scalable|distributed|full.stack|"
        r"authentication|authorization|system|microservice|multi.step|pipeline|integrate|"
        r"complete|entire|all modules|every|multiple)\b",
        re.I,
    ),
    Complexity.SIMPLE: re.compile(
        r"\b(quick|simple|brief|snippet|short|basic|minimal|single|one.liner|example|just)\b",
        re.I,
    ),
}

_SIMPLE_WORD_THRESHOLD  = 25
_COMPLEX_WORD_THRESHOLD = 80

# ── Agent chain lookup table ────────────────────────────────────────────────
#
# Design rules:
#   CODING    — coder/debugger/reviewer are the core agents; researcher skipped
#               (no retrieval backing, just training-weight guesses);
#               review_critic runs reviewer then critic sequentially in one step.
#   DEBUGGING — debugger leads; coder handles the actual fix on moderate/complex.
#   RESEARCH  — researcher is appropriate (output IS prose); reviewer excluded
#               (code-focused persona adds nothing to prose chains).
#   ANALYSIS  — reviewer is the core agent; critic adds a quality gate.
#   PLANNING  — planner leads; reviewer excluded (code-focused).
#   QA        — researcher answers from knowledge; planner structures complex answers.
#   GENERAL   — true fallback (no intent keyword matched); no code assumed;
#               conservative prose-only chains.

_CHAINS = {
    # ── CODING ──────────────────────────────────────────────────────────────
    (Intent.CODING,    Complexity.SIMPLE):   ["coder", "reviewer"],
    (Intent.CODING,    Complexity.MODERATE): ["planner", "coder", "review_critic"],
    (Intent.CODING,    Complexity.COMPLEX):  ["planner", "coder", "debugger", "review_critic"],

    # ── DEBUGGING ───────────────────────────────────────────────────────────
    (Intent.DEBUGGING, Complexity.SIMPLE):   ["debugger", "reviewer"],
    (Intent.DEBUGGING, Complexity.MODERATE): ["debugger", "coder", "reviewer"],
    (Intent.DEBUGGING, Complexity.COMPLEX):  ["planner", "debugger", "coder", "review_critic"],

    # ── RESEARCH ────────────────────────────────────────────────────────────
    (Intent.RESEARCH,  Complexity.SIMPLE):   ["researcher"],
    (Intent.RESEARCH,  Complexity.MODERATE): ["researcher", "critic"],
    (Intent.RESEARCH,  Complexity.COMPLEX):  ["planner", "researcher", "critic"],

    # ── ANALYSIS ────────────────────────────────────────────────────────────
    (Intent.ANALYSIS,  Complexity.SIMPLE):   ["reviewer"],
    (Intent.ANALYSIS,  Complexity.MODERATE): ["review_critic"],
    (Intent.ANALYSIS,  Complexity.COMPLEX):  ["planner", "researcher", "review_critic"],

    # ── PLANNING ────────────────────────────────────────────────────────────
    (Intent.PLANNING,  Complexity.SIMPLE):   ["planner"],
    (Intent.PLANNING,  Complexity.MODERATE): ["planner", "critic"],
    (Intent.PLANNING,  Complexity.COMPLEX):  ["planner", "researcher", "critic"],

    # ── QA ──────────────────────────────────────────────────────────────────
    (Intent.QA,        Complexity.SIMPLE):   ["researcher"],
    (Intent.QA,        Complexity.MODERATE): ["researcher", "critic"],
    (Intent.QA,        Complexity.COMPLEX):  ["planner", "researcher", "critic"],

    # ── GENERAL (fallback — no keyword matched) ──────────────────────────────
    (Intent.GENERAL,   Complexity.SIMPLE):   ["planner"],
    (Intent.GENERAL,   Complexity.MODERATE): ["planner", "researcher"],
    (Intent.GENERAL,   Complexity.COMPLEX):  ["planner", "researcher", "critic"],
}


class QueryRouter:
    """Stateless prompt classifier.  No I/O, no model calls."""

    def route(self, prompt: str) -> RouteDecision:
        intent, keywords = self._classify_intent(prompt)
        complexity = self._classify_complexity(prompt)
        chain = _CHAINS.get((intent, complexity), ["planner"])
        return RouteDecision(
            intent=intent.value,
            complexity=complexity.value,
            agent_chain=chain,
            matched_keywords=keywords,
        )

    def _classify_intent(self, prompt: str):
        # Score ALL patterns — highest keyword count wins, not first match.
        # On a tie the pattern listed earlier in _INTENT_PATTERNS wins (stable).
        scores: dict = {}
        all_keywords: dict = {}
        for intent, pattern in _INTENT_PATTERNS:
            matches = pattern.findall(prompt)
            if matches:
                if intent not in scores or len(matches) > scores[intent]:
                    scores[intent] = len(matches)
                    all_keywords[intent] = [m.lower() for m in matches[:3]]
        if not scores:
            return Intent.GENERAL, []
        best = max(scores, key=scores.__getitem__)
        return best, all_keywords[best]

    def _classify_complexity(self, prompt: str) -> Complexity:
        for complexity, pattern in _COMPLEXITY_MARKERS.items():
            if pattern.search(prompt):
                return complexity
        word_count = len(prompt.split())
        if word_count < _SIMPLE_WORD_THRESHOLD:
            return Complexity.SIMPLE
        if word_count > _COMPLEX_WORD_THRESHOLD:
            return Complexity.COMPLEX
        return Complexity.MODERATE
