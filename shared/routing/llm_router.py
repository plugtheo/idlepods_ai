"""
LLM-backed query router
========================
Replaces the regex classifier with a cheap LLM call constrained by vLLM
guided JSON.  Used by the orchestration service when
``settings.router_mode in ("llm", "hybrid")``.

Design:
  - LLMQueryRouter calls a single inference request, expects JSON matching
    RouteClassification, and maps (intent, complexity) → agent_chain via the
    same lookup table the regex router uses (_resolve_chain).
  - HybridQueryRouter runs RegexQueryRouter first; if confidence is below
    ``confidence_threshold`` it upgrades to the LLM call.  Most prompts that
    have clear intent keywords never pay the LLM cost.
  - In-process LRU cache (``hash(prompt)``) deduplicates repeated routing of
    the exact same prompt within a single process.
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from dataclasses import asdict
from typing import Optional, Protocol

from shared.contracts.agent_prompts import AGENT_PROMPTS
from shared.contracts.inference import GenerateRequest, GenerateResponse, Message
from shared.contracts.routing import RouteClassification

from .query_router import RegexQueryRouter, RouteDecision, _resolve_chain

logger = logging.getLogger(__name__)


class _InferenceClientProto(Protocol):
    async def generate(self, request: GenerateRequest) -> GenerateResponse: ...


class LLMQueryRouter:
    """Router that classifies intent + complexity by calling the inference service."""

    def __init__(
        self,
        inference_client: _InferenceClientProto,
        backend: str,
        max_tokens: int = 64,
        temperature: float = 0.0,
        cache_size: int = 256,
    ) -> None:
        self._client = inference_client
        self._backend = backend
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._cache: "OrderedDict[str, RouteDecision]" = OrderedDict()
        self._cache_size = cache_size
        self._schema = RouteClassification.model_json_schema()

    async def route(self, prompt: str) -> RouteDecision:
        cached = self._cache.get(prompt)
        if cached is not None:
            self._cache.move_to_end(prompt)
            return cached

        request = GenerateRequest(
            backend=self._backend,
            role="router",
            messages=[
                Message(role="system", content=AGENT_PROMPTS["router"]),
                Message(role="user",   content=prompt),
            ],
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            response_schema=self._schema,
        )

        try:
            response = await self._client.generate(request)
            classification = self._parse_response(response)
        except Exception as exc:
            logger.warning("LLMQueryRouter classify failed: %s — falling back to regex", exc)
            return RegexQueryRouter().route(prompt)

        chain = _resolve_chain(classification.intent, classification.complexity)
        decision = RouteDecision(
            intent=classification.intent,
            complexity=classification.complexity,
            agent_chain=chain,
            matched_keywords=[],
            confidence=classification.confidence,
            source="llm",
        )
        self._remember(prompt, decision)
        return decision

    def _parse_response(self, response: GenerateResponse) -> RouteClassification:
        """Prefer pre-parsed payload from guided decoding; fall back to JSON-in-content."""
        if response.parsed is not None:
            return RouteClassification.model_validate(response.parsed)
        return RouteClassification.model_validate(json.loads(response.content))

    def _remember(self, prompt: str, decision: RouteDecision) -> None:
        self._cache[prompt] = decision
        self._cache.move_to_end(prompt)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)


class HybridQueryRouter:
    """Run regex first; upgrade to LLM only when regex confidence is below threshold."""

    def __init__(
        self,
        llm_router: LLMQueryRouter,
        confidence_threshold: float = 0.6,
    ) -> None:
        self._regex = RegexQueryRouter()
        self._llm = llm_router
        self._threshold = confidence_threshold

    async def route(self, prompt: str) -> RouteDecision:
        regex_decision = self._regex.route(prompt)
        if regex_decision.confidence >= self._threshold:
            logger.debug(
                "HybridQueryRouter: accepted regex decision %s (confidence=%.2f)",
                regex_decision.intent, regex_decision.confidence,
            )
            return RouteDecision(**{**asdict(regex_decision), "source": "hybrid_regex"})
        logger.debug(
            "HybridQueryRouter: regex low-confidence (%.2f) — calling LLM",
            regex_decision.confidence,
        )
        llm_decision = await self._llm.route(prompt)
        return RouteDecision(**{**asdict(llm_decision), "source": "hybrid_llm"})
