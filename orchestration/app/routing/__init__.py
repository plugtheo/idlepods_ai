"""Routing utilities — orchestration-side factory."""

from __future__ import annotations

from typing import Union

from shared.routing.query_router import RegexQueryRouter
from shared.routing.llm_router import HybridQueryRouter, LLMQueryRouter

from ..clients.inference import get_inference_client
from ..config.settings import settings


# Async-capable router union — RegexQueryRouter.route is sync but the orchestration
# code path always awaits, so we wrap regex in a tiny async shim below.
_RouterLike = Union[RegexQueryRouter, LLMQueryRouter, HybridQueryRouter]

_router_instance: _RouterLike | None = None


class _AsyncRegexAdapter:
    """Async shim around RegexQueryRouter so the call site can always ``await``."""

    def __init__(self) -> None:
        self._inner = RegexQueryRouter()

    async def route(self, prompt: str):
        return self._inner.route(prompt)


def _build_router() -> _RouterLike:
    mode = settings.router_mode
    if mode == "regex":
        return _AsyncRegexAdapter()  # type: ignore[return-value]
    llm = LLMQueryRouter(
        inference_client=get_inference_client(),
        backend=settings.router_backend,
        max_tokens=settings.router_max_tokens,
        temperature=settings.router_temperature,
        cache_size=settings.router_cache_size,
    )
    if mode == "llm":
        return llm
    if mode == "hybrid":
        return HybridQueryRouter(
            llm_router=llm,
            confidence_threshold=settings.router_confidence_threshold,
        )
    raise ValueError(
        f"Unknown router_mode={mode!r}; expected 'regex', 'llm', or 'hybrid'"
    )


def get_query_router() -> _RouterLike:
    """Return the singleton router for this process, building on first call."""
    global _router_instance
    if _router_instance is None:
        _router_instance = _build_router()
    return _router_instance


def reset_query_router() -> None:
    """Test hook: clear the singleton so the next call re-reads settings."""
    global _router_instance
    _router_instance = None
