"""
LocalVLLMBackend
================
Calls a locally running vLLM server via its OpenAI-compatible
`/v1/chat/completions` endpoint.

One vLLM server process handles one model family (deepseek or mistral).
The backend maps `model_family` → server URL, then POSTs the standard
`messages` array.

Adapter handling
----------------
If an `adapter_name` is provided the backend first checks whether that
adapter is registered with the vLLM server by querying `GET /v1/models`.
If the adapter is available it is passed as the `model` field using vLLM's
`{base_model_id}/{adapter_name}` convention, activating the LoRA adapter
for that request.

If the adapter is NOT yet registered (e.g. Training Service hasn't produced
it yet, or the server just started), the request falls back to the base
model silently — a warning is logged but inference still succeeds.

The available-adapter list is cached per server and refreshed every
ADAPTER_CACHE_TTL_SECONDS (default 120 s) so newly trained adapters are
picked up automatically without a restart.
"""

from __future__ import annotations

import json
import logging
import time
from typing import AsyncGenerator, Optional

import httpx

from shared.contracts.inference import GenerateRequest, GenerateResponse
from ..config.settings import settings
from .base import InferenceBackend

# ---------------------------------------------------------------------------
# GPT-2 / Byte-level BPE artifact correction
# ---------------------------------------------------------------------------
# Unsloth-trained LoRA adapters can cause vLLM to return raw BPE byte-level
# unicode placeholders instead of their ASCII equivalents. In GPT-2 style BPE,
# bytes 0-32, 127-160, and 173 are mapped to Unicode starting at U+0100.
# The two most impactful:
#   Ġ (U+0120) = byte 32 = ASCII space
#   Ċ (U+010A) = byte 10 = ASCII newline
#   ĉ (U+0109) = byte  9 = ASCII tab
#   č (U+010D) = byte 13 = ASCII carriage return
#
# Build the full inverse table once at import time so the hot path is just
# a str.translate() call with no per-call computation.
def _build_bpe_decode_table() -> dict[int, int]:
    """Return a str.translate() table: BPE-unicode codepoint → original byte."""
    # Bytes that map to themselves (printable Latin-1 range)
    passthrough = (
        list(range(ord("!"), ord("~") + 1))       # 33–126
        + list(range(ord("¡"), ord("¬") + 1))     # 161–172
        + list(range(ord("®"), ord("ÿ") + 1))     # 174–255
    )
    passthrough_set = set(passthrough)
    table: dict[int, int] = {}
    n = 0
    for b in range(256):
        if b in passthrough_set:
            continue  # mapped to itself — no translation needed
        unicode_cp = 256 + n  # U+0100, U+0101, …
        table[unicode_cp] = b
        n += 1
    return table


_BPE_DECODE_TABLE: dict[int, int] = _build_bpe_decode_table()


def _fix_bpe_artifacts(text: str) -> str:
    """
    Replace GPT-2 byte-level BPE unicode placeholders with their original
    ASCII/Latin-1 bytes, then re-decode as UTF-8.

    This is a no-op for text that does not contain BPE placeholder characters
    (U+0100–U+0143 range), so it is safe to apply unconditionally.
    """
    # Fast path: skip the heavier processing when no BPE chars are present.
    # All BPE placeholders live in U+0100–U+0143; check for any char in that range.
    if not any("\u0100" <= ch <= "\u0143" for ch in text):
        return text
    try:
        raw = bytes(
            _BPE_DECODE_TABLE[ord(ch)] if ord(ch) in _BPE_DECODE_TABLE else ord(ch)
            for ch in text
        )
        return raw.decode("utf-8", errors="replace")
    except Exception:
        return text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Training-format chat template
# ---------------------------------------------------------------------------
# Adapters (coding_lora, debugging_lora, planning_lora, …) were trained with
# the following plain-text format (see lora_trainer.py / trainer_entry.py):
#
#   [SYSTEM]
#   <system message content>
#
#   [USER]
#   <user message content>
#
#   [RESPONSE]
#   <response>
#
# For adapter inference we build this string via string concatenation and call
# /v1/completions.  The Jinja2 chat_template approach was ABANDONED because
# {%- -%} whitespace stripping silently removes the \n\n separators between
# sections, producing [SYSTEM]\nS[USER]\nU[RESPONSE] instead of the correct
# [SYSTEM]\nS\n\n[USER]\nU\n\n[RESPONSE]\n — a training/inference mismatch.
#
# Base-model calls (no adapter) continue to use /v1/chat/completions with
# the model's native template, which is correct for non-fine-tuned inference.
#
# Note: adapters are NOT language-specific — coding_lora, debugging_lora, etc.
# cover all languages (Python, JS/TS, Go, Rust, Java, C#, SQL, shell …).
# The system prompt injected by the orchestration layer establishes the
# capability context; the adapter supplies the fine-tuned response style.

def _build_adapter_prompt(messages: list) -> str:
    """
    Assemble the canonical adapter prompt string from a messages list.

    Output (for system + user turn):
        [SYSTEM]
        <system content>

        [USER]
        <user content>

        [RESPONSE]

    This is the EXACT format written to every training pair by load_sft_pairs()
    in train_gpu_simple.py and by _format_messages_as_prompt() in trainer_entry.py.
    DataCollatorForCompletionOnlyLM masks everything before "[RESPONSE]\\n",
    so this boundary must be byte-for-byte identical at training and inference.
    """
    parts: list[str] = []
    for msg in messages:
        role    = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            parts.append(f"[SYSTEM]\n{content}")
        elif role == "user":
            parts.append(f"[USER]\n{content}")
        elif role == "assistant":
            parts.append(f"[ASSISTANT]\n{content}")
    return "\n\n".join(parts) + "\n\n[RESPONSE]\n"


# Stop tokens prevent the model from hallucinating the next training pair.
_ADAPTER_STOP_TOKENS = ["[SYSTEM]", "[USER]", "[ASSISTANT]", "\n[RESPONSE]"]


_FAMILY_MAP: dict[str, tuple[str, str]] = {
    "deepseek": (settings.deepseek_url, settings.deepseek_model_id),
    "mistral":  (settings.mistral_url,  settings.mistral_model_id),
}

# How often to re-query /v1/models to discover newly trained adapters
ADAPTER_CACHE_TTL_SECONDS = 120


class _AdapterRegistry:
    """
    Per-vLLM-server cache of registered model IDs (base + all LoRA adapters).

    Queries GET /v1/models on first use and refreshes after TTL expires.
    Thread/coroutine safe for read-heavy workloads — worst case two
    concurrent refreshes race, both produce the same result.
    """

    def __init__(self, base_url: str, base_model_id: str, ttl: float = ADAPTER_CACHE_TTL_SECONDS) -> None:
        self._base_url = base_url
        self._base_model_id = base_model_id
        self._ttl = ttl
        self._known: set[str] = set()
        self._fetched_at: float = 0.0
        self._timeout = settings.request_timeout_seconds
        self._client = httpx.AsyncClient(
            timeout=self._timeout,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )

    async def adapter_available(self, adapter_name: str) -> bool:
        """Return True if adapter_name is registered with vLLM.

        Matches both the bare adapter name (e.g. 'coding_lora') and the
        qualified form vLLM returns in /v1/models (e.g.
        'deepseek-ai/deepseek-coder-6.7b-instruct/coding_lora').
        """
        await self._refresh_if_stale()
        return adapter_name in self._known or any(
            m.endswith(f"/{adapter_name}") for m in self._known
        )

    async def _refresh_if_stale(self) -> None:
        if time.monotonic() - self._fetched_at < self._ttl:
            return
        try:
            resp = await self._client.get(f"{self._base_url}/v1/models")
            resp.raise_for_status()
            ids = {m["id"] for m in resp.json().get("data", [])}
            self._known = ids
            self._fetched_at = time.monotonic()
            logger.debug(
                "AdapterRegistry refreshed for %s: %d models known",
                self._base_url, len(ids),
            )
        except Exception as exc:
            # Non-fatal: keep the stale cache so inference can still proceed
            logger.warning(
                "AdapterRegistry refresh failed for %s: %s — using stale cache",
                self._base_url, exc,
            )


# One registry per vLLM server, created at module load
_REGISTRIES: dict[str, _AdapterRegistry] = {
    family: _AdapterRegistry(base_url, model_id)
    for family, (base_url, model_id) in _FAMILY_MAP.items()
}


class LocalVLLMBackend(InferenceBackend):
    """Routes generation requests to the appropriate local vLLM server."""

    def __init__(self) -> None:
        self._timeout = settings.request_timeout_seconds
        # Persistent connection pool — eliminates per-call TCP setup overhead.
        self._client = httpx.AsyncClient(
            timeout=self._timeout,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        family = request.model_family.lower()
        if family not in _FAMILY_MAP:
            raise ValueError(
                f"Unknown model_family '{family}'. Supported: {list(_FAMILY_MAP)}"
            )

        base_url, model_id = _FAMILY_MAP[family]
        registry = _REGISTRIES[family]

        # Resolve effective model: prefer adapter if available, fall back to base
        effective_model = await _resolve_model(
            model_id, request.adapter_name, registry, request.role
        )

        messages = [m.model_dump() for m in request.messages]
        using_adapter = effective_model != model_id

        logger.debug(
            "LocalVLLM → %s  model=%s  role=%s  msgs=%d  adapter=%s",
            base_url, effective_model, request.role, len(messages), using_adapter,
        )

        # Adapter calls: build the canonical prompt string via string concatenation
        # and call /v1/completions.  This guarantees the prompt is byte-for-byte
        # identical to the training format ([SYSTEM]\n…\n\n[USER]\n…\n\n[RESPONSE]\n).
        # Base-model calls keep /v1/chat/completions with the model's native template.
        if using_adapter:
            payload = {
                "model":       effective_model,
                "prompt":      _build_adapter_prompt(messages),
                "max_tokens":  request.max_tokens,
                "temperature": request.temperature,
                "top_p":       request.top_p,
                "stop":        _ADAPTER_STOP_TOKENS,
            }
            resp = await self._client.post(
                f"{base_url}/v1/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
            content = _fix_bpe_artifacts(data["choices"][0]["text"])
        else:
            payload = {
                "model":       effective_model,
                "messages":    messages,
                "max_tokens":  request.max_tokens,
                "temperature": request.temperature,
                "top_p":       request.top_p,
            }
            resp = await self._client.post(
                f"{base_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
            content = _fix_bpe_artifacts(data["choices"][0]["message"]["content"])
        tokens_out = data.get("usage", {}).get("completion_tokens", 0)

        logger.info(
            "LocalVLLM ← role=%s  family=%s  model=%s  tokens=%d",
            request.role, family, effective_model, tokens_out,
        )

        return GenerateResponse(
            content=content,
            model_family=family,
            role=request.role,
            tokens_generated=tokens_out,
            session_id=request.session_id,
        )

    async def generate_stream(
        self, request: GenerateRequest
    ) -> AsyncGenerator[str, None]:
        """
        Stream tokens from the vLLM server using its OpenAI-compatible SSE
        format (``stream=true`` on ``/v1/chat/completions``).

        Each SSE line is ``data: <JSON>`` where the JSON has:
            choices[0].delta.content  -- the new token fragment
        The stream ends with ``data: [DONE]``.
        """
        family = request.model_family.lower()
        if family not in _FAMILY_MAP:
            raise ValueError(
                f"Unknown model_family '{family}'. Supported: {list(_FAMILY_MAP)}"
            )

        base_url, model_id = _FAMILY_MAP[family]
        registry = _REGISTRIES[family]
        effective_model = await _resolve_model(
            model_id, request.adapter_name, registry, request.role
        )

        messages = [m.model_dump() for m in request.messages]
        using_adapter = effective_model != model_id

        # Adapter calls use /v1/completions with the canonical prompt string so
        # the input matches the training format byte-for-byte.  Base-model fallback
        # (no adapter for this role yet) keeps /v1/chat/completions with the
        # model's native chat template.  Once every role has a trained adapter
        # this else-branch can be removed to unify on /v1/completions.
        if using_adapter:
            stream_payload: dict = {
                "model":       effective_model,
                "prompt":      _build_adapter_prompt(messages),
                "max_tokens":  request.max_tokens,
                "temperature": request.temperature,
                "top_p":       request.top_p,
                "stop":        _ADAPTER_STOP_TOKENS,
                "stream":      True,
            }
            stream_endpoint = f"{base_url}/v1/completions"
        else:
            stream_payload = {
                "model":       effective_model,
                "messages":    messages,
                "max_tokens":  request.max_tokens,
                "temperature": request.temperature,
                "top_p":       request.top_p,
                "stream":      True,
            }
            stream_endpoint = f"{base_url}/v1/chat/completions"

        async with self._client.stream(
            "POST",
            stream_endpoint,
            json=stream_payload,
            headers={"Content-Type": "application/json"},
        ) as resp:
            resp.raise_for_status()
            async for raw_line in resp.aiter_lines():
                if not raw_line.startswith("data: "):
                    continue
                data = raw_line[6:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if using_adapter:
                    token = chunk.get("choices", [{}])[0].get("text", "")
                else:
                    token = (
                        chunk.get("choices", [{}])[0]
                        .get("delta", {})
                        .get("content", "")
                    )
                if token:
                    yield _fix_bpe_artifacts(token)


async def _resolve_model(
    model_id: str,
    adapter_name: Optional[str],
    registry: _AdapterRegistry,
    role: str,
) -> str:
    """
    Return the model string to send to vLLM.

    - No adapter requested  → base model ID
    - Adapter requested + available in vLLM  → "{model_id}/{adapter_name}"
    - Adapter requested but NOT yet registered  → base model ID (fallback, warning logged)
    """
    if not adapter_name:
        return model_id

    if await registry.adapter_available(adapter_name):
        return adapter_name  # vLLM uses the bare adapter name as the model field

    logger.warning(
        "Adapter '%s' for role '%s' is not registered with vLLM — "
        "falling back to base model '%s'. "
        "The Training Service will produce it once enough experiences accumulate.",
        adapter_name, role, model_id,
    )
    return model_id
