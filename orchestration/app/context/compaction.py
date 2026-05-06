"""
Context compaction — two-tier strategy for keeping prompts within token budget.

Tier 1 (cheap): truncate_tool_outputs — replaces large tool-output messages with
                a hash stub. In-memory only; Redis originals untouched.
Tier 2 (LLM):   summarize_oldest_turns — rewrites the oldest N turns as a single
                <SUMMARY> system message via the summarizer role.
"""
from __future__ import annotations

import hashlib


def _msg_role(msg) -> str:
    return msg.role if hasattr(msg, "role") else msg.get("role", "")


def _msg_content(msg) -> str:
    c = msg.content if hasattr(msg, "content") else msg.get("content", "")
    return c or ""


def _msg_with_content(msg, new_content: str):
    if hasattr(msg, "model_copy"):
        return msg.model_copy(update={"content": new_content})
    return {**msg, "content": new_content}


def truncate_tool_outputs(
    messages: list,
    *,
    threshold_tokens: int,
    retain_recent_n: int,
) -> tuple[list, int, int]:
    """
    Replace oversized tool-output messages with a hash stub.

    Walks messages; for each role:tool entry whose estimated token count exceeds
    threshold_tokens AND whose index is older than the last retain_recent_n
    non-tool messages, replaces content with '<TRUNCATED — N bytes — sha256={hash}>'.

    Returns:
        (new_messages, n_truncated, bytes_saved)
    """
    from ..config.settings import settings

    non_tool_indices = [i for i, m in enumerate(messages) if _msg_role(m) != "tool"]
    cutoff_idx = (
        non_tool_indices[-retain_recent_n]
        if len(non_tool_indices) >= retain_recent_n
        else 0
    )

    new_messages = []
    n_truncated = 0
    bytes_saved = 0

    for i, msg in enumerate(messages):
        if _msg_role(msg) == "tool" and i < cutoff_idx:
            content = _msg_content(msg)
            content_bytes = len(content.encode("utf-8"))
            est_tokens = len(content) // max(settings.chars_per_token, 1)
            if est_tokens > threshold_tokens:
                digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
                stub = f"<TRUNCATED — {content_bytes} bytes — sha256={digest}>"
                new_messages.append(_msg_with_content(msg, stub))
                n_truncated += 1
                bytes_saved += content_bytes - len(stub.encode("utf-8"))
                continue
        new_messages.append(msg)

    return new_messages, n_truncated, bytes_saved


async def summarize_oldest_turns(
    messages: list,
    *,
    n_to_summarize: int,
) -> list:
    """
    Replace the oldest n_to_summarize turns with a single SUMMARY system message.

    Invokes the summarizer role via the inference client (role='summarizer',
    adapter from role_adapter['summarizer']).  Caches the result in Redis so
    identical windows are never re-summarized.

    Returns the new messages list with the summary prepended and old turns removed.

    TODO (Phase 14.5): if summarizer latency exceeds p95_latency_budget_s, move this
    call to a background refresher and read cached summaries at prompt-assembly time.
    """
    import hashlib
    import json
    import time
    from datetime import datetime, timezone

    from ..config.settings import settings
    from ..clients.inference import get_inference_client
    from ..db import redis as _redis
    from shared.contracts.inference import GenerateRequest, Message

    turns_to_summarize = messages[:n_to_summarize]
    remaining = messages[n_to_summarize:]

    # Build cache key from content hash of the turns being summarized
    turns_json = json.dumps([_msg_content(m) for m in turns_to_summarize], sort_keys=True)
    content_hash = hashlib.sha256(turns_json.encode("utf-8")).hexdigest()[:24]
    cache_key = f"summary:v1:{content_hash}:summarizer_lora"

    cached = await _redis.get_summary(cache_key)
    if cached and cached.get("compacted"):
        summary_text = cached.get("summary", "")
    else:
        # Build summarizer request
        turns_text = "\n---\n".join(
            f"[{_msg_role(m).upper()}]: {_msg_content(m)}" for m in turns_to_summarize
        )
        summarizer_messages = [
            Message(
                role="system",
                content=(
                    "Summarise the following conversation turns as bullet points. "
                    "Do not invent facts. Keep under 300 tokens."
                ),
            ),
            Message(role="user", content=turns_text),
        ]

        t0 = time.monotonic()
        client = get_inference_client()
        request = GenerateRequest(
            backend=settings.role_backend.get("summarizer", "primary"),
            role="summarizer",
            messages=summarizer_messages,
            adapter_name=settings.role_adapter.get("summarizer"),
            max_tokens=settings.role_max_tokens.get("summarizer", 512),
            session_id="compaction",
        )
        response = await client.generate(request)
        elapsed = time.monotonic() - t0
        summary_text = response.output if hasattr(response, "output") else str(response)

        # TODO (Phase 14.5 latency gate): if elapsed > settings.p95_latency_budget_s,
        # log a warning and flag that summarization should be moved to a background refresher.
        if elapsed > settings.p95_latency_budget_s:
            import logging
            logging.getLogger(__name__).warning(
                "summarize_oldest_turns latency %.1fs exceeds budget %.1fs — "
                "consider moving to background refresh (Phase 14.5 TODO)",
                elapsed, settings.p95_latency_budget_s,
            )

        summary_entry = {
            "compacted": True,
            "covers_turns": [0, n_to_summarize],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "summary": summary_text,
        }
        ttl = settings.compaction_retention_days * 86400
        await _redis.save_summary(cache_key, summary_entry, ttl)

    summary_msg = Message(
        role="system",
        content=f"<SUMMARY of {n_to_summarize} prior turns>\n{summary_text}\n</SUMMARY>",
    )
    return [summary_msg] + remaining
