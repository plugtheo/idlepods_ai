"""
Tests for the token queue registry and streaming path in nodes.py.

Covers
------
- register_token_queue: stores queue keyed by session_id
- unregister_token_queue: removes queue (no-op when already absent)
- get_token_queue: returns queue when registered, None when not
- _run_agent_node (streaming path): when queue registered + client has
  generate_stream, token events are put on the queue and output is accumulated
- _run_agent_node (streaming path): error mid-stream falls back to partial
  tokens; output is still appended to history
- _run_agent_node (blocking path): uses generate() when no queue is registered
- _run_agent_node (blocking path): uses generate() when client lacks
  generate_stream attribute
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from shared.contracts.inference import GenerateResponse


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _base_state(**overrides):
    state = {
        "session_id": "test-sess",
        "user_prompt": "write a binary search",
        "agent_chain": ["coder"],
        "agent_chain_index": 0,
        "few_shots": [],
        "repo_snippets": [],
        "system_hints": "",
        "current_iteration": 1,
        "max_iterations": 3,
        "convergence_threshold": 0.85,
        "iteration_history": [],
        "conversation_history": [],
        "last_output": "",
        "iteration_scores": [],
        "best_score": 0.0,
        "best_output": "",
        "converged": False,
        "quality_converged": False,
        "final_output": "",
        "final_score": 0.0,
        "pending_tool_calls": [],
        "tool_steps_used": 0,
        "tool_originating_role": "",
    }
    state.update(overrides)
    return state


def _blocking_client(content="generated content"):
    client = MagicMock()
    client.generate = AsyncMock(
        return_value=GenerateResponse(
            content=content,
            model_family="qwen",
            role="coder",
            tokens_generated=5,
        )
    )
    return client


def _streaming_client(*tokens):
    """Client that has both generate() and generate_stream()."""
    client = MagicMock()
    client.generate = AsyncMock(
        return_value=GenerateResponse(
            content="".join(tokens),
            model_family="qwen",
            role="coder",
            tokens_generated=len(tokens),
        )
    )

    async def _stream(req):
        for t in tokens:
            yield t

    client.generate_stream = MagicMock(side_effect=_stream)
    return client


# ─── Token queue registry ─────────────────────────────────────────────────────

class TestTokenQueueRegistry:
    def setup_method(self):
        # Import fresh each time to ensure clean state
        from services.orchestration.app.graph import nodes as _nodes
        self.nodes = _nodes
        # Clear any leftover queues from other tests
        self.nodes._token_queues.clear()

    def test_register_stores_queue(self):
        q = asyncio.Queue()
        self.nodes.register_token_queue("sess-a", q)
        assert self.nodes._token_queues["sess-a"] is q

    def test_get_returns_registered_queue(self):
        q = asyncio.Queue()
        self.nodes.register_token_queue("sess-b", q)
        result = self.nodes.get_token_queue("sess-b")
        assert result is q

    def test_get_returns_none_when_not_registered(self):
        result = self.nodes.get_token_queue("nonexistent-sess")
        assert result is None

    def test_unregister_removes_queue(self):
        q = asyncio.Queue()
        self.nodes.register_token_queue("sess-c", q)
        self.nodes.unregister_token_queue("sess-c")
        assert self.nodes.get_token_queue("sess-c") is None

    def test_unregister_is_noop_when_not_registered(self):
        # Should not raise
        self.nodes.unregister_token_queue("not-there")

    def test_multiple_sessions_independent(self):
        q1 = asyncio.Queue()
        q2 = asyncio.Queue()
        self.nodes.register_token_queue("s1", q1)
        self.nodes.register_token_queue("s2", q2)
        assert self.nodes.get_token_queue("s1") is q1
        assert self.nodes.get_token_queue("s2") is q2
        self.nodes.unregister_token_queue("s1")
        assert self.nodes.get_token_queue("s1") is None
        assert self.nodes.get_token_queue("s2") is q2


# ─── _run_agent_node (streaming path) ─────────────────────────────────────────
# Streaming is only triggered for non-tool-using roles (reviewer, planner, etc.).
# Coder is a tool-using role and always uses the blocking path — see blocking tests.

@pytest.mark.asyncio
class TestRunAgentNodeStreamingPath:
    def setup_method(self):
        from services.orchestration.app.graph import nodes as _nodes
        self.nodes = _nodes
        self.nodes._token_queues.clear()

    async def test_token_events_put_on_queue(self):
        q: asyncio.Queue = asyncio.Queue()
        self.nodes.register_token_queue("test-sess", q)

        client = _streaming_client("Hello", " world")
        state = _base_state(session_id="test-sess")

        with patch(
            "services.orchestration.app.graph.nodes.get_inference_client",
            return_value=client,
        ):
            delta = await self.nodes.reviewer_node(state)

        events = []
        while not q.empty():
            events.append(q.get_nowait())

        token_events = [e for e in events if e.get("type") == "chunk"]
        assert len(token_events) == 2
        assert token_events[0]["content"] == "Hello"
        assert token_events[1]["content"] == " world"

    async def test_output_accumulated_from_tokens(self):
        q: asyncio.Queue = asyncio.Queue()
        self.nodes.register_token_queue("test-sess", q)

        client = _streaming_client("def ", "binary_search", "():", " pass")
        state = _base_state(session_id="test-sess")

        with patch(
            "services.orchestration.app.graph.nodes.get_inference_client",
            return_value=client,
        ):
            delta = await self.nodes.reviewer_node(state)

        assert delta["last_output"] == "def binary_search(): pass"

    async def test_history_entry_appended_with_streamed_output(self):
        q: asyncio.Queue = asyncio.Queue()
        self.nodes.register_token_queue("test-sess", q)

        client = _streaming_client("result")
        state = _base_state(session_id="test-sess")

        with patch(
            "services.orchestration.app.graph.nodes.get_inference_client",
            return_value=client,
        ):
            delta = await self.nodes.reviewer_node(state)

        history = delta["iteration_history"]
        assert len(history) == 1
        assert history[0]["role"] == "reviewer"
        assert history[0]["iteration"] == 1

    async def test_stream_error_uses_partial_tokens(self):
        """If streaming fails mid-way, partial accumulated tokens are used."""
        q: asyncio.Queue = asyncio.Queue()
        self.nodes.register_token_queue("test-sess", q)

        async def _bad_stream(req):
            yield "partial"
            raise RuntimeError("connection reset")

        client = MagicMock()
        client.generate_stream = MagicMock(side_effect=_bad_stream)
        client.generate = AsyncMock(
            return_value=GenerateResponse(
                content="fallback",
                model_family="qwen",
                role="reviewer",
                tokens_generated=1,
            )
        )
        state = _base_state(session_id="test-sess")

        with patch(
            "services.orchestration.app.graph.nodes.get_inference_client",
            return_value=client,
        ):
            delta = await self.nodes.reviewer_node(state)

        assert "partial" in delta["last_output"]

    async def test_generate_stream_not_called_when_not_registered(self):
        """When no queue is registered, generate_stream must NOT be called."""
        client = _streaming_client("text")
        state = _base_state(session_id="unregistered-sess")

        with patch(
            "services.orchestration.app.graph.nodes.get_inference_client",
            return_value=client,
        ):
            await self.nodes.reviewer_node(state)

        client.generate_stream.assert_not_called()
        client.generate.assert_awaited_once()

    async def test_cleanup_after_unregister(self):
        """After unregistering the queue, the next call uses blocking path."""
        q: asyncio.Queue = asyncio.Queue()
        self.nodes.register_token_queue("test-sess", q)
        self.nodes.unregister_token_queue("test-sess")

        client = _streaming_client("text")
        state = _base_state(session_id="test-sess")

        with patch(
            "services.orchestration.app.graph.nodes.get_inference_client",
            return_value=client,
        ):
            await self.nodes.reviewer_node(state)

        client.generate_stream.assert_not_called()
        client.generate.assert_awaited_once()


# ─── _run_agent_node (blocking path) ──────────────────────────────────────────

@pytest.mark.asyncio
class TestRunAgentNodeBlockingPath:
    def setup_method(self):
        from services.orchestration.app.graph import nodes as _nodes
        self.nodes = _nodes
        self.nodes._token_queues.clear()

    async def test_uses_blocking_generate_when_no_queue(self):
        client = _blocking_client("the answer")
        state = _base_state(session_id="no-queue-sess")

        with patch(
            "services.orchestration.app.graph.nodes.get_inference_client",
            return_value=client,
        ):
            delta = await self.nodes.coder_node(state)

        client.generate.assert_awaited_once()
        assert delta["last_output"] == "the answer"

    async def test_uses_blocking_generate_when_client_lacks_stream(self):
        """Falls back to blocking if client has no generate_stream attribute."""
        client = MagicMock()
        del client.generate_stream  # ensure hasattr returns False
        client.generate = AsyncMock(
            return_value=GenerateResponse(
                content="no-stream result",
                model_family="qwen",
                role="reviewer",
                tokens_generated=3,
            )
        )

        q: asyncio.Queue = asyncio.Queue()
        self.nodes.register_token_queue("no-stream-sess", q)

        state = _base_state(session_id="no-stream-sess")

        with patch(
            "services.orchestration.app.graph.nodes.get_inference_client",
            return_value=client,
        ):
            delta = await self.nodes.reviewer_node(state)

        client.generate.assert_awaited_once()
        assert delta["last_output"] == "no-stream result"

    async def test_coder_always_uses_blocking_even_with_queue(self):
        """Coder is a tool-using role — it always uses the blocking path."""
        q: asyncio.Queue = asyncio.Queue()
        self.nodes.register_token_queue("coder-sess", q)

        client = _streaming_client("def f(): pass")
        state = _base_state(session_id="coder-sess")

        with patch(
            "services.orchestration.app.graph.nodes.get_inference_client",
            return_value=client,
        ):
            await self.nodes.coder_node(state)

        client.generate_stream.assert_not_called()
        client.generate.assert_awaited_once()

    async def test_history_entry_appended_on_blocking_path(self):
        client = _blocking_client("block output")
        state = _base_state(session_id="block-sess")

        with patch(
            "services.orchestration.app.graph.nodes.get_inference_client",
            return_value=client,
        ):
            delta = await self.nodes.coder_node(state)

        history = delta["iteration_history"]
        assert len(history) == 1
        assert history[0]["role"] == "coder"
        assert history[0]["full_output"] == "block output"

    async def test_agent_chain_index_incremented(self):
        client = _blocking_client()
        state = _base_state(session_id="idx-sess", agent_chain_index=3)

        with patch(
            "services.orchestration.app.graph.nodes.get_inference_client",
            return_value=client,
        ):
            delta = await self.nodes.coder_node(state)

        assert delta["agent_chain_index"] == 4

    async def test_inference_failure_produces_placeholder_output(self):
        client = MagicMock()
        client.generate = AsyncMock(side_effect=RuntimeError("service down"))

        state = _base_state(session_id="fail-sess")

        with patch(
            "services.orchestration.app.graph.nodes.get_inference_client",
            return_value=client,
        ):
            delta = await self.nodes.coder_node(state)

        # Should not raise; placeholder output should reference the role
        assert "coder" in delta["last_output"]
        assert "iteration_history" in delta
