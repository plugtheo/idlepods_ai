"""
Tests for all shared contract Pydantic schemas.

Covers:
- Valid construction and field defaults
- Validation errors for invalid/missing required fields
- Edge cases (empty lists, optional fields)
"""
import pytest
from pydantic import ValidationError


# ── Inference contracts ────────────────────────────────────────────────────


class TestMessage:
    def test_valid_message(self):
        from shared.contracts.inference import Message
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_missing_role_raises(self):
        from shared.contracts.inference import Message
        with pytest.raises(ValidationError):
            Message(content="Hello")

    def test_missing_content_raises(self):
        from shared.contracts.inference import Message
        with pytest.raises(ValidationError):
            Message(role="user")


class TestGenerateRequest:
    def test_minimal_valid_request(self):
        from shared.contracts.inference import GenerateRequest, Message
        req = GenerateRequest(
            model_family="deepseek",
            role="coder",
            messages=[Message(role="user", content="Write a function")],
        )
        assert req.model_family == "deepseek"
        assert req.adapter_name is None
        assert req.max_tokens == 1024
        assert req.temperature == 0.2

    def test_with_adapter(self):
        from shared.contracts.inference import GenerateRequest, Message
        req = GenerateRequest(
            model_family="mistral",
            role="planner",
            messages=[Message(role="user", content="Plan this")],
            adapter_name="planning_lora",
        )
        assert req.adapter_name == "planning_lora"

    def test_temperature_bounds(self):
        from shared.contracts.inference import GenerateRequest, Message
        with pytest.raises(ValidationError):
            GenerateRequest(
                model_family="deepseek",
                role="coder",
                messages=[Message(role="user", content="x")],
                temperature=3.0,  # above max 2.0
            )

    def test_max_tokens_bounds(self):
        from shared.contracts.inference import GenerateRequest, Message
        with pytest.raises(ValidationError):
            GenerateRequest(
                model_family="deepseek",
                role="coder",
                messages=[Message(role="user", content="x")],
                max_tokens=0,  # below min 1
            )


class TestGenerateResponse:
    def test_valid_response(self):
        from shared.contracts.inference import GenerateResponse
        resp = GenerateResponse(
            content="def hello(): pass",
            model_family="deepseek",
            role="coder",
        )
        assert resp.tokens_generated == 0
        assert resp.session_id is None

    def test_with_all_fields(self):
        from shared.contracts.inference import GenerateResponse
        resp = GenerateResponse(
            content="result",
            model_family="mistral",
            role="planner",
            tokens_generated=42,
            session_id="sess-123",
        )
        assert resp.tokens_generated == 42
        assert resp.session_id == "sess-123"


# ── Context contracts ──────────────────────────────────────────────────────


class TestContextRequest:
    def test_valid(self):
        from shared.contracts.context import ContextRequest
        req = ContextRequest(
            prompt="How do I sort a list?",
            intent="coding",
            complexity="simple",
        )
        assert req.session_id is None

    def test_with_session(self):
        from shared.contracts.context import ContextRequest
        req = ContextRequest(
            prompt="debug this",
            intent="debugging",
            complexity="moderate",
            session_id="s1",
        )
        assert req.session_id == "s1"


class TestBuiltContext:
    def test_empty_context(self):
        from shared.contracts.context import BuiltContext
        ctx = BuiltContext(few_shots=[], repo_snippets=[], system_hints="")
        assert ctx.few_shots == []
        assert ctx.repo_snippets == []

    def test_with_few_shots(self):
        from shared.contracts.context import BuiltContext, FewShotExample
        shot = FewShotExample(problem="q", solution="a", score=0.9, category="coding")
        ctx = BuiltContext(few_shots=[shot], repo_snippets=[], system_hints="use types")
        assert len(ctx.few_shots) == 1
        assert ctx.few_shots[0].score == 0.9


# ── Orchestration contracts ────────────────────────────────────────────────


class TestOrchestrationRequest:
    def test_defaults(self):
        from shared.contracts.orchestration import OrchestrationRequest
        req = OrchestrationRequest(prompt="build a REST API")
        assert req.max_iterations is None
        assert req.convergence_threshold is None
        assert req.agent_chain is None
        assert req.session_id is None

    def test_custom_chain(self):
        from shared.contracts.orchestration import OrchestrationRequest
        req = OrchestrationRequest(
            prompt="fix bug",
            agent_chain=["debugger", "reviewer"],
            max_iterations=3,
        )
        assert req.agent_chain == ["debugger", "reviewer"]

    def test_missing_prompt_raises(self):
        from shared.contracts.orchestration import OrchestrationRequest
        with pytest.raises(ValidationError):
            OrchestrationRequest()


class TestOrchestrationResponse:
    def test_valid(self):
        from shared.contracts.orchestration import OrchestrationResponse
        resp = OrchestrationResponse(
            session_id="s1",
            output="done",
            success=True,
            confidence=0.9,
            iterations=2,
            best_score=0.9,
            converged=True,
            agent_steps=[],
        )
        assert resp.success is True


# ── Experience contracts ───────────────────────────────────────────────────


class TestExperienceEvent:
    def test_valid_event(self):
        from shared.contracts.experience import ExperienceEvent, AgentContribution
        event = ExperienceEvent(
            session_id="sess-1",
            prompt="do X",
            final_output="done X",
            agent_chain=["coder", "reviewer"],
            contributions=[
                AgentContribution(role="coder", output="code", quality_score=0.8, iteration=1),
                AgentContribution(role="reviewer", output="SCORE: 0.85", quality_score=0.85, iteration=1),
            ],
            final_score=0.85,
            iterations=1,
            converged=True,
        )
        assert event.timestamp is None
        assert len(event.contributions) == 2

    def test_agent_contribution_score_range(self):
        from shared.contracts.experience import AgentContribution
        contrib = AgentContribution(role="coder", output="code", quality_score=0.5, iteration=1)
        assert contrib.quality_score == 0.5


# ── Training contracts ─────────────────────────────────────────────────────


class TestTrainingContracts:
    def test_trigger_request(self):
        from shared.contracts.training import TrainingTriggerRequest
        req = TrainingTriggerRequest(capability="coding", new_experience_count=55)
        assert req.session_id is None

    def test_trigger_response_triggered(self):
        from shared.contracts.training import TrainingTriggerResponse
        resp = TrainingTriggerResponse(
            capability="coding",
            triggered=True,
            reason="criteria met: n=55, spread=0.30, diversity=0.75",
        )
        assert resp.triggered is True

    def test_trigger_response_not_triggered(self):
        from shared.contracts.training import TrainingTriggerResponse
        resp = TrainingTriggerResponse(
            capability="coding",
            triggered=False,
            reason="too few experiences: 10 < 50",
        )
        assert resp.triggered is False
