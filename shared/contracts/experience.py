"""
Experience Service contracts
=============================
Shapes used when Orchestration Service publishes to the Experience Service,
and when the Experience Service signals the Training Service.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_serializer

SCORER_RULE_VERSION: str = "1"


class AgentContribution(BaseModel):
    """One agent's contribution within a completed run."""

    role: str = Field(description="Agent role, e.g. 'coder'.")
    output: str = Field(description="Raw text output from this agent.")
    quality_score: float = Field(
        description="Per-agent quality score (0–1) assigned by the scoring engine."
    )
    iteration: int = Field(description="Iteration this contribution belongs to.")
    messages: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description=(
            "Full message list sent to the LLM for this agent step "
            "(role/content dicts). Together with `output` this forms a complete "
            "supervised fine-tuning example for future from-scratch training."
        ),
    )
    tool_turns: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description=(
            "Interleaved tool-call/tool-result message dicts emitted during this "
            "agent's ReAct loop. Each entry is an OpenAI-format message dict "
            "(role='assistant' with tool_calls, or role='tool' with tool_call_id). "
            "None for agents that did not invoke any tools."
        ),
    )


class ExperienceEvent(BaseModel):
    """
    Published by Orchestration Service after every completed run.
    Consumed by Experience Service to persist structured data + embeddings.

    Both converged and non-converged runs are captured so the full quality
    spectrum is available for future training (contrastive learning needs
    both good and poor examples).
    """

    session_id: str = Field(description="Loop session identifier for correlation.")
    prompt: str = Field(description="Original user prompt.")
    final_output: str = Field(description="Best output produced by the pipeline.")
    agent_chain: List[str] = Field(description="Ordered list of agent roles that ran.")
    contributions: List[AgentContribution] = Field(
        description="Per-agent outputs, scores, and full message context."
    )
    final_score: float = Field(description="Best quality score achieved (0–1).")
    iterations: int = Field(description="Total number of iterations that ran.")
    converged: bool = Field(
        description="True if quality threshold was met (not just max-iterations exit)."
    )
    iteration_scores: List[float] = Field(
        default_factory=list,
        description="Quality score recorded after each complete iteration.",
    )
    intent: Optional[str] = Field(
        default=None,
        description="Query intent classification (e.g. 'coding', 'research').",
    )
    complexity: Optional[str] = Field(
        default=None,
        description="Query complexity classification (e.g. 'simple', 'complex').",
    )
    timestamp: Optional[datetime] = Field(default=None)
    scorer_rule_version: Optional[str] = Field(
        default=None,
        description="Scorer rule version active when this record was scored. Used for cohort filtering during training.",
    )

    @field_serializer("timestamp")
    def _serialize_timestamp(self, v: Optional[datetime]) -> Optional[str]:  # noqa: PLR6301
        return v.isoformat() if v is not None else None
