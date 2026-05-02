"""
Orchestration Service contracts
================================
Shapes used by the API Gateway when calling the Orchestration Service,
and returned back to the end user.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class OrchestrationRequest(BaseModel):
    """Request sent from Gateway → Orchestration Service."""

    prompt: str = Field(description="Raw user prompt.")
    agent_chain: Optional[List[str]] = Field(
        default=None,
        description=(
            "Explicit ordered list of agent roles to run. "
            "Set to None to let the QueryRouter decide."
        ),
    )
    intent: Optional[str] = Field(
        default=None,
        description="Intent classification already performed by the gateway QueryRouter. "
                    "When set, the orchestration service skips its own routing pass.",
    )
    complexity: Optional[str] = Field(
        default=None,
        description="Complexity classification already performed by the gateway QueryRouter.",
    )
    max_iterations: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Maximum number of agent-chain iterations before forced stop. Defaults to service setting when None.",
    )
    convergence_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Quality score (0–1) at which the loop is considered converged. Defaults to service setting when None.",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Opaque string for log correlation. Auto-generated if not supplied.",
    )
    task_id: Optional[str] = Field(
        default=None,
        description="Stable identifier scoping multi-turn context state. Falls back to session_id when absent.",
    )
    allowed_files: Optional[List[str]] = Field(
        default=None,
        description="Repo-relative POSIX paths the context builder is permitted to scan. None means unrestricted.",
    )
    plan_path: Optional[str] = Field(
        default="plans/current-task.md",
        description=(
            "Repo-relative path to the markdown plan file. "
            "When task_id is None the plan is ephemeral (read-only, not persisted)."
        ),
    )


class AgentStep(BaseModel):
    """Record of one agent run within the pipeline."""

    role: str = Field(description="Agent role that ran, e.g. 'coder'.")
    iteration: int = Field(description="Which iteration this step belongs to (1-based).")
    output_summary: str = Field(
        description="First 300 chars of the agent's output (full output in final_output)."
    )
    score: float = Field(description="Quality score for this step (0–1).")


class OrchestrationResponse(BaseModel):
    """Response returned from Orchestration Service → Gateway → User."""

    session_id: str = Field(description="Unique identifier for this request.")
    output: str = Field(description="Final generated output for the user.")
    success: bool = Field(description="True if the pipeline converged or completed.")
    confidence: float = Field(description="Final quality score (0–1).")
    iterations: int = Field(description="Number of agent-chain iterations that ran.")
    best_score: float = Field(description="Highest score achieved across all iterations.")
    agent_steps: List[AgentStep] = Field(
        default_factory=list,
        description="Per-step trace for debugging and observability.",
    )
    converged: bool = Field(
        description="True if quality threshold was reached before max_iterations."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extras (route decision, timing, etc.) for observability.",
    )
    history_volatile: bool = Field(
        default=False,
        description="True when Redis was unavailable at request time; conversation history for this turn may be incomplete.",
    )
