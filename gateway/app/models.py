"""
Gateway service — request/response models.

All Pydantic shapes specific to the gateway HTTP API live here.
Cross-service contracts belong in shared/contracts/.
"""

from typing import Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    prompt: str = Field(..., description="User's natural-language request")
    session_id: Optional[str] = Field(None, description="Optional session identifier for continuity")
    task_id: Optional[str] = Field(None, description="Stable identifier scoping multi-turn context state. Defaults to session_id when absent.")
    suppress_few_shots: bool = False
    max_iterations: Optional[int] = Field(None, description="Override default max iterations (1–10)")
    convergence_threshold: Optional[float] = Field(None, description="Override convergence score threshold (0.0–1.0)")


class ChatResponse(BaseModel):
    session_id: str
    output: str
    success: bool
    confidence: float
    iterations: int
    converged: bool
