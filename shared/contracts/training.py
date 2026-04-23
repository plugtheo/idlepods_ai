"""
Training Service contracts
===========================
Shapes used when the Experience Service notifies the Training Service
that a capability may have crossed the training threshold.
"""

from typing import Optional

from pydantic import BaseModel, Field


class TrainingTriggerRequest(BaseModel):
    """
    Sent from Experience Service → Training Service after a new experience
    is stored.  The Training Service decides whether the threshold is met
    and starts a training job if so.
    """

    capability: str = Field(
        description="Agent capability to evaluate for training, e.g. 'coding'."
    )
    new_experience_count: int = Field(
        description=(
            "Number of new experiences that prompted this trigger call. "
            "Currently always sent as 1 (one experience per call). "
            "The Training Service does not use this field for decisions — "
            "it loads and counts experiences itself."
        )
    )
    session_id: Optional[str] = Field(
        default=None, description="Session that produced the triggering experience."
    )


class TrainingTriggerResponse(BaseModel):
    """Response from Training Service after evaluating a trigger request."""

    capability: str
    triggered: bool = Field(
        description="True if a training job was started."
    )
    reason: str = Field(
        description="Human-readable explanation of why training was or was not triggered."
    )
