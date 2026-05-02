"""Plan domain models for orchestration/app/plans."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Literal, Optional

from pydantic import BaseModel, field_validator, model_validator


class PlanStep(BaseModel):
    id: str
    description: str
    status: Literal["pending", "in_progress", "done", "blocked"] = "pending"
    owner_role: Optional[str] = None
    evidence: str = ""
    files_touched: List[str] = []
    tools_used: List[str] = []
    depends_on: List[str] = []


class Plan(BaseModel):
    goal: str
    steps: List[PlanStep]
    created_at: datetime
    updated_at: datetime

    @model_validator(mode="after")
    def _unique_step_ids(self) -> "Plan":
        ids = [s.id for s in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate step ids: {ids}")
        return self
