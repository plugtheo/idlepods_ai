"""
Pydantic v2 schema for manifest.json — v2 only.
Readers must reject schema_version > 2; writers always emit v2.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class HistoryEntry(BaseModel):
    version: str
    status: Literal["staging", "active", "retired", "failed"]
    trained_at: datetime
    backend: str
    base_model: str
    peft_type: str
    target_modules: List[str]
    r: int
    alpha: int
    dropout: float
    quantization: Optional[str] = None
    recipe: Dict[str, Any] = Field(default_factory=dict)
    dataset_hash: str
    tokenizer_hash: str
    trainer_version: str
    n_samples: int
    final_loss: float
    size_mb: float
    eval_metrics: Dict[str, float] = Field(default_factory=dict)
    smoke: Dict[str, Any] = Field(default_factory=dict)
    used_base_fallback_aggregate: float = 0.0


class AdapterEntry(BaseModel):
    schema_version: int = 2
    active_version: str
    active_path: str
    previous_version: str = ""
    previous_path: str = ""
    backend: str
    updated_at: datetime
    history: List[HistoryEntry]


class Manifest(BaseModel):
    schema_version: int = 2
    updated_at: datetime
    adapters: Dict[str, AdapterEntry] = Field(default_factory=dict)
